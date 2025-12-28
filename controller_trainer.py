#!/usr/bin/env python3
"""
Standalone Controller Trainer for gpt-oss MoE models.

This module implements REINFORCE-style training for the controller,
treating it as an RNN-like policy that makes sequential decisions
across tokens and layers.

Key design principles:
1. Controller is trained independently from the LLM
2. LLM is only used for rollout generation (with torch.no_grad())
3. Controller update uses only recorded data - no LLM forward pass
4. Value baseline is integrated into the controller for variance reduction

Training loop (with ADVANTAGE NORMALIZATION):
1. Collect rollouts across gradient_accumulation_steps
2. For each rollout, compute per-timestep advantages: A_t = R - V(s_t)
3. Normalize advantages globally: A'_t = (A_t - mean) / (std + eps)
4. Compute policy loss with normalized advantages: -A'_t * log π(a_t|s_t)
5. Accumulate gradients, then optimizer.step()

Why advantage normalization instead of reward normalization:
- Reward normalization: if V(s) learns to predict normalized rewards perfectly,
  advantages become zero, killing the policy gradient signal.
- Advantage normalization: advantages are normalized AFTER subtracting V(s),
  so the signal is always present regardless of how well V fits.

This follows TRL's standard practice for policy gradient methods (GRPO, RLOO).
"""

import gc
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from tqdm import tqdm

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_logprob(probs: torch.Tensor) -> torch.Tensor:
    """Compute log(p) safely, avoiding log(0)."""
    # Use a larger epsilon to avoid extreme log values
    eps = 1e-6
    return torch.log(probs.clamp(min=eps, max=1.0 - eps))


def _safe_log1m(probs: torch.Tensor) -> torch.Tensor:
    """Compute log(1-p) safely, avoiding log(0)."""
    eps = 1e-6
    return torch.log1p(-probs.clamp(min=eps, max=1.0 - eps))


def _plackett_luce_logprob_batched(
    logits: torch.Tensor,
    selections: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of selections under Plackett-Luce distribution.
    
    BATCHED VERSION: Processes all samples in parallel.
    This is equivalent to _plackett_luce_logprob but optimized for large batches.
    
    Args:
        logits: [batch, num_experts] - unnormalized log probabilities
        selections: [batch, k] - indices of selected experts in order
        
    Returns:
        log_prob: [batch] - log probability of the selection sequence
    """
    batch_size, k = selections.shape
    num_experts = logits.shape[1]
    
    # Clone logits so we can mark selected experts as -inf
    remaining = logits.clone()
    total_logprob = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
    
    # Batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=logits.device)
    
    for step in range(k):
        step_indices = selections[:, step]
        
        # Check for valid indices (>= 0)
        active = step_indices >= 0
        if not torch.any(active):
            continue
        
        # Clamp indices to valid range
        valid_indices = step_indices.clamp(min=0, max=num_experts - 1)
        
        # Compute log_softmax for ALL samples (more GPU-friendly than indexing subset)
        log_probs = F.log_softmax(remaining, dim=-1)
        
        # Gather log prob for selected expert for each sample
        gathered = log_probs[batch_indices, valid_indices]
        
        # Only add to total for active samples
        total_logprob = total_logprob + torch.where(active, gathered, torch.zeros_like(gathered))
        
        # Mark selected expert as unavailable for all active samples
        # Use advanced indexing: remaining[batch_indices[active], valid_indices[active]] = -inf
        active_batch = batch_indices[active]
        active_valid = valid_indices[active]
        remaining[active_batch, active_valid] = float("-inf")
    
    return total_logprob


def empty_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ControllerTrainerConfig:
    """Configuration for the controller trainer."""
    
    # Output
    output_dir: str = "./controller_output"
    run_name: str = "controller_rl"
    
    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    num_update_epochs: int = 1  # Number of epochs per batch (1 = fully on-policy)
    
    # Batch sizes
    per_device_train_batch_size: int = 4  # 4 samples, one per GPU
    gradient_accumulation_steps: int = 1
    
    # Generation
    response_length: int = 128
    temperature: float = 1.0
    
    # Loss coefficients
    value_coef: float = 0.1
    
    # Reward
    latency_cost_per_switch: float = 10.0  # For GRPO: penalty = cost * switch_rate
    option_critic_deliberation_cost: float = 0.1  # For Option-Critic: per-switch cost η (Harb et al. 2017)
    
    # Advantage computation method
    # - "option_critic": Option-Critic (Harb et al., 2017) with per-token TD and deliberation cost
    # - "grpo": GRPO (group-level baseline, recommended for simplicity)
    advantage_method: str = "option_critic"
    num_generations_per_prompt: int = 4  # Number of rollouts per prompt (only used for GRPO)
    gamma: float = 0.99  # Discount factor for Option-Critic TD targets
    gae_lambda: float = 0.95  # GAE lambda for bias-variance tradeoff (0=TD(0), 1=MC)
    
    # Initialization
    switch_init_bias: float = 0.0  # Initial bias for switch head (negative = less switching)
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    report_to: str = "wandb"
    wandb_project: str = "controller-rl"
    wandb_entity: Optional[str] = None
    
    # Misc
    seed: int = 42


# =============================================================================
# Recorded Actions Data Structure
# =============================================================================

@dataclass
class ControllerRollout:
    """
    Stores recorded controller actions and inputs from a rollout.
    
    For each layer, we store (canonical RNN architecture):
    - router_logits: [batch, seq_len, num_experts] - router logits at each token
    - switches: [batch, seq_len] - binary switch decisions
    - selected_indices: [batch, seq_len, k] - selected expert indices
    - values: [batch, seq_len] - value estimates from controller
    - controller_inputs: [batch, seq_len, 2*num_experts] - x_t = [router_softmax, expert_mask]
    """
    
    # Per-layer recorded data
    layer_data: Dict[int, Dict[str, torch.Tensor]]
    
    # Trajectory-level data
    queries: torch.Tensor  # [batch, query_len]
    responses: torch.Tensor  # [batch, response_len]
    rewards: torch.Tensor  # [batch] - final rewards (with latency penalty)
    base_rewards: torch.Tensor  # [batch] - quality scores (without latency penalty)
    response_lengths: torch.Tensor  # [batch] - actual response lengths (not including padding)
    pad_token_id: int  # tokenizer pad token id for computing attention mask
    
    # GRPO-specific: group IDs for samples that share the same prompt
    # If None, each sample is its own group (PPO/Option-Critic mode)
    group_ids: Optional[torch.Tensor] = None  # [batch] - group ID for each sample
    
    # Option-Critic specific: per-token KL divergence for dense rewards
    # If None, trajectory-level reward is used (PPO/GRPO mode)
    per_token_kl: Optional[torch.Tensor] = None  # [batch, response_len] - KL at each position
    
    # Terminal vs truncation flag: True = hit EOS (true terminal), False = truncated (hit max_length)
    # Used for GAE: bootstrap at truncation boundaries, not at true terminals
    terminated: Optional[torch.Tensor] = None  # [batch] - True if sequence ended with EOS
    
    @property
    def batch_size(self) -> int:
        return self.queries.shape[0]
    
    @property
    def num_layers(self) -> int:
        return len(self.layer_data)
    
    def get_total_switch_count(self) -> torch.Tensor:
        """Get total number of switches per trajectory.
        
        Only counts switches for meaningful tokens:
        - Excludes left-padding in query
        - Excludes right-padding in response (after EOS)
        """
        total = torch.zeros(self.batch_size, device=self.queries.device)
        query_len = self.queries.shape[1]
        
        # Compute attention mask for query (1 for real tokens, 0 for padding)
        query_attention_mask = (self.queries != self.pad_token_id).long()  # [batch, query_len]
        # Number of real query tokens per sample (excluding left-padding)
        real_query_lengths = query_attention_mask.sum(dim=1)  # [batch]
        # Number of left-padding tokens per sample
        left_padding_lengths = query_len - real_query_lengths  # [batch]
        
        for layer_data in self.layer_data.values():
            switches = layer_data.get("switches")
            if switches is not None:
                seq_len = switches.shape[1]
                positions = torch.arange(seq_len, device=switches.device).unsqueeze(0)  # [1, seq_len]
                
                # Valid range: [left_padding_length, query_len + response_length)
                start = left_padding_lengths.unsqueeze(1)  # [batch, 1]
                end = (query_len + self.response_lengths).unsqueeze(1)  # [batch, 1]
                mask = (positions >= start) & (positions < end)  # [batch, seq_len]
                
                # Only count switches within valid positions
                total += (switches.float() * mask.float()).sum(dim=1)
        return total
    
    def get_real_seq_length(self) -> torch.Tensor:
        """Get the real sequence length (excluding padding) per sample."""
        query_len = self.queries.shape[1]
        query_attention_mask = (self.queries != self.pad_token_id).long()
        real_query_lengths = query_attention_mask.sum(dim=1)
        return real_query_lengths + self.response_lengths


# =============================================================================
# Controller Trainer
# =============================================================================

class ControllerTrainer:
    """
    Trainer for controller-only RL.
    
    The controller is treated as an RNN-like policy that makes
    sequential decisions across tokens and layers.
    """
    
    def __init__(
        self,
        config: ControllerTrainerConfig,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        reward_fn: Callable,
        accelerator: Optional[Accelerator] = None,
        ppl_scorer: Optional[Any] = None,  # For accessing perplexity/repetition metrics
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.reward_fn = reward_fn
        self.ppl_scorer = ppl_scorer  # Store for accessing batch metrics
        
        # Initialize accelerator if not provided
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
        self.accelerator = accelerator
        
        # Prepare model and dataloader with accelerator (but NOT optimizer!)
        # We'll use a plain PyTorch optimizer for the controller - no DeepSpeed wrapping needed
        self.model, self.train_dataloader = accelerator.prepare(
            self.model, self.train_dataloader
        )
        
        # Get controller parameters AFTER prepare (from the wrapped model)
        self.controller_params = self._get_controller_params()
        if accelerator.is_main_process:
            num_params = sum(p.numel() for p in self.controller_params)
            sample_dtype = next(iter(self.controller_params)).dtype
            print(f"[CONTROLLER-TRAINER] Found {len(self.controller_params)} controller parameter tensors")
            print(f"[CONTROLLER-TRAINER] Total controller parameters: {num_params:,}")
            print(f"[CONTROLLER-TRAINER] Controller dtype BEFORE conversion: {sample_dtype}")
        
        # Convert controller parameters to float32 AFTER model loading
        # from_pretrained() with torch_dtype=bfloat16 overrides the dtype set in __init__
        # bfloat16 has poor precision (~0.02 at magnitude 3.0), causing small gradient updates to round to 0
        self._convert_controller_to_fp32()
        
        # Refresh params list and verify dtype
        self.controller_params = self._get_controller_params()
        if accelerator.is_main_process:
            sample_dtype = next(iter(self.controller_params)).dtype
            print(f"[CONTROLLER-TRAINER] Controller dtype AFTER conversion: {sample_dtype}")
        
        # Initialize switch head bias if specified
        if config.switch_init_bias != 0.0:
            self._initialize_switch_bias(config.switch_init_bias)
            if accelerator.is_main_process:
                print(f"[CONTROLLER-TRAINER] Initialized switch_head.bias to {config.switch_init_bias:.2f}")
                expected_switch_prob = 1.0 / (1.0 + math.exp(-config.switch_init_bias))
                print(f"[CONTROLLER-TRAINER] Expected initial switch probability: {expected_switch_prob:.4f} ({expected_switch_prob*100:.2f}%)")
        
        # Initialize LayerNorm parameters (they have garbage values after loading from checkpoint
        # because the checkpoint doesn't have LayerNormGRU - we just added it)
        self._initialize_layer_norm()
        
        # Debug: Check LayerNorm parameters (now inside GRU cell - 6 modules for separate input/hidden)
        if accelerator.is_main_process:
            controller = self._get_controller_module()
            gru = controller.gru_cell
            if hasattr(gru, 'ln_ri') and gru.ln_ri is not None:
                print(f"[DEBUG-LAYERNORM] GRU has 6 separate LN modules (input: ri,zi,ni; hidden: rh,zh,nh)")
                print(f"[DEBUG-LAYERNORM] ln_ri.weight mean={gru.ln_ri.weight.mean().item():.4f}, ln_rh.weight mean={gru.ln_rh.weight.mean().item():.4f}")
            else:
                print(f"[DEBUG-LAYERNORM] LayerNorm not present or disabled in GRU")
        
        # Create a PLAIN PyTorch optimizer (not wrapped by DeepSpeed)
        # This is much simpler and avoids all the DeepSpeed parameter management issues
        self.optimizer = torch.optim.AdamW(
            self.controller_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if accelerator.is_main_process:
            print(f"[CONTROLLER-TRAINER] Using plain PyTorch AdamW optimizer (not DeepSpeed wrapped)")
        
        # Generation config - DETERMINISTIC token decoding
        # Randomness should only be in controller decisions (switch/expert selection),
        # not in token generation, for clean credit assignment
        self.generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            do_sample=False,  # Greedy decoding for tokens
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize wandb (offline mode)
        self.wandb_run = None
        if accelerator.is_main_process and config.report_to == "wandb" and HAS_WANDB:
            os.environ["WANDB_MODE"] = "offline"
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                config={
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "max_grad_norm": config.max_grad_norm,
                    "num_train_epochs": config.num_train_epochs,
                    "num_update_epochs": config.num_update_epochs,
                    "per_device_train_batch_size": config.per_device_train_batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "response_length": config.response_length,
                    "token_decoding": "greedy",  # Deterministic for clean credit assignment
                    "value_coef": config.value_coef,
                    "latency_cost_per_switch": config.latency_cost_per_switch,
                    "switch_init_bias": config.switch_init_bias,
                },
            )
            print(f"[WANDB] Initialized in offline mode: {self.wandb_run.dir}")
    
    def _get_controller_params(self) -> List[nn.Parameter]:
        """Get all controller parameters from the model."""
        # Handle DeepSpeed wrapped model
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
            
        params = []
        for name, param in model.named_parameters():
            if "controller" in name and param.requires_grad:
                params.append(param)
        return params
    
    def _get_controller_for_layer(self, layer_idx: int) -> nn.Module:
        """Get the controller module for a specific layer."""
        # Get the model (handle DeepSpeed wrapping)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Handle PolicyAndValueWrapper
        if hasattr(model, 'policy'):
            policy = model.policy
        else:
            policy = model
        
        # Navigate to the specific layer's controller
        # Structure: model.model.layers[layer_idx].mlp.controller
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            layer = policy.model.layers[layer_idx]
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller'):
                if layer.mlp.controller is not None:
                    return layer.mlp.controller
        
        raise ValueError(f"Could not find controller for layer {layer_idx}")
    
    def _get_controller_sampling_temperature(self, layer_idx: int = 0) -> float:
        """Get the controller_sampling_temperature from the MLP module.
        
        This must match the temperature used during inference/rollout to ensure
        log_probs are computed correctly.
        """
        # Get the model (handle DeepSpeed wrapping)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Handle PolicyAndValueWrapper
        if hasattr(model, 'policy'):
            policy = model.policy
        else:
            policy = model
        
        # Navigate to the MLP module which stores the temperature
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            layer = policy.model.layers[layer_idx]
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller_sampling_temperature'):
                return layer.mlp.controller_sampling_temperature
        
        # Default to 1.0 if not found
        return 1.0
    
    def _get_controller_module(self) -> nn.Module:
        """Get any controller module from the model (for dtype checking etc)."""
        return self._get_controller_for_layer(0)
    
    def _convert_controller_to_fp32(self) -> None:
        """Convert all controller parameters to float32 AFTER model loading.
        
        This must be called after from_pretrained() because:
        - from_pretrained(..., torch_dtype=bfloat16) loads ALL weights in bfloat16
        - This overrides any dtype set in __init__
        
        bfloat16 has only 7 bits of mantissa, giving precision of ~0.02 at magnitude 3.0.
        This means small gradient updates (e.g., lr=1e-3 * grad=1e-3 = 1e-6) get rounded to 0.
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        policy = model
        while hasattr(policy, 'module'):
            policy = policy.module
        
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            num_converted = 0
            for layer in policy.model.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller'):
                    controller = layer.mlp.controller
                    if controller is not None:
                        # Convert each parameter to float32
                        for param in controller.parameters():
                            param.data = param.data.float()
                        num_converted += 1
            
            if self.accelerator.is_main_process:
                print(f"[FP32] Converted {num_converted} controller modules to float32")
    
    def _initialize_switch_bias(self, bias_value: float) -> None:
        """Initialize switch head bias for all controllers.
        
        A negative bias results in lower initial switch probability:
        - bias = 0   -> sigmoid(0) = 0.50 (50% switch rate)
        - bias = -3  -> sigmoid(-3) ≈ 0.047 (5% switch rate)
        - bias = -4  -> sigmoid(-4) ≈ 0.018 (2% switch rate)
        - bias = -5  -> sigmoid(-5) ≈ 0.007 (0.7% switch rate)
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Access the underlying model (handle various wrapping)
        policy = model
        while hasattr(policy, 'module'):
            policy = policy.module
        
        # Initialize all layer controllers
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            num_initialized = 0
            for layer_idx, layer in enumerate(policy.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller'):
                    controller = layer.mlp.controller
                    if controller is not None and hasattr(controller, 'switch_head'):
                        with torch.no_grad():
                            controller.switch_head.bias.data.fill_(bias_value)
                        num_initialized += 1
            
            if self.accelerator.is_main_process:
                print(f"[INIT] Initialized switch_head.bias for {num_initialized} layers to {bias_value:.2f}")
    
    def _clamp_all_switch_biases(self, min_val: float, max_val: float) -> int:
        """Clamp switch_head.bias for ALL layer controllers.
        
        This prevents any layer's switch probability from collapsing to 0 or 1.
        
        Returns:
            Number of layers that were actually clamped (had out-of-range values)
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Access the underlying model (handle various wrapping)
        policy = model
        while hasattr(policy, 'module'):
            policy = policy.module
        
        num_clamped = 0
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            for layer_idx, layer in enumerate(policy.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller'):
                    controller = layer.mlp.controller
                    if controller is not None and hasattr(controller, 'switch_head'):
                        with torch.no_grad():
                            old_val = controller.switch_head.bias.data.clone()
                            controller.switch_head.bias.data.clamp_(min_val, max_val)
                            if (old_val < min_val).any() or (old_val > max_val).any():
                                num_clamped += 1
        
        return num_clamped
    
    def _initialize_layer_norm(self) -> None:
        """Initialize LayerNorm parameters to default values (weight=1, bias=0).
        
        This is needed because when loading from a checkpoint that doesn't have
        LayerNormGRU (we just added it), the parameters contain garbage memory.
        The LayerNorm modules are now inside the GRU cell (ln_r, ln_z, ln_n).
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Access the underlying model (handle various wrapping)
        policy = model
        while hasattr(policy, 'module'):
            policy = policy.module
        
        # Initialize all layer controllers' LayerNorm (now inside GRU cell)
        if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            num_initialized = 0
            for layer_idx, layer in enumerate(policy.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'controller'):
                    controller = layer.mlp.controller
                    if controller is not None and hasattr(controller, 'gru_cell'):
                        gru = controller.gru_cell
                        # Initialize LayerNorm modules inside GRU (separate input/hidden: ri, zi, ni, rh, zh, nh)
                        for ln_name in ['ln_ri', 'ln_zi', 'ln_ni', 'ln_rh', 'ln_zh', 'ln_nh']:
                            if hasattr(gru, ln_name):
                                ln = getattr(gru, ln_name)
                                if ln is not None:
                                    with torch.no_grad():
                                        ln.weight.data.fill_(1.0)
                                        ln.bias.data.zero_()
                        num_initialized += 1
            
            if self.accelerator.is_main_process:
                print(f"[INIT] Initialized LayerNormGRU (6 LN modules) for {num_initialized} layers to weight=1.0, bias=0.0")
    
    # =========================================================================
    # Rollout Phase
    # =========================================================================
    
    @torch.no_grad()
    def generate_rollout(
        self,
        queries: torch.Tensor,
    ) -> ControllerRollout:
        """
        Generate a rollout by running the model with the controller.
        
        This records all controller decisions and inputs for later training.
        
        Args:
            queries: [batch, query_len] - input token IDs
            
        Returns:
            ControllerRollout containing recorded actions and rewards
        """
        self.model.eval()
        device = queries.device
        
        # Debug: Check batch distribution across GPUs
        print(f"  [ROLLOUT] GPU {self.accelerator.process_index}: batch_size={queries.shape[0]}, query_len={queries.shape[1]}", flush=True)
        
        # Set up controller runtime to record actions
        controller_runtime = {
            "sampling": True,
            "record_actions": {},
        }
        
        # Get the unwrapped model for generation
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        # Generate responses
        attention_mask = (queries != self.tokenizer.pad_token_id).long()
        
        gen_start = time.time()
        outputs = policy.generate(
            input_ids=queries,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            controller_runtime=controller_runtime,
        )
        gen_time = time.time() - gen_start
        
        # Extract responses (remove query prefix)
        query_len = queries.shape[1]
        responses = outputs[:, query_len:]
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] Generated {responses.shape[1]} tokens in {gen_time:.1f}s")
        
        # Compute response lengths (number of non-padding tokens)
        # Response length = position of EOS token or total length if no EOS
        # Also track whether sequence truly terminated (EOS) or was truncated (hit max_length)
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = responses.shape[0]
        response_lengths = torch.zeros(batch_size, device=responses.device, dtype=torch.long)
        terminated = torch.zeros(batch_size, device=responses.device, dtype=torch.bool)  # True = hit EOS
        
        for i in range(batch_size):
            # Find first EOS or PAD token
            resp = responses[i]
            eos_positions = (resp == eos_token_id).nonzero(as_tuple=True)[0]
            pad_positions = (resp == pad_token_id).nonzero(as_tuple=True)[0] if pad_token_id is not None else torch.tensor([], device=resp.device)
            
            if len(eos_positions) > 0:
                # Length is position of first EOS + 1 (to include the EOS)
                response_lengths[i] = eos_positions[0].item() + 1
                terminated[i] = True  # True terminal: hit EOS
            elif len(pad_positions) > 0:
                # Length is position of first PAD (truncated before EOS, then padded)
                response_lengths[i] = pad_positions[0].item()
                terminated[i] = False  # Truncation
            else:
                # No EOS or PAD found, use full length (truncated at max_length)
                response_lengths[i] = resp.shape[0]
                terminated[i] = False  # Truncation
        
        if self.accelerator.is_main_process:
            num_terminated = terminated.sum().item()
            print(f"  [ROLLOUT] Response lengths: mean={response_lengths.float().mean().item():.1f}, min={response_lengths.min().item()}, max={response_lengths.max().item()}")
            print(f"  [ROLLOUT] Terminated (EOS): {num_terminated}/{batch_size} ({100*num_terminated/batch_size:.1f}%), Truncated: {batch_size - num_terminated}")
        
        # Get recorded controller actions
        recorded_actions = controller_runtime.get("record_actions", {})
        
        if self.accelerator.is_main_process:
            num_layers = len(recorded_actions)
            print(f"  [ROLLOUT] Recorded actions from {num_layers} layers")
        
        # Compute rewards (pass query_len and response_lengths for normalization)
        rewards, base_rewards, per_token_kl_list = self._compute_rewards(queries, responses, recorded_actions, query_len, response_lengths)
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] Rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item() if rewards.numel() > 1 else 0:.4f}")
            print(f"  [ROLLOUT] Base rewards (quality): mean={base_rewards.mean().item():.4f}")
        
        # Convert per_token_kl list to padded tensor for Option-Critic
        # Shape: [batch, max_response_len], padded with 0 for shorter responses
        per_token_kl_tensor = None
        if per_token_kl_list is not None and self.config.advantage_method == "option_critic":
            max_len = response_lengths.max().item()
            batch_size = len(per_token_kl_list)
            per_token_kl_tensor = torch.zeros(batch_size, int(max_len), dtype=torch.float32)
            for i, kl in enumerate(per_token_kl_list):
                if kl is not None:
                    length = min(len(kl), int(max_len))
                    per_token_kl_tensor[i, :length] = kl[:length]
            per_token_kl_tensor = per_token_kl_tensor.to(queries.device)
            if self.accelerator.is_main_process:
                print(f"  [OPTION-CRITIC] per_token_kl_tensor shape: {per_token_kl_tensor.shape}", flush=True)
        
        return ControllerRollout(
            layer_data=recorded_actions,
            queries=queries,
            responses=responses,
            rewards=rewards,
            base_rewards=base_rewards,
            response_lengths=response_lengths,
            pad_token_id=self.tokenizer.pad_token_id,
            per_token_kl=per_token_kl_tensor,  # For Option-Critic
            terminated=terminated,  # True = hit EOS, False = truncated
        )
    
    @torch.no_grad()
    def generate_grpo_rollouts(
        self,
        queries: torch.Tensor,
        num_generations: int,
    ) -> ControllerRollout:
        """
        Generate multiple rollouts per prompt for GRPO.
        
        Each unique prompt gets `num_generations` different responses.
        This enables group-level baseline computation.
        
        Args:
            queries: [batch, query_len] - unique prompts (one per sample)
            num_generations: Number of responses to generate per prompt
            
        Returns:
            ControllerRollout with batch_size = original_batch * num_generations
            and group_ids indicating which responses share the same prompt
        """
        self.model.eval()
        device = queries.device
        original_batch_size = queries.shape[0]
        
        if self.accelerator.is_main_process:
            print(f"  [GRPO-ROLLOUT] Generating {num_generations} responses per prompt", flush=True)
            print(f"  [GRPO-ROLLOUT] Original batch: {original_batch_size}, expanded: {original_batch_size * num_generations}", flush=True)
        
        # Duplicate each query num_generations times
        # [batch, query_len] -> [batch * num_generations, query_len]
        expanded_queries = queries.repeat_interleave(num_generations, dim=0)
        
        # Create group IDs: samples 0,1,2,3 belong to prompt 0; 4,5,6,7 to prompt 1; etc.
        group_ids = torch.arange(original_batch_size, device=device).repeat_interleave(num_generations)
        
        # Set up controller runtime to record actions
        controller_runtime = {
            "sampling": True,
            "record_actions": {},
        }
        
        # Get the unwrapped model for generation
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        # Generate responses
        attention_mask = (expanded_queries != self.tokenizer.pad_token_id).long()
        
        gen_start = time.time()
        outputs = policy.generate(
            input_ids=expanded_queries,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            controller_runtime=controller_runtime,
        )
        gen_time = time.time() - gen_start
        
        # Extract responses (remove query prefix)
        query_len = expanded_queries.shape[1]
        responses = outputs[:, query_len:]
        
        if self.accelerator.is_main_process:
            print(f"  [GRPO-ROLLOUT] Generated {responses.shape[1]} tokens in {gen_time:.1f}s")
        
        # Compute response lengths and terminated flags
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        expanded_batch_size = responses.shape[0]
        response_lengths = torch.zeros(expanded_batch_size, device=responses.device, dtype=torch.long)
        terminated = torch.zeros(expanded_batch_size, device=responses.device, dtype=torch.bool)
        
        for i in range(expanded_batch_size):
            resp = responses[i]
            eos_positions = (resp == eos_token_id).nonzero(as_tuple=True)[0]
            pad_positions = (resp == pad_token_id).nonzero(as_tuple=True)[0] if pad_token_id is not None else torch.tensor([], device=resp.device)
            
            if len(eos_positions) > 0:
                response_lengths[i] = eos_positions[0].item() + 1
                terminated[i] = True  # True terminal: hit EOS
            elif len(pad_positions) > 0:
                response_lengths[i] = pad_positions[0].item()
                terminated[i] = False  # Truncation
            else:
                response_lengths[i] = resp.shape[0]
                terminated[i] = False  # Truncation
        
        if self.accelerator.is_main_process:
            num_terminated = terminated.sum().item()
            print(f"  [GRPO-ROLLOUT] Response lengths: mean={response_lengths.float().mean().item():.1f}")
            print(f"  [GRPO-ROLLOUT] Terminated (EOS): {num_terminated}/{expanded_batch_size} ({100*num_terminated/expanded_batch_size:.1f}%)")
        
        # Get recorded controller actions
        recorded_actions = controller_runtime.get("record_actions", {})
        
        if self.accelerator.is_main_process:
            num_layers = len(recorded_actions)
            print(f"  [GRPO-ROLLOUT] Recorded actions from {num_layers} layers")
        
        # Compute rewards (ignore per_token_kl for GRPO)
        rewards, base_rewards, _ = self._compute_rewards(
            expanded_queries, responses, recorded_actions, query_len, response_lengths
        )
        
        if self.accelerator.is_main_process:
            print(f"  [GRPO-ROLLOUT] Rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}")
            print(f"  [GRPO-ROLLOUT] Base rewards: mean={base_rewards.mean().item():.4f}")
            
            # Print per-group statistics
            for g in range(min(original_batch_size, 2)):  # Only print first 2 groups
                mask = (group_ids == g)
                group_rewards = rewards[mask]
                print(f"  [GRPO-ROLLOUT] Group {g}: rewards={group_rewards.tolist()}", flush=True)
        
        return ControllerRollout(
            layer_data=recorded_actions,
            queries=expanded_queries,
            responses=responses,
            rewards=rewards,
            base_rewards=base_rewards,
            response_lengths=response_lengths,
            pad_token_id=self.tokenizer.pad_token_id,
            group_ids=group_ids,
            terminated=terminated,  # True = hit EOS, False = truncated
        )
    
    def _compute_rewards(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        recorded_actions: Dict[int, Dict[str, torch.Tensor]],
        query_len: int,
        response_lengths: torch.Tensor,  # [batch] - actual response lengths
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute rewards for generated responses.
        
        Returns:
            rewards: [batch] - final rewards (with latency penalty)
            base_rewards: [batch] - quality scores (without latency penalty)
            per_token_kl: Optional list of [response_len] tensors - per-token KL for Option-Critic
        """
        # Decode texts (use clean_up_tokenization_spaces=False to avoid whitespace changes
        # when re-encoding in the reward function)
        query_texts = self.tokenizer.batch_decode(queries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Compute switch counts for latency penalty
        # Only count switches for meaningful tokens:
        # - Exclude left-padding in query (using attention_mask)
        # - Exclude right-padding in response (after EOS)
        batch_size = queries.shape[0]
        switch_counts = torch.zeros(batch_size, device=queries.device)
        
        # Compute attention mask for query (1 for real tokens, 0 for padding)
        query_attention_mask = (queries != self.tokenizer.pad_token_id).long()  # [batch, query_len]
        # Number of real query tokens per sample (excluding left-padding)
        real_query_lengths = query_attention_mask.sum(dim=1)  # [batch]
        # Number of left-padding tokens per sample
        left_padding_lengths = query_len - real_query_lengths  # [batch]
        
        for layer_data in recorded_actions.values():
            switches = layer_data.get("switches")
            if switches is not None:
                seq_len = switches.shape[1]
                positions = torch.arange(seq_len, device=switches.device).unsqueeze(0)  # [1, seq_len]
                
                # Valid range: [left_padding_length, query_len + response_length)
                # i.e., position >= left_padding_length AND position < query_len + response_length
                start = left_padding_lengths.unsqueeze(1)  # [batch, 1]
                end = (query_len + response_lengths).unsqueeze(1)  # [batch, 1]
                mask = (positions >= start) & (positions < end)  # [batch, seq_len]
                
                # Only count switches within valid positions
                switch_counts += (switches.float() * mask.float()).sum(dim=1)
        
        # Get base rewards from reward function
        base_rewards = self.reward_fn(query_texts, response_texts)
        
        # Ensure base_rewards is on the same device
        if not isinstance(base_rewards, torch.Tensor):
            base_rewards = torch.tensor(base_rewards, dtype=torch.float32)
        base_rewards = base_rewards.to(device=queries.device, dtype=torch.float32)
        
        # Apply latency penalty normalized by sequence length AND number of layers
        # This penalizes switch RATE (fraction of token-layer pairs that switched)
        num_layers = len(recorded_actions)
        # Use real_query_lengths (excluding left-padding) + response_lengths (excluding right-padding)
        total_seq_len = real_query_lengths.float() + response_lengths.float()  # [batch]
        # switch_rate is now between 0 and 1 (fraction of possible switches)
        switch_rate = switch_counts / (total_seq_len * num_layers).clamp(min=1)  # [batch]
        latency_penalty = self.config.latency_cost_per_switch * switch_rate
        rewards = base_rewards - latency_penalty
        
        # Debug: Print model output and reward score (first sample only) - ALWAYS print
        if self.accelerator.is_main_process:
            print(f"  [DEBUG-OUTPUT] ===== QUERY =====", flush=True)
            print(f"{query_texts[0]}", flush=True)
            print(f"  [DEBUG-OUTPUT] ===== RESPONSE =====", flush=True)
            print(f"{response_texts[0]}", flush=True)
            print(f"  [DEBUG-OUTPUT] ===== REWARD INFO =====", flush=True)
            print(f"  [DEBUG-OUTPUT] Base reward (quality score): {base_rewards[0].item():.4f}", flush=True)
            print(f"  [DEBUG-OUTPUT] Switch count: {switch_counts[0].item():.0f}", flush=True)
            print(f"  [DEBUG-OUTPUT] Sequence length: {total_seq_len[0].item():.0f} (query={query_len}, resp={response_lengths[0].item():.0f})", flush=True)
            print(f"  [DEBUG-OUTPUT] Num layers: {num_layers}", flush=True)
            print(f"  [DEBUG-OUTPUT] Switch rate: {switch_rate[0].item():.4f} (fraction of token-layer pairs)", flush=True)
            print(f"  [DEBUG-OUTPUT] Latency penalty: {latency_penalty[0].item():.4f}", flush=True)
            print(f"  [DEBUG-OUTPUT] Final reward: {rewards[0].item():.4f}", flush=True)
            print(f"  [DEBUG-OUTPUT] ===================", flush=True)
        
        # Get per-token KL from scorer if available (for Option-Critic)
        per_token_kl = None
        if hasattr(self, 'ppl_scorer') and self.ppl_scorer is not None:
            if hasattr(self.ppl_scorer, 'last_batch_per_token_kl'):
                per_token_kl = self.ppl_scorer.last_batch_per_token_kl
                if self.accelerator.is_main_process and per_token_kl is not None:
                    valid_count = sum(1 for x in per_token_kl if x is not None)
                    print(f"  [OPTION-CRITIC] Retrieved per_token_kl: {valid_count}/{len(per_token_kl)} valid tensors", flush=True)
        
        return rewards, base_rewards, per_token_kl
    
    # =========================================================================
    # Controller Update Phase
    # =========================================================================
    
    def _process_single_layer(
        self,
        layer_idx: int,  # Layer index - needed for temperature lookup
        controller,
        router_logits: torch.Tensor,  # [batch, seq, num_experts]
        switches: torch.Tensor,  # [batch, seq]
        selected_indices: torch.Tensor,  # [batch, seq, k]
        controller_inputs_recorded: torch.Tensor,  # [batch, seq, 2*num_experts] - x_t = [router_softmax, expert_mask]
        controller_dtype: torch.dtype,
        rewards: torch.Tensor,  # [batch] - final reward for advantage computation
        valid_mask: Optional[torch.Tensor] = None,  # [batch, seq] - True for valid positions, False for padding
    ) -> Dict[str, Any]:
        """
        Process a single layer with canonical RNN sequential computation.
        
        OPTIMIZED: Runs GRU sequentially but batches log_prob computation after the loop.
        
        Returns per-timestep advantages and SEPARATE log_probs for switch and expert decisions.
        
        Canonical RNN architecture:
            h_t = GRU(h_{t-1}, x_t)  where x_t = [router_softmax, expert_mask]
            switch_logits, candidate_logits, value = heads(h_t)
        
        The hidden state h is RECOMPUTED for BPTT (gradients flow through h).
        The expert_mask is RECORDED (discrete, no gradient benefit from recomputing).
        
        t=0: Initialize hidden state, no explicit loss
        t>0: Normal training with policy/value loss
        
        Returns dict with:
            advantages: List of [batch] tensors - raw advantages A_t = R - V(s_t)
            switch_log_probs: List of [batch] tensors - log π(switch_t|s_t) for ALL timesteps
            expert_log_probs: List of [batch] tensors - log π(experts_t|s_t) for ALL timesteps
            switch_decisions: List of [batch] bool tensors - which timesteps had switch=True
            layer_value_loss: [batch] - sum of (V(s_t) - R)^2 for this layer
            layer_switch_log_prob: [batch] - sum of switch log probs (for logging)
            layer_expert_log_prob: [batch] - sum of expert log probs where switch=True (for logging)
            num_total_timesteps: int - total number of timesteps
            num_switch_timesteps: int - number of timesteps where switch=True
        """
        device = router_logits.device
        batch_size, seq_len, num_experts = router_logits.shape
        
        # =========================================================================
        # Phase 1: Sequential GRU loop - collect outputs for batched log_prob later
        # =========================================================================
        # Store outputs for all timesteps t > 0 (t=0 is just initialization)
        all_switch_probs = []      # List of [batch] tensors
        all_switch_decisions = []  # List of [batch] bool tensors
        all_candidate_logits = []  # List of [batch, num_experts] tensors
        all_values = []            # List of [batch] tensors
        
        # Initialize hidden state for canonical RNN (BPTT)
        hidden_state = controller.init_hidden(batch_size, device, controller_dtype)
        
        for t in range(seq_len):
            # Use EXACT recorded controller_inputs for on-policy state
            # controller_inputs structure depends on input_type:
            # - "router_softmax": [router_softmax, expert_mask], each num_experts dims
            # - "hidden_states": [hidden_states, expert_mask], hidden_dim + num_experts dims
            # CRITICAL: We must use the RECORDED inputs, not recompute from router_logits
            # because router_logits were recorded BEFORE masking - using the exact same state ensures
            # we evaluate log π(a|s) under the correct state s
            input_type = getattr(controller, 'input_type', 'router_softmax')
            if input_type == "hidden_states":
                hidden_dim = controller.model_hidden_size
                hidden_states_recorded = controller_inputs_recorded[:, t, :hidden_dim].to(device=device, dtype=controller_dtype)
                expert_mask_recorded = controller_inputs_recorded[:, t, hidden_dim:].to(device=device, dtype=controller_dtype)
                x_t = torch.cat([hidden_states_recorded, expert_mask_recorded], dim=-1)
            else:
                router_softmax_recorded = controller_inputs_recorded[:, t, :num_experts].to(device=device, dtype=controller_dtype)
                expert_mask_recorded = controller_inputs_recorded[:, t, num_experts:].to(device=device, dtype=controller_dtype)
                x_t = torch.cat([router_softmax_recorded, expert_mask_recorded], dim=-1)
            
            # Forward through canonical RNN controller
            hidden_state, switch_logits, controller_perturbation, value = controller(x_t, hidden_state)
            
            # RESIDUAL CONNECTION: candidate_logits = router_logits + controller_perturbation
            # This must match inference exactly! (see _apply_controller_step)
            router_logits_t = router_logits[:, t, :].to(controller_perturbation.dtype)
            candidate_logits = router_logits_t + controller_perturbation
            
            if t == 0:
                # t=0: Initialize hidden state, no explicit loss
                # Gradient flows back via BPTT from t=1's loss through hidden_state
                continue
            
            # DEBUG: Check if outputs require grad (only on first token, first call)
            if t == 1 and hasattr(self, '_debug_grad_check_done') is False:
                self._debug_grad_check_done = True
                print(f"  [DEBUG-GRAD] switch_logits.requires_grad={switch_logits.requires_grad}", flush=True)
                print(f"  [DEBUG-GRAD] value.requires_grad={value.requires_grad}", flush=True)
                print(f"  [DEBUG-GRAD] candidate_logits.requires_grad={candidate_logits.requires_grad}", flush=True)
                print(f"  [DEBUG-GRAD] hidden_state.requires_grad={hidden_state.requires_grad}", flush=True)
                print(f"  [DEBUG-GRAD] x_t.requires_grad={x_t.requires_grad}", flush=True)
                # Debug input magnitudes
                print(f"  [DEBUG-INPUT] controller_input_type={input_type}", flush=True)
                print(f"  [DEBUG-INPUT] hidden_state: min={hidden_state.min().item():.2f}, max={hidden_state.max().item():.2f}, mean={hidden_state.mean().item():.2f}", flush=True)
                print(f"  [DEBUG-INPUT] x_t: min={x_t.min().item():.2f}, max={x_t.max().item():.2f}, mean={x_t.mean().item():.2f}", flush=True)
            
            # Clamp outputs - MUST match inference exactly!
            switch_logits = switch_logits.clamp(-20, 20)
            candidate_logits = candidate_logits.clamp(-20, 20)
            
            # Apply temperature scaling - MUST match inference exactly!
            # During rollout, actions are sampled with temperature-scaled logits,
            # so training must compute log_probs with the same scaled logits.
            sampling_temperature = self._get_controller_sampling_temperature(layer_idx)
            if sampling_temperature != 1.0:
                candidate_logits = candidate_logits / sampling_temperature
            
            # Compute switch probs (but not log prob yet - will batch later)
            switch_probs = torch.sigmoid(switch_logits.float())
            switch_decisions = switches[:, t].bool()
            
            # DEBUG: Print switch_probs on first few tokens of first layer (only for batch_size=1)
            if t <= 3 and hasattr(self, '_debug_switch_probs_done') is False and switch_logits.numel() == 1:
                print(f"  [DEBUG-SWITCH] t={t}: switch_logits={switch_logits.item():.4f}, switch_probs={switch_probs.item():.4f}, switch_decision={switch_decisions.item()}", flush=True)
                if t == 3:
                    self._debug_switch_probs_done = True
            
            # Store for batched computation
            all_switch_probs.append(switch_probs)
            all_switch_decisions.append(switch_decisions)
            all_candidate_logits.append(candidate_logits.float())  # Keep in float32
            all_values.append(value.float())
        
        # =========================================================================
        # Phase 2: Batched log_prob computation (after the loop)
        # =========================================================================
        num_timesteps = len(all_switch_probs)  # seq_len - 1 (excluding t=0)
        
        if num_timesteps == 0:
            # No timesteps to process (shouldn't happen in practice)
            return {
                "advantages": [],
                "switch_log_probs": [],
                "expert_log_probs": [],
                "switch_decisions": [],
                "valid_masks": [],
                "layer_value_loss": torch.zeros(batch_size, device=device, dtype=torch.float32),
                "layer_switch_log_prob": torch.zeros(batch_size, device=device, dtype=torch.float32),
                "layer_expert_log_prob": torch.zeros(batch_size, device=device, dtype=torch.float32),
                "num_total_timesteps": 0,
                "num_valid_timesteps": 0,
                "num_switch_timesteps": 0,
            }
        
        # Stack all tensors: [num_timesteps, batch, ...] -> [batch * num_timesteps, ...]
        stacked_switch_probs = torch.stack(all_switch_probs, dim=0)  # [num_timesteps, batch]
        stacked_switch_decisions = torch.stack(all_switch_decisions, dim=0)  # [num_timesteps, batch]
        stacked_candidate_logits = torch.stack(all_candidate_logits, dim=0)  # [num_timesteps, batch, num_experts]
        stacked_values = torch.stack(all_values, dim=0)  # [num_timesteps, batch]
        
        # Compute expert softmax entropy: H = -sum(p * log(p))
        # Higher entropy = more uniform distribution = more diverse expert selection
        # Lower entropy = peaked distribution = potential mode collapse
        expert_probs = F.softmax(stacked_candidate_logits, dim=-1)  # [num_timesteps, batch, num_experts]
        log_probs = torch.log(expert_probs.clamp(min=1e-10))
        expert_entropy = -(expert_probs * log_probs).sum(dim=-1)  # [num_timesteps, batch]
        mean_expert_entropy = expert_entropy.mean()  # Scalar
        
        # Flatten for batched computation
        flat_switch_probs = stacked_switch_probs.view(-1)  # [num_timesteps * batch]
        flat_switch_decisions = stacked_switch_decisions.view(-1)  # [num_timesteps * batch]
        flat_candidate_logits = stacked_candidate_logits.view(-1, num_experts)  # [num_timesteps * batch, num_experts]
        
        # Get selected_indices for timesteps 1 to seq_len-1 (matching our stored outputs)
        # selected_indices shape: [batch, seq_len, k]
        selected_indices_for_training = selected_indices[:, 1:, :]  # [batch, num_timesteps, k]
        # Transpose to [num_timesteps, batch, k] to match the stacking order of other tensors
        selected_indices_transposed = selected_indices_for_training.transpose(0, 1)  # [num_timesteps, batch, k]
        flat_selected_indices = selected_indices_transposed.reshape(-1, selected_indices.shape[-1])  # [num_timesteps * batch, k]
        
        # Compute switch log probs (batched)
        flat_log_p_switch = torch.where(
            flat_switch_decisions,
            _safe_logprob(flat_switch_probs),
            _safe_log1m(flat_switch_probs),
        ).clamp(min=-50)
        
        # Compute expert selection log probs (batched) - ONE call instead of num_timesteps calls!
        flat_log_p_experts = _plackett_luce_logprob_batched(
            flat_candidate_logits,
            flat_selected_indices,
        ).clamp(min=-50)
        
        # DEBUG: Check if log probs have gradients (first layer only)
        if not hasattr(self, '_debug_layer_grad_checked'):
            self._debug_layer_grad_checked = True
            print(f"  [DEBUG-LAYER-GRAD] flat_switch_probs.requires_grad={flat_switch_probs.requires_grad}", flush=True)
            print(f"  [DEBUG-LAYER-GRAD] flat_switch_probs.grad_fn={flat_switch_probs.grad_fn}", flush=True)
            print(f"  [DEBUG-LAYER-GRAD] flat_log_p_switch.requires_grad={flat_log_p_switch.requires_grad}", flush=True)
            print(f"  [DEBUG-LAYER-GRAD] flat_log_p_switch.grad_fn={flat_log_p_switch.grad_fn}", flush=True)
            print(f"  [DEBUG-LAYER-GRAD] flat_candidate_logits.requires_grad={flat_candidate_logits.requires_grad}", flush=True)
            print(f"  [DEBUG-LAYER-GRAD] flat_log_p_experts.requires_grad={flat_log_p_experts.requires_grad}", flush=True)
        
        # Reshape back to [num_timesteps, batch]
        log_p_switch_all = flat_log_p_switch.view(num_timesteps, batch_size)
        log_p_experts_all = flat_log_p_experts.view(num_timesteps, batch_size)
        switch_decisions_all = stacked_switch_decisions  # Already [num_timesteps, batch]
        
        # =========================================================================
        # Phase 3: Compute advantages and collect SEPARATE log_probs
        # =========================================================================
        advantages = []
        switch_log_probs = []
        expert_log_probs = []
        switch_decisions_list = []
        valid_mask_list = []  # Track which positions are valid for this layer
        
        # Accumulators for value loss and logging
        layer_value_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        layer_switch_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        layer_expert_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        num_switch_timesteps = 0
        num_valid_timesteps = 0  # Count actual valid timesteps (not padding)
        
        for t_idx in range(num_timesteps):
            switch_log_prob_t = log_p_switch_all[t_idx]  # [batch]
            expert_log_prob_t = log_p_experts_all[t_idx]  # [batch]
            switch_decision_t = switch_decisions_all[t_idx]  # [batch] bool
            value_t = stacked_values[t_idx]   # [batch]
            
            # Get valid mask for this timestep (if provided)
            # t_idx corresponds to t+1 in original seq (since we skip t=0 in the loop)
            # But we stored all t from 1 onwards, so t_idx corresponds to original position t_idx+1
            # Actually looking at the code, t_idx here is 0-based index into num_timesteps which is seq_len-1
            # Since we skip t=0 in the GRU loop, t_idx=0 corresponds to original position 1
            if valid_mask is not None:
                # valid_mask is [batch, seq_len], we need position t_idx+1 since t=0 is skipped
                valid_t = valid_mask[:, t_idx + 1]  # [batch] bool
            else:
                valid_t = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # Per-timestep advantage: A_t = R - V(s_t)
            # Value predicts expected future return, which is just R (final reward) since r_t = 0 for t < T
            # NOTE: Advantages will be normalized AFTER collecting from all layers
            advantage_t = rewards - value_t.detach()
            
            # Collect for later normalization (only for valid positions)
            # Store the original values - we'll mask them out during loss computation
            advantages.append(advantage_t)
            switch_log_probs.append(switch_log_prob_t)
            expert_log_probs.append(expert_log_prob_t)
            switch_decisions_list.append(switch_decision_t)
            valid_mask_list.append(valid_t)
            
            # Accumulate value loss: (V(s_t) - R)^2 - only for VALID positions
            # Use valid_t to mask out padding positions
            valid_value_loss = torch.where(valid_t, (value_t - rewards) ** 2, torch.zeros_like(value_t))
            layer_value_loss = layer_value_loss + valid_value_loss
            
            # Accumulate log probs for logging - only for VALID positions
            layer_switch_log_prob = layer_switch_log_prob + torch.where(
                valid_t, switch_log_prob_t, torch.zeros_like(switch_log_prob_t)
            )
            # Only count expert log prob where switch=True AND position is valid
            layer_expert_log_prob = layer_expert_log_prob + torch.where(
                switch_decision_t & valid_t, expert_log_prob_t, torch.zeros_like(expert_log_prob_t)
            )
            # Count switches only at valid positions
            num_switch_timesteps += (switch_decision_t & valid_t).sum().item()
            num_valid_timesteps += valid_t.sum().item()
        
        return {
            "advantages": advantages,
            "switch_log_probs": switch_log_probs,
            "expert_log_probs": expert_log_probs,
            "switch_decisions": switch_decisions_list,
            "valid_masks": valid_mask_list,  # NEW: track which positions are valid
            "layer_value_loss": layer_value_loss,
            "layer_switch_log_prob": layer_switch_log_prob,
            "layer_expert_log_prob": layer_expert_log_prob,
            "num_total_timesteps": num_timesteps,  # Total timesteps (for compatibility)
            "num_valid_timesteps": num_valid_timesteps,  # NEW: actual valid timesteps
            "num_switch_timesteps": int(num_switch_timesteps),
            "candidate_logits": stacked_candidate_logits,  # DEBUG: [num_timesteps, batch, num_experts]
            "expert_entropy": mean_expert_entropy,  # Entropy of softmax(candidate_logits)
        }

    def controller_forward(
        self,
        rollout: ControllerRollout,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass through controller only (no LLM).
        
        EXACT Markovian computation: sequential within each layer,
        but layers are processed independently (could be parallelized).
        
        Collects per-timestep advantages, normalizes them globally, then computes policy loss.
        This ensures stable gradients even when the value function fits well.
        
        Args:
            rollout: ControllerRollout with recorded actions and rewards
            
        Returns:
            total_policy_loss: [batch] - SUM of (-normalized_A_t * log_prob_t) across all decisions
            total_value_loss: [batch] - SUM of (V(s_t) - R)^2 across all decisions
            num_decisions: int - total number of decisions made
        """
        controller = self._get_controller_module()
        device = rollout.queries.device
        batch_size = rollout.batch_size
        rewards = rollout.rewards.float().to(device)  # [batch] - final reward
        
        # Get dimensions from first layer
        first_layer_data = next(iter(rollout.layer_data.values()))
        num_experts = first_layer_data["router_logits"].shape[-1]
        
        # Get controller dtype from actual parameters
        controller_dtype = next(controller.parameters()).dtype
        
        # =========================================================================
        # Compute valid_mask: exclude left-padding and post-EOS padding
        # valid_mask[i, t] = True if position t is a real token for sample i
        # =========================================================================
        query_len = rollout.queries.shape[1]
        response_len = rollout.responses.shape[1]
        seq_len = first_layer_data["router_logits"].shape[1]  # Total sequence length
        
        # Query: left-padded, so valid positions are [left_pad_len, query_len)
        query_attention_mask = (rollout.queries != rollout.pad_token_id)  # [batch, query_len]
        left_padding_lengths = query_len - query_attention_mask.sum(dim=1)  # [batch]
        
        # Response: right-padded after EOS, valid positions are [query_len, query_len + response_lengths)
        response_lengths = rollout.response_lengths.to(device)  # [batch]
        
        # Build valid_mask: [batch, seq_len]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        valid_start = left_padding_lengths.unsqueeze(1)  # [batch, 1]
        valid_end = (query_len + response_lengths).unsqueeze(1)  # [batch, 1]
        valid_mask = (positions >= valid_start) & (positions < valid_end)  # [batch, seq_len]
        
        # Debug: Check what data we have - ALWAYS print
        if self.accelerator.is_main_process:
            print(f"  [CTRL-FWD] num_layers={len(rollout.layer_data)}, batch_size={batch_size}, num_experts={num_experts}", flush=True)
            print(f"  [CTRL-FWD] controller_dtype={controller_dtype}", flush=True)
            print(f"  [CTRL-FWD] first_layer router_logits shape: {first_layer_data['router_logits'].shape}", flush=True)
            print(f"  [CTRL-FWD] rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item() if rewards.numel() > 1 else 0:.4f}", flush=True)
            print(f"  [CTRL-FWD] valid_mask: {valid_mask.sum().item()} / {valid_mask.numel()} positions valid ({100*valid_mask.sum().item()/valid_mask.numel():.1f}%)", flush=True)
            print(f"  [CTRL-FWD] Using NORMALIZED advantages: A'_t = (A_t - mean) / (std + eps)", flush=True)
        
        # Collect SEPARATE advantages, switch log_probs, expert log_probs from all layers
        all_advantages = []           # List of [batch] tensors
        all_switch_log_probs = []     # List of [batch] tensors (for ALL timesteps)
        all_expert_log_probs = []     # List of [batch] tensors (for ALL timesteps, but loss only on switch=True)
        all_switch_decisions = []     # List of [batch] bool tensors
        all_valid_masks = []          # List of [batch] bool tensors - which positions are valid
        
        # Accumulators
        total_value_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        total_switch_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        total_expert_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        num_total_timesteps = 0
        num_valid_timesteps = 0  # Count actual valid timesteps (not padding)
        num_switch_timesteps = 0
        
        # Phase 1: Collect advantages and log_probs from all layers
        # IMPORTANT: Each layer has its own controller with independent weights!
        for layer_idx in sorted(rollout.layer_data.keys()):
            layer_data = rollout.layer_data[layer_idx]
            
            # Get the controller for THIS specific layer
            layer_controller = self._get_controller_for_layer(layer_idx)
            
            router_logits = layer_data["router_logits"].to(device, dtype=controller_dtype)
            switches = layer_data["switches"].to(device)
            selected_indices = layer_data["selected_indices"].to(device)
            # Canonical RNN: controller_inputs is x_t = [router_softmax, expert_mask]
            controller_inputs = layer_data["controller_inputs"].to(device, dtype=controller_dtype)
            
            # Process this layer with canonical RNN sequential computation
            layer_result = self._process_single_layer(
                layer_idx,  # Layer index for temperature lookup
                layer_controller,
                router_logits,
                switches,
                selected_indices,
                controller_inputs,
                controller_dtype,
                rewards,
                valid_mask=valid_mask,  # Pass valid_mask to exclude padding from loss
            )
            
            # Collect advantages and log_probs for global normalization
            all_advantages.extend(layer_result["advantages"])
            all_switch_log_probs.extend(layer_result["switch_log_probs"])
            all_expert_log_probs.extend(layer_result["expert_log_probs"])
            all_switch_decisions.extend(layer_result["switch_decisions"])
            all_valid_masks.extend(layer_result["valid_masks"])  # Track valid positions
            
            total_value_loss = total_value_loss + layer_result["layer_value_loss"]
            total_switch_log_prob = total_switch_log_prob + layer_result["layer_switch_log_prob"]
            total_expert_log_prob = total_expert_log_prob + layer_result["layer_expert_log_prob"]
            num_total_timesteps += layer_result["num_total_timesteps"]
            num_valid_timesteps += layer_result["num_valid_timesteps"]  # Use valid count
            num_switch_timesteps += layer_result["num_switch_timesteps"]
        
        # Phase 2: Normalize advantages globally across all VALID timesteps and batch
        if len(all_advantages) > 0:
            # Stack advantages and valid_masks: [num_timesteps, batch]
            stacked_advantages = torch.stack(all_advantages, dim=0)  # [num_timesteps, batch]
            stacked_valid_masks = torch.stack(all_valid_masks, dim=0)  # [num_timesteps, batch]
            
            # Compute global mean and std using ONLY VALID positions
            flat_advantages = stacked_advantages.flatten()  # [num_timesteps * batch]
            flat_valid = stacked_valid_masks.flatten()  # [num_timesteps * batch]
            valid_advantages = flat_advantages[flat_valid]  # [num_valid]
            
            if valid_advantages.numel() > 1:
                adv_mean = valid_advantages.mean()
                adv_std = valid_advantages.std(unbiased=False).clamp(min=1e-8)
            elif valid_advantages.numel() == 1:
                # Single element: use its value as mean, std=1 (no normalization)
                adv_mean = valid_advantages.mean()
                adv_std = torch.ones(1, device=device)
            else:
                # No valid elements
                adv_mean = torch.zeros(1, device=device)
                adv_std = torch.ones(1, device=device)
            
            # Debug advantage stats
            if self.accelerator.is_main_process:
                print(f"  [ADV-NORM] raw (valid only): mean={adv_mean.item():.4f}, std={adv_std.item():.4f}, n_valid={valid_advantages.numel()}", flush=True)
            
            # Normalize each advantage (using global stats from valid positions)
            normalized_advantages = [(adv - adv_mean) / adv_std for adv in all_advantages]
            
            # Phase 3: Compute SEPARATE policy losses with normalized advantages
            # switch_loss = Σ_t (-A'_t * switch_log_prob_t) for VALID timesteps only
            # expert_loss = Σ_t (-A'_t * expert_log_prob_t) only for switch=True AND VALID timesteps
            total_switch_policy_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
            total_expert_policy_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
            
            for norm_adv, switch_lp, expert_lp, switch_dec, valid_t in zip(
                normalized_advantages, all_switch_log_probs, all_expert_log_probs, all_switch_decisions, all_valid_masks
            ):
                # Switch loss: Only VALID timesteps (exclude padding)
                switch_contribution = torch.where(
                    valid_t,
                    -norm_adv * switch_lp,
                    torch.zeros_like(switch_lp)
                )
                total_switch_policy_loss = total_switch_policy_loss + switch_contribution
                
                # Expert loss: Only switch=True AND VALID timesteps
                expert_contribution = torch.where(
                    switch_dec & valid_t,
                    -norm_adv * expert_lp,
                    torch.zeros_like(expert_lp)
                )
                total_expert_policy_loss = total_expert_policy_loss + expert_contribution
            
            # Normalize each loss by its own count (using VALID counts)
            # This ensures balanced gradients regardless of switch rate
            # NOTE: Use .sum() not .mean() since num_valid_timesteps already includes batch dimension
            mean_switch_loss = total_switch_policy_loss.sum() / max(num_valid_timesteps, 1)
            mean_expert_loss = total_expert_policy_loss.sum() / max(num_switch_timesteps, 1)
            
            # Combined policy loss (for backward)
            # We scale back up by a common factor so compute_loss normalizes properly
            total_policy_loss = mean_switch_loss + mean_expert_loss
        else:
            total_policy_loss = torch.zeros(1, device=device, dtype=torch.float32)
            mean_switch_loss = torch.zeros(1, device=device, dtype=torch.float32)
            mean_expert_loss = torch.zeros(1, device=device, dtype=torch.float32)
        
        # Debug output - ALWAYS print
        if self.accelerator.is_main_process:
            switch_rate = num_switch_timesteps / max(num_valid_timesteps, 1)
            # Use .sum() since num_valid_timesteps already includes batch dimension
            mean_switch_lp = total_switch_log_prob.sum().item() / max(num_valid_timesteps, 1)
            mean_expert_lp = total_expert_log_prob.sum().item() / max(num_switch_timesteps, 1) if num_switch_timesteps > 0 else 0
            mean_value_loss = total_value_loss.sum().item() / max(num_valid_timesteps, 1)
            print(f"  [CTRL-FWD] num_valid_timesteps={num_valid_timesteps}, num_switch_timesteps={num_switch_timesteps} ({switch_rate:.2%})", flush=True)
            print(f"  [CTRL-FWD] mean_switch_log_prob={mean_switch_lp:.4f}, mean_expert_log_prob={mean_expert_lp:.4f}", flush=True)
            print(f"  [CTRL-FWD] mean_switch_loss={mean_switch_loss.item():.4f}, mean_expert_loss={mean_expert_loss.item():.4f}", flush=True)
            print(f"  [CTRL-FWD] mean_value_loss={mean_value_loss:.4f}", flush=True)
            print(f"  [CTRL-FWD] finite={torch.isfinite(total_policy_loss).all().item() and torch.isfinite(total_value_loss).all().item()}", flush=True)
        
        # Return: total_policy_loss is now already normalized (mean_switch + mean_expert)
        # We return num_valid_timesteps for compute_loss to use
        return total_policy_loss, total_value_loss, num_valid_timesteps
    
    def controller_forward_grpo(
        self,
        rollout: ControllerRollout,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass for GRPO: uses group-level baseline instead of V(s_t).
        
        For GRPO:
        - Advantage = (R - group_mean) / (group_std + eps)
        - Same advantage for ALL timesteps in a trajectory
        - No value function needed for advantage computation
        
        Args:
            rollout: ControllerRollout with recorded actions, rewards, and group_ids
            
        Returns:
            total_policy_loss: [batch] - SUM of (-A * log_prob_t) across all decisions
            total_value_loss: [batch] - SUM of (V(s_t) - R)^2 (for monitoring only)
            num_decisions: int - total number of decisions made
        """
        controller = self._get_controller_module()
        device = rollout.queries.device
        batch_size = rollout.batch_size
        rewards = rollout.rewards.float().to(device)  # [batch]
        group_ids = rollout.group_ids  # [batch] - which samples share the same prompt
        
        if group_ids is None:
            raise ValueError("GRPO requires group_ids in rollout. Use generate_grpo_rollouts().")
        
        # Get dimensions from first layer
        first_layer_data = next(iter(rollout.layer_data.values()))
        num_experts = first_layer_data["router_logits"].shape[-1]
        controller_dtype = next(controller.parameters()).dtype
        
        # =========================================================================
        # Compute valid_mask: exclude left-padding and post-EOS padding
        # =========================================================================
        query_len = rollout.queries.shape[1]
        response_len = rollout.responses.shape[1]
        seq_len = first_layer_data["router_logits"].shape[1]
        
        query_attention_mask = (rollout.queries != rollout.pad_token_id)
        left_padding_lengths = query_len - query_attention_mask.sum(dim=1)
        response_lengths = rollout.response_lengths.to(device)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_start = left_padding_lengths.unsqueeze(1)
        valid_end = (query_len + response_lengths).unsqueeze(1)
        valid_mask = (positions >= valid_start) & (positions < valid_end)
        
        # =========================================================================
        # GRPO Advantage Computation: group-level baseline
        # =========================================================================
        unique_groups = group_ids.unique()
        advantages = torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        for g in unique_groups:
            mask = (group_ids == g)
            group_rewards = rewards[mask]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std(unbiased=False).clamp(min=1e-8) if group_rewards.numel() > 1 else torch.ones(1, device=device)
            
            # GRPO advantage: (R - mean) / std
            advantages[mask] = (rewards[mask] - group_mean) / group_std
        
        if self.accelerator.is_main_process:
            print(f"  [GRPO-FWD] num_groups={len(unique_groups)}, batch_size={batch_size}", flush=True)
            print(f"  [GRPO-FWD] advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}", flush=True)
            print(f"  [GRPO-FWD] rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}", flush=True)
            print(f"  [GRPO-FWD] valid_mask: {valid_mask.sum().item()} / {valid_mask.numel()} positions valid", flush=True)
        
        # =========================================================================
        # Collect SEPARATE log_probs from all layers (skip V-based advantage for GRPO)
        # =========================================================================
        all_switch_log_probs = []     # List of [batch] tensors
        all_expert_log_probs = []     # List of [batch] tensors
        all_switch_decisions = []     # List of [batch] bool tensors
        all_valid_masks = []          # List of [batch] bool tensors
        
        # NOTE: No total_value_loss accumulator for GRPO (we don't use value function)
        total_switch_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        total_expert_log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        num_total_timesteps = 0
        num_valid_timesteps = 0
        num_switch_timesteps = 0
        total_expert_entropy = 0.0  # Accumulate expert entropy across layers
        num_layers_with_entropy = 0
        
        all_layer_ids = sorted(rollout.layer_data.keys())
        
        for layer_idx in all_layer_ids:
            layer_data = rollout.layer_data[layer_idx]
            layer_controller = self._get_controller_for_layer(layer_idx)
            
            router_logits = layer_data["router_logits"].to(device, dtype=controller_dtype)
            switches = layer_data["switches"].to(device)
            selected_indices = layer_data["selected_indices"].to(device)
            controller_inputs = layer_data["controller_inputs"].to(device, dtype=controller_dtype)
            
            # Process layer - get SEPARATE log_probs
            layer_result = self._process_single_layer(
                layer_idx,  # Layer index for temperature lookup
                layer_controller,
                router_logits,
                switches,
                selected_indices,
                controller_inputs,
                controller_dtype,
                rewards,
                valid_mask=valid_mask,  # Pass valid_mask
            )
            
            # We IGNORE layer_result["advantages"] from V(s_t) - use GRPO advantages instead
            # We also IGNORE layer_result["layer_value_loss"] - GRPO doesn't use value function
            all_switch_log_probs.extend(layer_result["switch_log_probs"])
            all_expert_log_probs.extend(layer_result["expert_log_probs"])
            all_switch_decisions.extend(layer_result["switch_decisions"])
            all_valid_masks.extend(layer_result["valid_masks"])
            
            # NOTE: We skip total_value_loss for GRPO (it's not used for advantage)
            total_switch_log_prob = total_switch_log_prob + layer_result["layer_switch_log_prob"]
            total_expert_log_prob = total_expert_log_prob + layer_result["layer_expert_log_prob"]
            num_total_timesteps += layer_result["num_total_timesteps"]
            num_valid_timesteps += layer_result["num_valid_timesteps"]
            num_switch_timesteps += layer_result["num_switch_timesteps"]
            
            # Accumulate expert entropy
            if "expert_entropy" in layer_result:
                total_expert_entropy += layer_result["expert_entropy"].item()
                num_layers_with_entropy += 1
        
        # =========================================================================
        # Compute SEPARATE policy losses with GRPO advantages
        # Same advantage for ALL VALID timesteps in a trajectory
        # =========================================================================
        if len(all_switch_log_probs) > 0:
            # DEBUG: Check if log probs have gradients
            if self.accelerator.is_main_process and not hasattr(self, '_debug_grad_checked'):
                self._debug_grad_checked = True
                first_switch_lp = all_switch_log_probs[0] if all_switch_log_probs else None
                first_expert_lp = all_expert_log_probs[0] if all_expert_log_probs else None
                print(f"  [DEBUG-GRAD-LOSS] first_switch_lp.requires_grad={first_switch_lp.requires_grad if first_switch_lp is not None else 'N/A'}", flush=True)
                print(f"  [DEBUG-GRAD-LOSS] first_switch_lp.grad_fn={first_switch_lp.grad_fn if first_switch_lp is not None else 'N/A'}", flush=True)
                print(f"  [DEBUG-GRAD-LOSS] first_expert_lp.requires_grad={first_expert_lp.requires_grad if first_expert_lp is not None else 'N/A'}", flush=True)
                print(f"  [DEBUG-GRAD-LOSS] first_expert_lp.grad_fn={first_expert_lp.grad_fn if first_expert_lp is not None else 'N/A'}", flush=True)
            
            total_switch_policy_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
            total_expert_policy_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
            
            for switch_lp, expert_lp, switch_dec, valid_t in zip(
                all_switch_log_probs, all_expert_log_probs, all_switch_decisions, all_valid_masks
            ):
                # GRPO: same advantage for all timesteps in a trajectory
                # Switch loss: Only VALID timesteps (exclude padding)
                switch_contribution = torch.where(
                    valid_t,
                    -advantages * switch_lp,
                    torch.zeros_like(switch_lp)
                )
                total_switch_policy_loss = total_switch_policy_loss + switch_contribution
                
                # Expert loss: Only switch=True AND VALID timesteps
                expert_contribution = torch.where(
                    switch_dec & valid_t,
                    -advantages * expert_lp,
                    torch.zeros_like(expert_lp)
                )
                total_expert_policy_loss = total_expert_policy_loss + expert_contribution
            
            # Normalize each loss by its own count (using VALID counts)
            # NOTE: Use .sum() not .mean() since num_valid_timesteps already includes batch dimension
            mean_switch_loss = total_switch_policy_loss.sum() / max(num_valid_timesteps, 1)
            mean_expert_loss = total_expert_policy_loss.sum() / max(num_switch_timesteps, 1)
            
            # Combined policy loss
            total_policy_loss = mean_switch_loss + mean_expert_loss
            
            # DEBUG: Check final policy loss gradients
            if self.accelerator.is_main_process and not hasattr(self, '_debug_final_grad_checked'):
                self._debug_final_grad_checked = True
                print(f"  [DEBUG-GRAD-LOSS] total_policy_loss.requires_grad={total_policy_loss.requires_grad}", flush=True)
                print(f"  [DEBUG-GRAD-LOSS] total_policy_loss.grad_fn={total_policy_loss.grad_fn}", flush=True)
        else:
            total_policy_loss = torch.zeros(1, device=device, dtype=torch.float32)
            mean_switch_loss = torch.zeros(1, device=device, dtype=torch.float32)
            mean_expert_loss = torch.zeros(1, device=device, dtype=torch.float32)
        
        # GRPO doesn't use value function - return 0
        total_value_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        # Compute mean expert entropy across layers
        mean_expert_entropy = total_expert_entropy / max(num_layers_with_entropy, 1)
        
        if self.accelerator.is_main_process:
            switch_rate = num_switch_timesteps / max(num_valid_timesteps, 1)
            # Use .sum() since num_valid_timesteps already includes batch dimension
            mean_switch_lp = total_switch_log_prob.sum().item() / max(num_valid_timesteps, 1)
            mean_expert_lp = total_expert_log_prob.sum().item() / max(num_switch_timesteps, 1) if num_switch_timesteps > 0 else 0
            print(f"  [GRPO-FWD] num_valid_timesteps={num_valid_timesteps}, num_switch_timesteps={num_switch_timesteps} ({switch_rate:.2%})", flush=True)
            print(f"  [GRPO-FWD] mean_switch_log_prob={mean_switch_lp:.4f}, mean_expert_log_prob={mean_expert_lp:.4f}", flush=True)
            print(f"  [GRPO-FWD] mean_switch_loss={mean_switch_loss.item():.4f}, mean_expert_loss={mean_expert_loss.item():.4f}", flush=True)
            # Log expert entropy (higher = more diverse, lower = mode collapse)
            # Max entropy is log(num_experts)
            print(f"  [GRPO-FWD] expert_entropy={mean_expert_entropy:.4f} (max={math.log(num_experts):.2f})", flush=True)
        
        # Store entropy for wandb logging
        self._last_expert_entropy = mean_expert_entropy
        
        return total_policy_loss, total_value_loss, num_valid_timesteps
    
    def controller_forward_option_critic(
        self,
        rollout: ControllerRollout,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Option-Critic forward pass with per-token TD and deliberation cost.
        
        Implements Harb et al., 2017 (https://arxiv.org/pdf/1709.04571):
        - Per-token reward: r_t = -KL_t - η·switch_t (deliberation cost)
        - TD targets: V_target(t) = r_t + γ·V(t+1), Q_U_target(t) = r_t + γ·next_value
        - Separate advantages (Harb et al. 2017):
          * Termination: adv_term = Q_U - V + η (direct gradient on β, not log-prob REINFORCE)
          * Selection: A_select = Q_U - V (standard REINFORCE with log-prob)
        
        IMPORTANT: This does NOT interfere with GRPO - it's a separate code path.
        
        Returns:
            total_policy_loss: scalar - normalized policy loss
            total_value_loss: [batch] - sum of V and Q_U MSE losses
            num_decisions: int - total number of decisions made
        """
        controller = self._get_controller_module()
        device = rollout.queries.device
        batch_size = rollout.batch_size
        trajectory_reward = rollout.rewards.float().to(device)  # [batch] - fallback if no per_token_kl
        gamma = self.config.gamma
        deliberation_cost = self.config.option_critic_deliberation_cost  # η in Harb et al. 2017
        
        # Get dimensions from first layer
        first_layer_data = next(iter(rollout.layer_data.values()))
        num_experts = first_layer_data["router_logits"].shape[-1]
        controller_dtype = next(controller.parameters()).dtype
        
        # =========================================================================
        # Compute valid_mask: exclude left-padding and post-EOS padding
        # =========================================================================
        query_len = rollout.queries.shape[1]
        response_len = rollout.responses.shape[1]
        seq_len = first_layer_data["router_logits"].shape[1]
        
        query_attention_mask = (rollout.queries != rollout.pad_token_id)
        left_padding_lengths = query_len - query_attention_mask.sum(dim=1)
        response_lengths = rollout.response_lengths.to(device)
        
        # Get terminated flag (True = hit EOS, False = truncated)
        # For truncated sequences, we bootstrap at the boundary instead of treating as terminal
        terminated = rollout.terminated.to(device) if rollout.terminated is not None else torch.ones(batch_size, device=device, dtype=torch.bool)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_start = left_padding_lengths.unsqueeze(1)
        valid_end = (query_len + response_lengths).unsqueeze(1)
        valid_mask = (positions >= valid_start) & (positions < valid_end)
        
        if self.accelerator.is_main_process:
            num_terminated = terminated.sum().item()
            print(f"  [OC-FWD] batch_size={batch_size}, num_layers={len(rollout.layer_data)}, num_experts={num_experts}", flush=True)
            print(f"  [OC-FWD] gamma={gamma}, deliberation_cost={deliberation_cost}", flush=True)
            print(f"  [OC-FWD] valid_mask: {valid_mask.sum().item()} / {valid_mask.numel()} ({100*valid_mask.sum().item()/valid_mask.numel():.1f}%)", flush=True)
            print(f"  [OC-FWD] Terminated: {num_terminated}/{batch_size}, Truncated: {batch_size - num_terminated}", flush=True)
        
        # =========================================================================
        # Per-token rewards: r_t = -KL_t for response tokens
        # Using actual per-token KL from the reward scorer
        # =========================================================================
        # Create per-token reward tensor [batch, seq_len]
        per_token_base_reward = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Use actual per-token KL if available
        if rollout.per_token_kl is not None:
            # per_token_kl is [batch, response_len], need to place in full seq tensor
            per_token_kl = rollout.per_token_kl.to(device=device, dtype=torch.float32)
            response_len = per_token_kl.shape[1]
            
            # Response tokens start at query_len
            # per_token_reward for response positions = -KL_t (negative because lower KL is better)
            for i in range(batch_size):
                resp_start = query_len
                # Clamp to not exceed seq_len AND not exceed available KL tokens
                available_space = seq_len - resp_start
                actual_resp_len = min(response_len, int(rollout.response_lengths[i].item()), available_space)
                if actual_resp_len > 0:
                    per_token_base_reward[i, resp_start:resp_start+actual_resp_len] = -per_token_kl[i, :actual_resp_len]
            
            if self.accelerator.is_main_process:
                print(f"  [OC-FWD] Using TRUE per-token KL rewards (shape={per_token_kl.shape})", flush=True)
                # Count how many response tokens actually got rewards
                response_positions = (positions >= query_len) & valid_mask
                num_response_tokens = response_positions.sum().item()
                num_query_tokens = valid_mask.sum().item() - num_response_tokens
                num_tokens_with_reward = (per_token_base_reward != 0).sum().item()
                print(f"  [OC-FWD] Token breakdown: {num_query_tokens} query + {num_response_tokens} response = {valid_mask.sum().item()} valid", flush=True)
                print(f"  [OC-FWD] Tokens with KL reward: {num_tokens_with_reward}", flush=True)
                print(f"  [OC-FWD] per_token_base_reward stats: min={per_token_base_reward.min().item():.4f}, max={per_token_base_reward.max().item():.4f}", flush=True)
        else:
            # Fallback: distribute trajectory reward uniformly (not recommended)
            if self.accelerator.is_main_process:
                print(f"  [OC-FWD] WARNING: No per_token_kl available, falling back to uniform distribution", flush=True)
            response_mask = (positions >= query_len) & valid_mask
            num_response_tokens = response_mask.sum(dim=1).clamp(min=1).float()
            per_token_base_reward = torch.where(
                response_mask,
                (trajectory_reward / num_response_tokens).unsqueeze(1).expand(-1, seq_len),
                torch.zeros_like(per_token_base_reward)
            )
        
        # =========================================================================
        # Normalize rewards by response length
        # This makes V and Q learn MEAN reward per token instead of SUM.
        # Benefits:
        #   1. Bounded values (V ≈ -0.5 instead of V ≈ -500)
        #   2. Consistent scale across different response lengths
        #   3. Numerical stability for TD learning
        # =========================================================================
        response_len_for_norm = response_lengths.float().clamp(min=1.0)  # [batch]
        per_token_base_reward = per_token_base_reward / response_len_for_norm.unsqueeze(1)
        
        if self.accelerator.is_main_process:
            # Show normalized reward stats
            valid_rewards = per_token_base_reward[valid_mask]
            nonzero_rewards = per_token_base_reward[per_token_base_reward != 0]
            print(f"  [OC-FWD] Reward normalized by response_length (mean={response_len_for_norm.mean().item():.1f})", flush=True)
            print(f"  [OC-FWD] Normalized reward stats: mean={nonzero_rewards.mean().item():.6f}, std={nonzero_rewards.std().item():.6f}", flush=True)
        
        # =========================================================================
        # Process each layer
        # =========================================================================
        total_policy_loss = torch.zeros(1, device=device, dtype=torch.float32)
        total_value_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        num_valid_timesteps = 0
        num_switch_timesteps = 0
        total_expert_entropy = 0.0
        num_layers_with_entropy = 0
        
        all_layer_ids = sorted(rollout.layer_data.keys())
        
        for layer_idx in all_layer_ids:
            layer_data = rollout.layer_data[layer_idx]
            layer_controller = self._get_controller_for_layer(layer_idx)
            
            router_logits = layer_data["router_logits"].to(device, dtype=controller_dtype)
            switches = layer_data["switches"].to(device)  # [batch, seq_len]
            selected_indices = layer_data["selected_indices"].to(device)
            controller_inputs = layer_data["controller_inputs"].to(device, dtype=controller_dtype)
            
            # Process this layer with Option-Critic TD
            layer_result = self._process_single_layer_option_critic(
                layer_idx,
                layer_controller,
                router_logits,
                switches,
                selected_indices,
                controller_inputs,
                controller_dtype,
                per_token_base_reward,
                deliberation_cost,
                gamma,
                valid_mask,
                valid_end,
                terminated,  # True = hit EOS, False = truncated
            )
            
            total_policy_loss = total_policy_loss + layer_result["policy_loss"]
            total_value_loss = total_value_loss + layer_result["value_loss"]
            num_valid_timesteps += layer_result["num_valid_timesteps"]
            num_switch_timesteps += layer_result["num_switch_timesteps"]
            
            if "expert_entropy" in layer_result:
                total_expert_entropy += layer_result["expert_entropy"].item()
                num_layers_with_entropy += 1
        
        # Normalize policy loss by number of layers
        num_layers = len(all_layer_ids)
        total_policy_loss = total_policy_loss / max(num_layers, 1)
        
        # Compute mean expert entropy
        mean_expert_entropy = total_expert_entropy / max(num_layers_with_entropy, 1)
        
        if self.accelerator.is_main_process:
            switch_rate = num_switch_timesteps / max(num_valid_timesteps, 1)
            print(f"  [OC-FWD] num_valid_timesteps={num_valid_timesteps}, num_switch_timesteps={num_switch_timesteps} ({switch_rate:.2%})", flush=True)
            print(f"  [OC-FWD] total_policy_loss={total_policy_loss.item():.4f}", flush=True)
            print(f"  [OC-FWD] expert_entropy={mean_expert_entropy:.4f} (max={math.log(num_experts):.2f})", flush=True)
        
        self._last_expert_entropy = mean_expert_entropy
        
        return total_policy_loss, total_value_loss, num_valid_timesteps
    
    def _process_single_layer_option_critic(
        self,
        layer_idx: int,
        controller,
        router_logits: torch.Tensor,  # [batch, seq_len, num_experts]
        switches: torch.Tensor,  # [batch, seq_len]
        selected_indices: torch.Tensor,  # [batch, seq_len, k]
        controller_inputs: torch.Tensor,  # [batch, seq_len, input_dim]
        controller_dtype: torch.dtype,
        per_token_base_reward: torch.Tensor,  # [batch, seq_len]
        deliberation_cost: float,  # η
        gamma: float,  # discount factor
        valid_mask: torch.Tensor,  # [batch, seq_len]
        valid_end: torch.Tensor,  # [batch, 1] - position where each trajectory ends
        terminated: torch.Tensor,  # [batch] - True if hit EOS, False if truncated
    ) -> Dict[str, Any]:
        """
        Process a single layer with Option-Critic TD updates.
        
        Implements (Harb et al. 2017):
        - Per-token reward: r_t = base_reward_t (no η in reward)
        - GAE targets for V and Q_U (with proper terminal/truncation handling)
        - Termination: adv_term = Q_U - V + η (direct gradient on β)
        - Selection: A_select = Q_U - V (REINFORCE with log-prob)
        """
        device = router_logits.device
        batch_size, seq_len, num_experts = router_logits.shape
        
        # =========================================================================
        # Phase 1: Forward pass - collect V, Q_U_old, Q_U_new, logits at each timestep
        # 
        # CRITICAL: We need TWO Q_U values (Harb et al. 2017):
        #   - Q_U_old: value of the CURRENT option (for termination advantage)
        #   - Q_U_new: value of the NEW option (for selection advantage)
        # 
        # At time t:
        #   - expert_mask (from controller_inputs) = OLD option (before switch decision)
        #   - selected_indices = NEW option (what would be/was selected if switch)
        # =========================================================================
        all_V = []  # V(s_t) values
        all_Q_U_old = []  # Q_U(s_t, o_old) - for termination advantage
        all_Q_U_new = []  # Q_U(s_t, o_new) - for selection advantage
        all_switch_logits = []
        all_candidate_logits = []
        all_expert_masks = []  # Current option (expert mask)
        
        hidden_state = controller.init_hidden(batch_size, device, controller_dtype)
        top_k = selected_indices.shape[-1]  # Number of experts in each option
        
        for t in range(seq_len):
            # Get input and expert mask from recorded controller inputs
            input_type = getattr(controller, 'input_type', 'router_softmax')
            if input_type == "hidden_states":
                hidden_dim = controller.model_hidden_size
                recorded_input = controller_inputs[:, t, :hidden_dim].to(device=device, dtype=controller_dtype)
                expert_mask = controller_inputs[:, t, hidden_dim:].to(device=device, dtype=controller_dtype)
            else:
                recorded_input = controller_inputs[:, t, :num_experts].to(device=device, dtype=controller_dtype)
                expert_mask = controller_inputs[:, t, num_experts:].to(device=device, dtype=controller_dtype)
            
            x_t = torch.cat([recorded_input, expert_mask], dim=-1)
            
            # Forward through controller
            hidden_state, switch_logits, perturbation, value = controller(x_t, hidden_state)
            
            # Clamp logits - MUST match inference exactly!
            switch_logits = switch_logits.clamp(-20, 20)
            
            # Compute Q_U_old: value of OLD/CURRENT option (for termination advantage)
            # Uses dueling-style: Q_U = V + A, where A is option-specific advantage
            # Convert binary expert_mask to indices using topk
            old_option_indices = expert_mask.topk(top_k, dim=-1).indices  # [batch, k]
            q_u_old = controller.compute_q_option(hidden_state, expert_mask, value, selected_indices=old_option_indices)
            
            # Compute Q_U_new: value of NEW option (for selection advantage)
            selected_indices_t = selected_indices[:, t, :]  # [batch, k]
            q_u_new = controller.compute_q_option(hidden_state, expert_mask, value, selected_indices=selected_indices_t)
            
            # Candidate logits with residual connection
            router_logits_t = router_logits[:, t, :].to(perturbation.dtype)
            candidate_logits = router_logits_t + perturbation
            candidate_logits = candidate_logits.clamp(-20, 20)  # MUST match inference
            
            # Apply temperature scaling to match rollout (if temperature != 1.0)
            # This ensures log_probs are computed under the same policy used during rollout
            sampling_temperature = self._get_controller_sampling_temperature(layer_idx)
            if sampling_temperature != 1.0:
                candidate_logits = candidate_logits / sampling_temperature
            
            all_V.append(value)
            all_Q_U_old.append(q_u_old)
            all_Q_U_new.append(q_u_new)
            all_switch_logits.append(switch_logits)
            all_candidate_logits.append(candidate_logits)
            all_expert_masks.append(expert_mask)
        
        # Stack into tensors [batch, seq_len]
        V_values = torch.stack(all_V, dim=1)  # [batch, seq_len]
        Q_U_old_values = torch.stack(all_Q_U_old, dim=1)  # [batch, seq_len] - for termination
        Q_U_new_values = torch.stack(all_Q_U_new, dim=1)  # [batch, seq_len] - for selection
        switch_logits = torch.stack(all_switch_logits, dim=1)  # [batch, seq_len]
        candidate_logits_all = torch.stack(all_candidate_logits, dim=1)  # [batch, seq_len, num_experts]
        
        # =========================================================================
        # Phase 2: Per-token rewards (NO deliberation cost here - faithful to Harb et al. 2017)
        # r_t = base_reward_t (just KL)
        # Deliberation cost η only appears in adv_term, not in reward signal
        # =========================================================================
        per_token_reward = per_token_base_reward  # No deliberation cost in reward
        
        # =========================================================================
        # Phase 3: Compute TD targets using GAE (Generalized Advantage Estimation)
        # 
        # GAE interpolates between TD(0) (λ=0) and Monte Carlo (λ=1):
        #   δ_t = r_t + γ·V(t+1) - V(t)  (TD error)
        #   A^GAE_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        #   V_target = V + A^GAE
        #
        # For Q_U, we use β-weighted bootstrap:
        #   δ^Q_t = r_t + γ·(β·V(t+1) + (1-β)·Q_U(t+1)) - Q_U(t)
        #   A^GAE_Q_t = δ^Q_t + (γλ)·δ^Q_{t+1} + ...
        #   Q_target = Q_U + A^GAE_Q
        #
        # Terminal vs Truncation handling:
        #   - Terminated (hit EOS): No bootstrap at boundary (true terminal)
        #   - Truncated (hit max_length): Bootstrap with V(s_T) at boundary
        #
        # Benefits: λ=0.95 gives much lower bias than TD(0) while keeping variance low
        # =========================================================================
        gae_lambda = self.config.gae_lambda
        
        # Compute termination probabilities for Q_U bootstrap
        beta_probs = torch.sigmoid(switch_logits)  # [batch, seq_len]
        
        # Initialize GAE accumulators
        gae_V = torch.zeros(batch_size, device=device, dtype=torch.float32)
        gae_Q = torch.zeros(batch_size, device=device, dtype=torch.float32)
        V_advantages = torch.zeros_like(V_values)
        Q_advantages = torch.zeros_like(Q_U_old_values)
        
        # Backward pass to compute GAE
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            V_t = V_values[:, t].detach()
            Q_t = Q_U_old_values[:, t].detach()
            
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)  # [batch]
                V_next = V_values[:, t+1].detach()
                Q_next = Q_U_old_values[:, t+1].detach()
                beta_next = beta_probs[:, t+1].detach()
                
                # At trajectory boundary: 
                # - If TRUNCATED (not terminated): bootstrap with V(t+1)
                # - If TERMINATED (hit EOS): no bootstrap (0)
                # `terminated` is [batch], True = hit EOS
                # At boundary (not next_is_valid), use V_next for truncated, 0 for terminated
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated  # Truncated sequences
                
                # TD errors (δ)
                # V: δ_t = r_t + γ·V(t+1) - V(t)
                # For truncated: bootstrap at boundary; for terminated: no bootstrap
                V_bootstrap = torch.where(
                    next_is_valid,
                    gamma * V_next,
                    torch.where(should_bootstrap_at_boundary, gamma * V_next, torch.zeros_like(V_next))
                )
                delta_V = r_t + V_bootstrap - V_t
                
                # Q: δ_t = r_t + γ·(β·V + (1-β)·Q) - Q(t)
                soft_bootstrap = gamma * (beta_next * V_next + (1 - beta_next) * Q_next)
                Q_bootstrap = torch.where(
                    next_is_valid,
                    soft_bootstrap,
                    torch.where(should_bootstrap_at_boundary, soft_bootstrap, torch.zeros_like(soft_bootstrap))
                )
                delta_Q = r_t + Q_bootstrap - Q_t
                
                # GAE accumulation: gae = δ + γλ·gae
                # Reset gae to 0 only at TRUE terminal (terminated), keep accumulating for truncation
                should_continue_gae = next_is_valid | should_bootstrap_at_boundary
                gae_V = torch.where(should_continue_gae, delta_V + gamma * gae_lambda * gae_V, delta_V)
                gae_Q = torch.where(should_continue_gae, delta_Q + gamma * gae_lambda * gae_Q, delta_Q)
            else:
                # Last position in tensor
                # Check if this is a truncated sequence that should bootstrap
                is_last_valid = (t == valid_end.squeeze(1) - 1)  # This is the last valid position
                should_bootstrap = is_last_valid & ~terminated  # Truncated at max seq_len
                
                # For truncated sequences at the very end of the tensor, we can't bootstrap
                # (no t+1 available), so we treat it as terminal. This is rare (only when 
                # response fills the entire tensor with no EOS).
                delta_V = r_t - V_t
                delta_Q = r_t - Q_t
                gae_V = delta_V
                gae_Q = delta_Q
            
            V_advantages[:, t] = gae_V
            Q_advantages[:, t] = gae_Q
        
        # Compute targets: V_target = V + A^GAE
        V_targets = V_values.detach() + V_advantages
        Q_U_old_targets = Q_U_old_values.detach() + Q_advantages
        
        if layer_idx == 0 and hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            num_truncated = (~terminated).sum().item()
            print(f"  [OC-LAYER0] GAE λ={gae_lambda}: V_adv mean={V_advantages[valid_mask].mean().item():.4f}, Q_adv mean={Q_advantages[valid_mask].mean().item():.4f}", flush=True)
            print(f"  [OC-LAYER0] Truncated (bootstrap at boundary): {num_truncated}/{batch_size}", flush=True)
        
        # =========================================================================
        # Phase 4: Compute advantages (Harb et al. 2017)
        # 
        # CRITICAL: Use DIFFERENT Q_U values for different purposes!
        #
        # Termination advantage (for direct β gradient):
        #   adv_term = Q_U_old - V + η
        #   Uses Q_U_old: value of CONTINUING current option
        #   - If positive: continuing is better → push β down
        #   - If negative: switching is better → push β up
        #
        # Selection advantage A_select (for log-prob REINFORCE):
        #   A_select = Q_U_new - V
        #   Uses Q_U_new: value of the NEW option we're selecting
        #   - η cancels in softmax (same for all options)
        # =========================================================================
        # Termination advantage: Q_U_old - V + η
        adv_term = Q_U_old_values.detach() - V_values.detach() + deliberation_cost
        
        # Selection advantage: Q_U_new - V (η cancels out in option selection)
        A_select = Q_U_new_values.detach() - V_values.detach()
        
        # =========================================================================
        # Phase 5: Compute policy losses
        # =========================================================================
        # Termination probability
        switch_probs = torch.sigmoid(switch_logits)
        t_mask = torch.arange(seq_len, device=device).unsqueeze(0) > 0  # Skip t=0
        
        # Normalize selection advantage over the ACTUAL subset used in the loss
        # (switches & valid_mask & t_mask), not all valid positions
        # This gives more accurate standardization and reduces variance
        # Note: We do NOT normalize adv_term because it's used in direct gradient, not REINFORCE
        select_mask = switches & valid_mask & t_mask
        A_select_used = A_select[select_mask]
        if A_select_used.numel() > 1:
            A_select_mean = A_select_used.mean()
            A_select_std = A_select_used.std().clamp(min=1e-8)
            A_select_norm = (A_select - A_select_mean) / A_select_std
        else:
            A_select_norm = A_select
        
        # Termination loss: DIRECT gradient on β, NOT log-prob REINFORCE!
        # Loss = β * (Q_U - V + η) produces gradient ∇β * (Q_U - V + η)
        # - If Q_U - V + η > 0: continuing is better → minimize β → correct!
        # - If Q_U - V + η < 0: switching is better → maximize β → correct!
        term_loss = torch.where(
            valid_mask & t_mask,
            switch_probs * adv_term,  # Direct β, not log β
            torch.zeros_like(switch_probs)
        )
        
        # Selection policy loss: -A_select * log_prob_option (only when switching, t > 0)
        # Compute Plackett-Luce log prob for expert selection
        top_k = selected_indices.shape[-1]
        selection_log_probs = torch.zeros(batch_size, seq_len, device=device, dtype=controller_dtype)
        
        for t in range(1, seq_len):  # Skip t=0
            cand_logits_t = candidate_logits_all[:, t, :]  # [batch, num_experts]
            sel_indices_t = selected_indices[:, t, :]  # [batch, k]
            
            # Plackett-Luce log probability
            log_prob = torch.zeros(batch_size, device=device, dtype=controller_dtype)
            remaining_logits = cand_logits_t.clone()
            
            for k_idx in range(top_k):
                selected_idx = sel_indices_t[:, k_idx]  # [batch]
                selected_logit = remaining_logits.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
                log_normalizer = torch.logsumexp(remaining_logits, dim=-1)
                log_prob = log_prob + (selected_logit - log_normalizer)
                
                # Mask out selected expert for next iteration
                remaining_logits = remaining_logits.scatter(1, selected_idx.unsqueeze(1), float('-inf'))
            
            selection_log_probs[:, t] = log_prob
        
        # Selection policy loss: only when switching AND valid AND t > 0
        select_loss = torch.where(
            switches & valid_mask & t_mask,
            -A_select_norm * selection_log_probs,
            torch.zeros_like(selection_log_probs)
        )
        
        # =========================================================================
        # Phase 6: Compute value losses (Separate V and Q networks)
        # 
        # V_loss: trains value_head to predict state value
        # Q_loss: trains q_head to predict option value
        # 
        # Separate networks avoid split ambiguity and match Option-Critic theory.
        # At convergence, V* = Q* for the optimal option.
        # 
        # We train the EXECUTED option value for Q:
        # - On no-switch steps: executed option is o_old → train Q_U_old
        # - On switch steps: executed option is o_new → train Q_U_new
        # =========================================================================
        V_loss = torch.where(
            valid_mask & t_mask,
            (V_values - V_targets.detach()) ** 2,
            torch.zeros_like(V_values)
        )
        
        # Q_exec = Q_U_new if switch, else Q_U_old
        Q_exec_values = torch.where(switches, Q_U_new_values, Q_U_old_values)
        
        Q_loss = torch.where(
            valid_mask & t_mask,
            (Q_exec_values - Q_U_old_targets.detach()) ** 2,
            torch.zeros_like(Q_exec_values)
        )
        
        # =========================================================================
        # Aggregate losses
        # =========================================================================
        num_valid = (valid_mask & t_mask).sum().item()
        num_switch = (switches & valid_mask & t_mask).sum().item()
        
        # Policy loss: average termination + average selection
        mean_term_loss = term_loss.sum() / max(num_valid, 1)
        mean_select_loss = select_loss.sum() / max(num_switch, 1)
        policy_loss = mean_term_loss + mean_select_loss
        
        # Value loss: V_loss + Q_loss (separate networks)
        value_loss = (V_loss + Q_loss).sum(dim=1)  # [batch]
        
        # Expert entropy (for monitoring)
        with torch.no_grad():
            expert_softmax = torch.softmax(candidate_logits_all, dim=-1)  # [batch, seq_len, num_experts]
            expert_log_softmax = torch.log_softmax(candidate_logits_all, dim=-1)
            entropy_per_position = -(expert_softmax * expert_log_softmax).sum(dim=-1)  # [batch, seq_len]
            expert_entropy = entropy_per_position[valid_mask & t_mask].mean()
        
        # =========================================================================
        # Sanity check printouts (only for layer 0 to avoid spam)
        # =========================================================================
        if layer_idx == 0 and hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            with torch.no_grad():
                valid_V = V_values[valid_mask & t_mask]
                valid_Q_U_old = Q_U_old_values[valid_mask & t_mask]
                valid_Q_U_new = Q_U_new_values[valid_mask & t_mask]
                valid_Q_exec = Q_exec_values[valid_mask & t_mask]
                valid_V_target = V_targets[valid_mask & t_mask]
                valid_Q_target = Q_U_old_targets[valid_mask & t_mask]
                valid_adv_term = adv_term[valid_mask & t_mask]
                valid_A_select = A_select[valid_mask & t_mask]
                valid_reward = per_token_reward[valid_mask]
                
                # TD error (should decrease over training)
                td_error_V = (valid_V - valid_V_target).abs().mean().item()
                td_error_Q = (valid_Q_exec - valid_Q_target).abs().mean().item()
                
                # Count switches
                num_switches_in_valid = switches[valid_mask & t_mask].sum().item()
                
                print(f"  [OC-LAYER0] === OPTION-CRITIC SANITY CHECK ===", flush=True)
                print(f"  [OC-LAYER0] V: mean={valid_V.mean().item():.4f}, std={valid_V.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] Q_U_old (for term): mean={valid_Q_U_old.mean().item():.4f}, std={valid_Q_U_old.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] Q_U_new (for select): mean={valid_Q_U_new.mean().item():.4f}, std={valid_Q_U_new.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] Q_exec (trained): mean={valid_Q_exec.mean().item():.4f}, std={valid_Q_exec.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] V_target: mean={valid_V_target.mean().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] Q_target: mean={valid_Q_target.mean().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] TD_error: V={td_error_V:.4f}, Q_exec={td_error_Q:.4f} (should decrease)", flush=True)
                print(f"  [OC-LAYER0] adv_term (Q_U_old-V+η): mean={valid_adv_term.mean().item():.4f}, std={valid_adv_term.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] A_select (Q_U_new-V): mean={valid_A_select.mean().item():.4f}, std={valid_A_select.std().item():.4f}", flush=True)
                print(f"  [OC-LAYER0] per_token_reward (normalized, no η): mean={valid_reward.mean().item():.6f}, min={valid_reward.min().item():.6f}, max={valid_reward.max().item():.6f}", flush=True)
                print(f"  [OC-LAYER0] switch_prob: mean={switch_probs[valid_mask & t_mask].mean().item():.4f}, num_switches={num_switches_in_valid}", flush=True)
                print(f"  [OC-LAYER0] term_loss: {mean_term_loss.item():.4f}, select_loss: {mean_select_loss.item():.4f}", flush=True)
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "num_valid_timesteps": num_valid,
            "num_switch_timesteps": num_switch,
            "expert_entropy": expert_entropy,
        }
    
    def compute_loss(
        self,
        rollout: ControllerRollout,
        total_policy_loss: torch.Tensor,
        total_value_loss: torch.Tensor,
        num_decisions: int,
        scale_factor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Combine policy loss and value loss.
        
        NOTE: total_policy_loss is now ALREADY NORMALIZED (mean_switch_loss + mean_expert_loss).
        We only need to normalize value_loss here.
        
        Args:
            rollout: ControllerRollout with rewards
            total_policy_loss: scalar - ALREADY NORMALIZED policy loss (mean_switch + mean_expert)
            total_value_loss: [batch] - SUM of (V(s_t) - R)^2 across all decisions
            num_decisions: int - number of decisions (for value loss normalization)
            scale_factor: float - loss scaling factor (1/gradient_accumulation_steps)
            
        Returns:
            Dict with loss components
        """
        rewards = rollout.rewards.float()
        
        # Policy loss is ALREADY normalized (mean_switch + mean_expert)
        # Just ensure it's a scalar
        policy_loss = total_policy_loss.mean() if total_policy_loss.numel() > 1 else total_policy_loss
        
        # Value loss still needs normalization
        # Use .sum() since num_decisions already includes batch dimension
        value_loss = total_value_loss.sum()
        mean_value_loss = value_loss / max(num_decisions, 1)
        
        # Total loss - policy is already normalized, only normalize value
        # Scale by factor for gradient accumulation
        total_loss = (policy_loss + self.config.value_coef * mean_value_loss) * scale_factor
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,        # Already normalized
            "value_loss": mean_value_loss,     # Per-decision for logging
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std() if rewards.numel() > 1 else torch.tensor(0.0),
            "num_decisions": num_decisions,
        }
    
    # =========================================================================
    # Training Loop with Gradient Accumulation
    # =========================================================================
    
    def train_step_with_accumulation(
        self,
        batch_queries: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Training step with per-timestep advantage computation.
        
        For each timestep t: A_t = R - V(s_t)
        Policy loss: Σ_t (-A_t * log π(a_t|s_t))
        Value loss: Σ_t (V(s_t) - R)²
        
        Args:
            batch_queries: List of [batch, query_len] tensors, one per accumulation step
            
        Returns:
            Dict with metrics
        """
        self.global_step += 1
        step_start = time.time()
        grad_accum_steps = len(batch_queries)
        
        # DEBUG: Print parameter values at START of step (to verify persistence)
        use_grpo = self.config.advantage_method == "grpo"
        use_option_critic = self.config.advantage_method == "option_critic"
        if self.accelerator.is_main_process:
            controller = self._get_controller_module()
            switch_bias = controller.switch_head.bias.data.item()
            logit_bias = controller.expert_head.bias.data[:3].tolist()
            print(f"  [DEBUG-STEP-START] Step {self.global_step}: switch_head.bias={switch_bias:.6f}, expert_head.bias[:3]={logit_bias}", flush=True)
            if use_grpo:
                method_str = "GRPO (group-level baseline)"
            elif use_option_critic:
                method_str = "Option-Critic (Harb et al., 2017, per-token TD)"
            else:
                method_str = f"Unknown advantage method: {self.config.advantage_method}"
            print(f"  [BATCH-ACCUM] Processing {grad_accum_steps} rollouts with {method_str}", flush=True)
        
        # =====================================================================
        # Phase 1: Generate all rollouts (no gradients)
        # =====================================================================
        rollout_start = time.time()
        rollouts = []
        all_local_rewards = []
        all_base_rewards = []
        all_response_lengths = []
        
        # Reset reward scorer stats for this training step (for proper accumulation)
        if hasattr(self.ppl_scorer, 'reset_batch_stats'):
            self.ppl_scorer.reset_batch_stats()
        
        for accum_idx, queries in enumerate(batch_queries):
            if self.accelerator.is_main_process:
                print(f"  [ROLLOUT {accum_idx+1}/{grad_accum_steps}] Generating rollout...", flush=True)
            
            if use_grpo:
                # GRPO: generate multiple rollouts per prompt
                rollout = self.generate_grpo_rollouts(queries, self.config.num_generations_per_prompt)
            else:
                # Option-Critic: generate one rollout per prompt (same as old PPO)
                rollout = self.generate_rollout(queries)
            
            rollouts.append(rollout)
            all_local_rewards.append(rollout.rewards)
            all_base_rewards.append(rollout.base_rewards)
            all_response_lengths.append(rollout.response_lengths)
        
        rollout_time = time.time() - rollout_start
        
        # Concatenate all local rewards/base_rewards/response_lengths for logging
        local_rewards = torch.cat(all_local_rewards, dim=0)
        local_base_rewards = torch.cat(all_base_rewards, dim=0)
        local_response_lengths = torch.cat(all_response_lengths, dim=0)
        
        if self.accelerator.is_main_process:
            print(f"  [BATCH-ACCUM] Collected {local_rewards.shape[0]} local samples", flush=True)
            print(f"  [BATCH-ACCUM] Rewards: mean={local_rewards.mean().item():.4f}, std={local_rewards.std().item() if local_rewards.numel() > 1 else 0:.4f}", flush=True)
        
        # =====================================================================
        # Phase 2: Advantage computation depends on method
        # =====================================================================
        # PPO: A_t = R - V(s_t), then normalize globally
        # GRPO: A = (R - group_mean) / (group_std + eps), same for all timesteps
        
        if self.accelerator.is_main_process:
            if use_grpo:
                print(f"  [GRPO] Using group-level baseline (no V(s_t))", flush=True)
                print(f"  [GRPO] num_generations_per_prompt={self.config.num_generations_per_prompt}", flush=True)
            elif use_option_critic:
                print(f"  [OPTION-CRITIC] Using per-token TD with deliberation cost (Harb et al., 2017)", flush=True)
                print(f"  [OPTION-CRITIC] gamma={self.config.gamma}, deliberation_cost={self.config.option_critic_deliberation_cost}", flush=True)
            print(f"  [REWARDS] Raw rewards: mean={local_rewards.mean().item():.4f}, std={local_rewards.std().item() if local_rewards.numel() > 1 else 0:.4f}", flush=True)
        
        # =====================================================================
        # Phase 3: Controller forward and loss with gradient accumulation
        # =====================================================================
        self.model.train()
        update_start = time.time()
        
        # Loss scaling factor
        scale_factor = 1.0 / grad_accum_steps
        
        # Accumulators for metrics
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_num_decisions = 0
        total_switch_rate = 0.0
        
        # Multiple update epochs on the same batch (like PPO)
        for epoch_idx in range(self.config.num_update_epochs):
            if self.accelerator.is_main_process and epoch_idx == 0:
                print(f"  [UPDATE] Running {self.config.num_update_epochs} update epochs with {grad_accum_steps} accumulation steps...", flush=True)
            
            # Zero gradients at start of each epoch
            self.optimizer.zero_grad()
            
            for accum_idx, rollout in enumerate(rollouts):
                # Forward - computes advantages based on method
                if use_grpo:
                    # GRPO: group-level baseline
                    policy_loss_sum, value_loss_sum, num_decisions = self.controller_forward_grpo(rollout)
                elif use_option_critic:
                    # Option-Critic: per-token TD with Q_U and deliberation cost
                    policy_loss_sum, value_loss_sum, num_decisions = self.controller_forward_option_critic(rollout)
                else:
                    raise ValueError(f"Unknown advantage_method: {self.config.advantage_method}")
                
                # Check for NaN in forward outputs (first accumulation step, first epoch)
                if self.accelerator.is_main_process and epoch_idx == 0 and accum_idx == 0:
                    pl_finite = torch.isfinite(policy_loss_sum).all().item()
                    # policy_loss_sum is ALREADY normalized (mean_switch + mean_expert), don't divide again
                    mean_pl = policy_loss_sum.mean().item() if policy_loss_sum.numel() > 1 else policy_loss_sum.item()
                    if use_grpo:
                        # GRPO doesn't use value loss for advantage, skip it
                        print(f"  [UPDATE] policy_loss={mean_pl:.4f}, num_decisions={num_decisions}, finite={pl_finite}", flush=True)
                    elif use_option_critic:
                        # Option-Critic uses both V and Q_U value losses
                        vl_finite = torch.isfinite(value_loss_sum).all().item()
                        mean_vl = value_loss_sum.mean().item() / max(num_decisions, 1)
                        print(f"  [UPDATE-OC] policy_loss={mean_pl:.4f}, value_loss={mean_vl:.4f}, num_decisions={num_decisions}, finite={pl_finite and vl_finite}", flush=True)
                
                # Compute final loss
                losses = self.compute_loss(
                    rollout, policy_loss_sum, value_loss_sum, num_decisions,
                    scale_factor=scale_factor,
                )
                
                # Check for NaN in loss (first accumulation step, first epoch)
                loss_finite = torch.isfinite(losses["loss"]).item()
                if self.accelerator.is_main_process and epoch_idx == 0 and accum_idx == 0:
                    print(f"  [UPDATE] loss={losses['loss'].item():.4f} (scaled by {scale_factor:.4f}), finite={loss_finite}", flush=True)
                
                # Skip backward if loss is NaN/Inf to prevent corrupting parameters
                if not loss_finite:
                    if self.accelerator.is_main_process:
                        print(f"  [WARNING] Skipping backward due to non-finite loss", flush=True)
                    continue
                
                # Backward - accumulate gradients
                losses["loss"].backward()
                
                # Accumulate metrics (only on first epoch to avoid double counting)
                if epoch_idx == 0:
                    total_loss += losses["policy_loss"].item() + self.config.value_coef * losses["value_loss"].item()
                    total_policy_loss += losses["policy_loss"].item()
                    total_value_loss += losses["value_loss"].item()
                    total_num_decisions += num_decisions
                    # Compute switch rate (fraction of token-layer pairs that switched)
                    # Uses real sequence length (excluding left and right padding)
                    switch_counts = rollout.get_total_switch_count()  # [batch]
                    num_layers = rollout.num_layers
                    real_seq_len = rollout.get_real_seq_length().float()  # [batch] - excludes padding
                    # Normalize by both seq_len AND num_layers -> value between 0 and 1
                    switch_rate = switch_counts / (real_seq_len * num_layers).clamp(min=1)  # [batch]
                    total_switch_rate += switch_rate.mean().item()
            
            # =========================================================
            # CRITICAL: Sync gradients across all GPUs before optimizer step
            # We use manual all_reduce because the controller params are a subset
            # of the model, and accelerator/DDP may not properly handle them.
            # =========================================================
            if torch.distributed.is_initialized():
                for p in self.controller_params:
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            
            # Check gradients before clipping (first epoch only)
            if self.accelerator.is_main_process and epoch_idx == 0:
                grad_norms = []
                grad_info = []  # (name, norm) tuples for debugging
                nan_count = 0
                # Get parameter names for debugging
                model = self.model.module if hasattr(self.model, 'module') else self.model
                param_names = {id(p): n for n, p in model.named_parameters() if 'controller' in n}
                
                for p in self.controller_params:
                    if p.grad is not None:
                        g = p.grad.float()
                        if not torch.isfinite(g).all():
                            nan_count += 1
                        gnorm = g.norm().item()
                        grad_norms.append(gnorm)
                        # Track param name if available
                        pname = param_names.get(id(p), f"param_{id(p)}")
                        grad_info.append((pname, gnorm))
                
                if grad_norms:
                    print(f"  [UPDATE] grad_norm: mean={sum(grad_norms)/len(grad_norms):.4e}, max={max(grad_norms):.4e}, nan_params={nan_count}/{len(grad_norms)}", flush=True)
                    # Print top 5 largest gradient parameters
                    grad_info.sort(key=lambda x: x[1], reverse=True)
                    print(f"  [UPDATE] Top 5 grad params: ", flush=True)
                    for name, gnorm in grad_info[:5]:
                        print(f"    {name}: {gnorm:.4e}", flush=True)
                else:
                    print(f"  [UPDATE] NO gradients computed! Check requires_grad on controller params", flush=True)
            
            # DEBUG: Check gradients BEFORE clipping
            if self.accelerator.is_main_process and epoch_idx == 0:
                controller = self._get_controller_module()
                if controller.switch_head.bias.grad is not None:
                    switch_bias_grad_pre_clip = controller.switch_head.bias.grad.item()
                    print(f"  [DEBUG-GRAD-PRE-CLIP] switch_head.bias.grad={switch_bias_grad_pre_clip:.6e}", flush=True)
                
                # Compute total grad norm before clipping
                total_norm_sq = 0.0
                for p in self.controller_params:
                    if p.grad is not None:
                        total_norm_sq += p.grad.float().norm().item() ** 2
                total_norm_pre_clip = total_norm_sq ** 0.5
                print(f"  [DEBUG-GRAD-PRE-CLIP] total_grad_norm={total_norm_pre_clip:.6e}", flush=True)
            
            # Gradient clipping
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.controller_params,
                    self.config.max_grad_norm,
                )
            
            # DEBUG: Check parameter values BEFORE optimizer step (first epoch only)
            if self.accelerator.is_main_process and epoch_idx == 0:
                controller = self._get_controller_module()
                switch_bias_before = controller.switch_head.bias.data.clone().item()
                logit_bias_before = controller.expert_head.bias.data[:3].clone().tolist()
                print(f"  [DEBUG-PARAM] BEFORE step: switch_head.bias={switch_bias_before:.6f}, expert_head.bias[:3]={logit_bias_before}", flush=True)
                
                # Detailed gradient analysis AFTER clipping
                if controller.switch_head.bias.grad is not None:
                    switch_bias_grad = controller.switch_head.bias.grad.item()
                    print(f"  [DEBUG-GRAD-POST-CLIP] switch_head.bias.grad={switch_bias_grad:.6e}", flush=True)
                
                # Check GRU cell gradients (canonical RNN)
                gru_grad_norm = 0.0
                for name, param in controller.gru_cell.named_parameters():
                    if param.grad is not None:
                        gru_grad_norm += param.grad.norm().item() ** 2
                gru_grad_norm = gru_grad_norm ** 0.5
                print(f"  [DEBUG-GRAD-POST-CLIP] gru_cell grad norm={gru_grad_norm:.6e}", flush=True)
            
            # Optimizer step (after all accumulation steps)
            self.optimizer.step()
            
            # Clamp switch_head.bias for ALL layers to prevent switch probability collapse
            # -10 corresponds to ~0.005% switch probability (sigmoid(-10) ≈ 4.5e-5)
            # +5 corresponds to ~99.3% switch probability (sigmoid(5) ≈ 0.993)
            # This prevents the model from completely shutting off switches or always switching
            min_switch_bias = -10.0
            max_switch_bias = 5.0
            num_clamped = self._clamp_all_switch_biases(min_switch_bias, max_switch_bias)
            if self.accelerator.is_main_process and epoch_idx == 0 and num_clamped > 0:
                print(f"  [CLAMP] Clamped switch_head.bias for {num_clamped} layers to [{min_switch_bias}, {max_switch_bias}]", flush=True)
            
            # DEBUG: Check parameter values AFTER optimizer step (first epoch only)
            if self.accelerator.is_main_process and epoch_idx == 0:
                controller = self._get_controller_module()  # Layer 0 for debug logging
                switch_bias_after = controller.switch_head.bias.data.clone().item()
                logit_bias_after = controller.expert_head.bias.data[:3].clone().tolist()
                print(f"  [DEBUG-PARAM] AFTER step (layer 0): switch_head.bias={switch_bias_after:.6f}, expert_head.bias[:3]={logit_bias_after}", flush=True)
                print(f"  [DEBUG-PARAM] CHANGE: switch_head.bias delta={switch_bias_after - switch_bias_before:.6e}", flush=True)
        
        update_time = time.time() - update_start
        step_time = time.time() - step_start
        
        # Aggregate metrics across all GPUs using reduce (more robust than gather)
        # This ensures mean/std are computed over ALL samples across all GPUs
        device = local_rewards.device
        
        # Compute local sums and counts for rewards
        local_reward_sum = local_rewards.sum()
        local_reward_sq_sum = (local_rewards ** 2).sum()
        local_base_reward_sum = local_base_rewards.sum()
        local_base_reward_sq_sum = (local_base_rewards ** 2).sum()
        local_resp_len_sum = local_response_lengths.float().sum()
        local_resp_len_sq_sum = (local_response_lengths.float() ** 2).sum()
        local_count = torch.tensor(float(local_rewards.numel()), device=device)
        # Also sync switch_rate (total_switch_rate is a Python float, need to convert)
        local_switch_rate_sum = torch.tensor(total_switch_rate, device=device)
        
        # Reduce across all GPUs (all ranks must participate)
        total_reward_sum = self.accelerator.reduce(local_reward_sum, reduction='sum')
        total_reward_sq_sum = self.accelerator.reduce(local_reward_sq_sum, reduction='sum')
        total_base_reward_sum = self.accelerator.reduce(local_base_reward_sum, reduction='sum')
        total_base_reward_sq_sum = self.accelerator.reduce(local_base_reward_sq_sum, reduction='sum')
        total_resp_len_sum = self.accelerator.reduce(local_resp_len_sum, reduction='sum')
        total_resp_len_sq_sum = self.accelerator.reduce(local_resp_len_sq_sum, reduction='sum')
        total_count = self.accelerator.reduce(local_count, reduction='sum')
        global_switch_rate_sum = self.accelerator.reduce(local_switch_rate_sum, reduction='sum')
        
        # Compute global mean and std
        global_reward_mean = (total_reward_sum / total_count).item()
        global_reward_var = (total_reward_sq_sum / total_count) - (total_reward_sum / total_count) ** 2
        global_reward_std = global_reward_var.clamp(min=0).sqrt().item()
        
        global_base_reward_mean = (total_base_reward_sum / total_count).item()
        global_base_reward_var = (total_base_reward_sq_sum / total_count) - (total_base_reward_sum / total_count) ** 2
        global_base_reward_std = global_base_reward_var.clamp(min=0).sqrt().item()
        
        global_resp_len_mean = (total_resp_len_sum / total_count).item()
        global_resp_len_var = (total_resp_len_sq_sum / total_count) - (total_resp_len_sum / total_count) ** 2
        global_resp_len_std = global_resp_len_var.clamp(min=0).sqrt().item()
        
        total_batch_size = int(total_count.item())
        
        # Compute global switch_rate (average across all GPUs and accumulation steps)
        num_gpus = self.accelerator.num_processes
        global_switch_rate = global_switch_rate_sum.item() / (grad_accum_steps * num_gpus)
        
        # Average metrics over accumulation steps
        n_accum = float(grad_accum_steps)
        metrics = {
            "loss": total_loss / n_accum,
            "policy_loss": total_policy_loss / n_accum,
            "value_loss": total_value_loss / n_accum,
            "reward_mean": global_reward_mean,
            "reward_std": global_reward_std,
            "base_reward_mean": global_base_reward_mean,  # Quality score (without latency penalty)
            "base_reward_std": global_base_reward_std,
            "response_length_mean": global_resp_len_mean,
            "response_length_std": global_resp_len_std,
            "num_decisions": total_num_decisions,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "step_time": step_time,
            "switch_rate": global_switch_rate,  # Now global across all GPUs
            "batch_size": total_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
        }
        
        return metrics
    
    def train(self):
        """
        Main training loop with per-timestep advantage computation.
        
        For each timestep t: A_t = R - V(s_t)
        This provides state-dependent credit assignment through the value baseline.
        """
        config = self.config
        accelerator = self.accelerator
        grad_accum_steps = config.gradient_accumulation_steps
        
        # Count total optimizer steps (not total batches)
        total_batches = len(self.train_dataloader) * config.num_train_epochs
        total_optimizer_steps = total_batches // grad_accum_steps
        
        if accelerator.is_main_process:
            print("=" * 60)
            print("Starting Controller Training with Per-Timestep Advantages")
            print("=" * 60)
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Per-device batch size: {config.per_device_train_batch_size}")
            print(f"  Gradient accumulation steps: {grad_accum_steps}")
            print(f"  Effective batch size: {config.per_device_train_batch_size * accelerator.num_processes * grad_accum_steps}")
            print(f"  Update epochs per batch: {config.num_update_epochs}")
            print(f"  Response length: {config.response_length}")
            print(f"  Value coefficient: {config.value_coef}")
            print(f"  Latency cost per switch: {config.latency_cost_per_switch}")
            print(f"  Total batches: {total_batches}")
            print(f"  Total optimizer steps: {total_optimizer_steps}")
            print("=" * 60)
        
        # Training loop
        start_time = time.time()
        
        # Track running metrics for display
        running_metrics = {
            "loss": 0.0,
            "reward": 0.0,
            "switches": 0.0,
        }
        running_count = 0
        
        for epoch in range(config.num_train_epochs):
            self.epoch = epoch
            
            # Create progress bar (only on main process)
            # Progress is per optimizer step, not per batch
            if accelerator.is_main_process:
                pbar = tqdm(
                    total=len(self.train_dataloader) // grad_accum_steps,
                    desc=f"Epoch {epoch+1}/{config.num_train_epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )
            
            # Collect batches for gradient accumulation
            batch_queries = []
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                queries = batch["input_ids"]
                batch_queries.append(queries)
                
                # Check if we've collected enough batches for an optimizer step
                if len(batch_queries) < grad_accum_steps:
                    continue
                
                # Training step with batch-wise advantage normalization
                metrics = self.train_step_with_accumulation(batch_queries)
                
                # Clear the batch buffer
                batch_queries = []
                
                # Update running metrics
                running_metrics["loss"] += metrics["loss"]
                running_metrics["reward"] += metrics["reward_mean"]
                running_metrics["switches"] += metrics["switch_rate"]
                running_count += 1
                
                # Update progress bar
                if accelerator.is_main_process:
                    avg_loss = running_metrics["loss"] / running_count
                    avg_reward = running_metrics["reward"] / running_count
                    avg_switches = running_metrics["switches"] / running_count
                    
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward_mean']:.4f}",
                        "batch": f"{metrics['batch_size']}",
                        "switch_rate": f"{metrics['switch_rate']:.4f}",
                        "time": f"{metrics['step_time']:.1f}s",
                    })
                    pbar.update(1)
                
                # Detailed logging
                if self.global_step % config.logging_steps == 0:
                    if accelerator.is_main_process:
                        elapsed = time.time() - start_time
                        is_grpo = self.config.advantage_method == "grpo"
                        # Conditional value_loss in log (only for PPO)
                        value_loss_str = "" if is_grpo else f"value_loss={metrics['value_loss']:.4f} "
                        tqdm.write(
                            f"[Step {self.global_step}/{total_optimizer_steps}] "
                            f"loss={metrics['loss']:.4f} "
                            f"policy={metrics['policy_loss']:.4f} "
                            f"{value_loss_str}"
                            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} "
                            f"base_reward={metrics['base_reward_mean']:.4f} "
                            f"resp_len={metrics['response_length_mean']:.1f} "
                            f"switch_rate={metrics['switch_rate']:.4f} "
                            f"time={metrics['step_time']:.1f}s "
                            f"(rollout={metrics['rollout_time']:.1f}s, update={metrics['update_time']:.1f}s)"
                        )
                        
                        # Log to wandb
                        if self.wandb_run is not None:
                            log_dict = {
                                "train/loss": metrics["loss"],
                                "train/policy_loss": metrics["policy_loss"],
                                "train/reward_mean": metrics["reward_mean"],
                                "train/reward_std": metrics["reward_std"],
                                "train/base_reward_mean": metrics["base_reward_mean"],
                                "train/base_reward_std": metrics["base_reward_std"],
                                "train/response_length_mean": metrics["response_length_mean"],
                                "train/response_length_std": metrics["response_length_std"],
                                "train/num_decisions": metrics["num_decisions"],
                                "train/switch_rate": metrics["switch_rate"],
                                "train/batch_size": metrics["batch_size"],
                                "timing/step_time": metrics["step_time"],
                                "timing/rollout_time": metrics["rollout_time"],
                                "timing/update_time": metrics["update_time"],
                                "progress/epoch": epoch,
                                "progress/global_step": self.global_step,
                            }
                            # Only log value_loss for PPO
                            if not is_grpo:
                                log_dict["train/value_loss"] = metrics["value_loss"]
                            # Add expert entropy (mode collapse indicator)
                            if hasattr(self, '_last_expert_entropy'):
                                log_dict["train/expert_entropy"] = self._last_expert_entropy
                            # Finalize accumulated stats before logging
                            if self.ppl_scorer is not None and hasattr(self.ppl_scorer, 'finalize_batch_stats'):
                                self.ppl_scorer.finalize_batch_stats()
                            
                            # Add reward-specific metrics if available
                            if self.ppl_scorer is not None:
                                # Check which type of scorer we have
                                if hasattr(self.ppl_scorer, 'last_batch_kl_mean'):
                                    # KLReward scorer
                                    log_dict["reward/kl_mean"] = self.ppl_scorer.last_batch_kl_mean
                                    log_dict["reward/kl_std"] = self.ppl_scorer.last_batch_kl_std
                                    log_dict["reward/teacher_ppl_mean"] = self.ppl_scorer.last_batch_teacher_ppl_mean
                                    log_dict["reward/student_ppl_mean"] = self.ppl_scorer.last_batch_student_ppl_mean
                                elif hasattr(self.ppl_scorer, 'last_batch_log_ppl_mean'):
                                    # PerplexityReward scorer
                                    log_dict["reward/log_ppl_mean"] = self.ppl_scorer.last_batch_log_ppl_mean
                                    log_dict["reward/log_ppl_std"] = self.ppl_scorer.last_batch_log_ppl_std
                                    log_dict["reward/repetition_rate_mean"] = self.ppl_scorer.last_batch_repetition_rate_mean
                                    log_dict["reward/repetition_rate_std"] = self.ppl_scorer.last_batch_repetition_rate_std
                            wandb.log(log_dict, step=self.global_step)
                
                # Save checkpoint
                if config.save_steps > 0 and self.global_step % config.save_steps == 0:
                    self.save_checkpoint()
                
                # Garbage collection
                empty_cache()
                gc.collect()
            
            # Handle remaining batches (if total batches not divisible by grad_accum_steps)
            if len(batch_queries) > 0:
                if accelerator.is_main_process:
                    print(f"  [WARN] {len(batch_queries)} remaining batches at end of epoch (not divisible by grad_accum_steps={grad_accum_steps})")
                    print(f"  [WARN] Processing remaining batches with smaller effective batch size...")
                
                metrics = self.train_step_with_accumulation(batch_queries)
                batch_queries = []
                
                running_metrics["loss"] += metrics["loss"]
                running_metrics["reward"] += metrics["reward_mean"]
                running_metrics["switches"] += metrics["switch_rate"]
                running_count += 1
                
                if accelerator.is_main_process:
                    pbar.update(1)
            
            # Close progress bar
            if accelerator.is_main_process:
                pbar.close()
                
                # Log epoch-level metrics
                if running_count > 0:
                    avg_loss = running_metrics["loss"] / running_count
                    avg_reward = running_metrics["reward"] / running_count
                    avg_switches = running_metrics["switches"] / running_count
                    
                    print(f"\n[Epoch {epoch+1}] Avg loss: {avg_loss:.4f}, Avg reward: {avg_reward:.4f}, Avg switches: {avg_switches:.1f}")
                    
                    if self.wandb_run is not None:
                        wandb.log({
                            "epoch/avg_loss": avg_loss,
                            "epoch/avg_reward": avg_reward,
                            "epoch/avg_switches": avg_switches,
                        }, step=self.global_step)
            
            # Synchronize all ranks at epoch boundary to prevent NCCL timeout
            # This ensures rank 0 finishes logging before others start the next epoch's DataLoader iteration
            accelerator.wait_for_everyone()
            
            # Reset running metrics for next epoch
            running_metrics = {"loss": 0.0, "reward": 0.0, "switches": 0.0}
            running_count = 0
        
        # Final save
        self.save_checkpoint()
        
        if accelerator.is_main_process:
            total_time = time.time() - start_time
            print("=" * 60)
            print(f"Training completed in {total_time:.1f}s")
            print(f"Total optimizer steps: {self.global_step}")
            print("=" * 60)
            
            # Finish wandb
            if self.wandb_run is not None:
                wandb.finish()
    
    def save_checkpoint(self):
        """Save controller weights."""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get controller state dict
        unwrapped = self.accelerator.unwrap_model(self.model)
        controller_state = {
            k: v.cpu() for k, v in unwrapped.state_dict().items()
            if "controller" in k
        }
        
        # Save
        checkpoint_path = output_dir / f"controller_step_{self.global_step}.pt"
        torch.save({
            "step": self.global_step,
            "epoch": self.epoch,
            "controller_state_dict": controller_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        
        print(f"[CHECKPOINT] Saved to {checkpoint_path}")
        
        # Also save latest
        latest_path = output_dir / "controller_latest.pt"
        torch.save({
            "step": self.global_step,
            "epoch": self.epoch,
            "controller_state_dict": controller_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load controller weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            load_optimizer: Whether to also load optimizer state (for resuming training)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.accelerator.is_main_process:
            print(f"[CHECKPOINT] Loading from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Load controller state dict
        controller_state = checkpoint.get("controller_state_dict", {})
        if not controller_state:
            raise ValueError("No controller_state_dict found in checkpoint")
        
        # Load into model
        unwrapped = self.accelerator.unwrap_model(self.model)
        current_state = unwrapped.state_dict()
        
        # Update only controller keys
        loaded_keys = []
        for k, v in controller_state.items():
            if k in current_state:
                current_state[k] = v.to(current_state[k].device)
                loaded_keys.append(k)
            else:
                print(f"  [WARN] Key {k} from checkpoint not found in model")
        
        unwrapped.load_state_dict(current_state)
        
        # Load optimizer state if requested
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.accelerator.is_main_process:
                    print(f"  [CHECKPOINT] Loaded optimizer state")
            except Exception as e:
                print(f"  [WARN] Failed to load optimizer state: {e}")
        
        # Restore step and epoch
        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        if self.accelerator.is_main_process:
            print(f"  [CHECKPOINT] Loaded {len(loaded_keys)} controller parameters")
            print(f"  [CHECKPOINT] Resuming from step {self.global_step}, epoch {self.epoch}")
        
        return checkpoint

