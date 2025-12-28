#!/usr/bin/env python3
"""
Activation-based Controller Trainer for gpt-oss MoE models.

This module implements training for the activation-based controller that uses 
LLM hidden states directly instead of learning an RNN hidden state.

Key architecture:
- Termination head: MLP(concat(h, s)) → switch_logit
- Selection head: Linear(h) → expert_logits (initialized from router)
- V head: Linear(h) → value
- Q head: MLP(concat(h, s)) → Q_U(s, option)

Where h = LLM hidden state, s = DeepSets embedding of current expert set.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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

# Import from controller_trainer
from controller_trainer import ControllerTrainerConfig, ControllerRollout


# =============================================================================
# Helper Functions
# =============================================================================

def _plackett_luce_logprob_batched(
    logits: torch.Tensor,
    selections: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of selections under Plackett-Luce distribution.
    
    BATCHED VERSION: Processes all samples in parallel.
    
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
        
        # Compute log_softmax for ALL samples
        log_probs = F.log_softmax(remaining, dim=-1)
        
        # Gather log prob for selected expert for each sample
        gathered = log_probs[batch_indices, valid_indices]
        
        # Only add to total for active samples
        total_logprob = total_logprob + torch.where(active, gathered, torch.zeros_like(gathered))
        
        # Mark selected expert as unavailable for all active samples
        active_batch = batch_indices[active]
        active_valid = valid_indices[active]
        remaining[active_batch, active_valid] = float("-inf")
    
    return total_logprob


# =============================================================================
# Activation-based Controller Trainer
# =============================================================================

class ActivationControllerTrainer:
    """
    Trainer for activation-based controller that uses LLM hidden states directly.
    
    Instead of learning an RNN hidden state, this controller uses the LLM's
    pre-MLP activations directly, making it more directly connected to the LLM's
    representations.
    
    Key differences from RNN-based controller:
    1. No hidden state - uses LLM activations directly
    2. Selection head is initialized from the router weights
    3. Termination and Q heads take both LLM activation and expert set embedding
    4. V head takes only LLM activation
    """
    
    def __init__(
        self,
        config: ControllerTrainerConfig,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        reward_fn: Callable,
        accelerator: Optional[Accelerator] = None,
        ppl_scorer: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.reward_fn = reward_fn
        self.ppl_scorer = ppl_scorer
        
        # Initialize accelerator if not provided
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
        self.accelerator = accelerator
        
        # Set controller_type to "activation" on the model config
        self._set_controller_type_activation()
        
        # Prepare model and dataloader with accelerator
        self.model, self.train_dataloader = accelerator.prepare(
            self.model, self.train_dataloader
        )
        
        # Get references to the model's built-in activation controllers
        self.activation_controllers = self._get_model_activation_controllers()
        
        # Explicitly initialize switch/termination head bias (like original ControllerTrainer)
        self._initialize_termination_bias(config.switch_init_bias)
        
        # Create optimizer for activation controllers
        self.controller_params = self._get_activation_controller_params()
        if accelerator.is_main_process:
            num_params = sum(p.numel() for p in self.controller_params)
            print(f"[INIT] Activation controller parameters: {num_params:,}")
        
        self.optimizer = torch.optim.AdamW(
            self.controller_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            do_sample=True,
            temperature=config.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize wandb (offline mode) - same as ControllerTrainer
        self.wandb_run = None
        if accelerator.is_main_process and config.report_to == "wandb" and HAS_WANDB:
            import os
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
                    "token_decoding": "greedy",
                    "value_coef": config.value_coef,
                    "latency_cost_per_switch": config.latency_cost_per_switch,
                    "switch_init_bias": config.switch_init_bias,
                    "controller_type": "activation",
                },
            )
            print(f"[WANDB] Initialized in offline mode: {self.wandb_run.dir}")
    
    def _get_moe_layers(self):
        """Get all MoE layers from the model."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        if hasattr(policy, 'model'):
            layers = policy.model.layers
        else:
            layers = policy.layers
        
        moe_layers = {}
        for idx, layer in enumerate(layers):
            mlp = layer.mlp
            if hasattr(mlp, 'controller_enabled') and mlp.controller_enabled:
                moe_layers[idx] = mlp
        
        return moe_layers
    
    def _set_controller_type_activation(self):
        """
        Verify controller_type is set to "activation" in the model config.
        
        This should already be set before model loading via the CLI argument.
        We just verify here since the model must be loaded with the correct
        controller_type for the activation controllers to be instantiated.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        ctrl_type = getattr(policy.config, 'controller_type', 'rnn')
        if ctrl_type != "activation":
            raise ValueError(
                f"Model was loaded with controller_type='{ctrl_type}', expected 'activation'. "
                f"Make sure --controller-type activation is passed BEFORE model loading."
            )
        
        if self.accelerator.is_main_process:
            print("[INIT] Verified controller_type='activation' in model config")
    
    def _get_model_activation_controllers(self) -> Dict[int, nn.Module]:
        """
        Get references to the model's built-in activation controllers.
        
        The model creates GptOssActivationController instances when 
        controller_type="activation" in the config.
        """
        moe_layers = self._get_moe_layers()
        controllers = {}
        
        for layer_idx, mlp in moe_layers.items():
            if hasattr(mlp, 'controller') and mlp.controller is not None:
                ctrl = mlp.controller
                # Verify it's an activation controller, not RNN
                ctrl_class = type(ctrl).__name__
                if "Activation" in ctrl_class:
                    controllers[layer_idx] = ctrl
                    if self.accelerator.is_main_process:
                        print(f"  [INIT] Found activation controller for layer {layer_idx}: {ctrl_class}")
                else:
                    raise ValueError(
                        f"Layer {layer_idx} has controller type {ctrl_class}, expected GptOssActivationController. "
                        f"Make sure controller_type='activation' is set in the model config before loading."
                    )
        
        if len(controllers) == 0:
            raise ValueError(
                "No activation controllers found in model. Make sure:\n"
                "1. controller_enabled=True in config\n"
                "2. controller_type='activation' in config\n"
                "The config must be set BEFORE model instantiation."
            )
        
        return controllers
    
    def _get_activation_controller_params(self) -> List[nn.Parameter]:
        """Get all parameters from activation controllers."""
        params = []
        for ctrl in self.activation_controllers.values():
            params.extend(ctrl.parameters())
        return params
    
    def _initialize_termination_bias(self, bias_value: float):
        """
        Explicitly initialize termination head bias for ALL activation controllers.
        
        This matches the original ControllerTrainer._initialize_switch_bias().
        The termination head is: Sequential(Linear, ReLU, Linear)
        We set the bias of the last Linear layer to bias_value.
        We also set the weights of the last Linear layer to be small (0.01x)
        so the output is dominated by the bias initially.
        
        Args:
            bias_value: The bias value (e.g., -3.0 for ~5% switch probability)
        """
        import math
        num_initialized = 0
        
        for layer_idx, ctrl in self.activation_controllers.items():
            with torch.no_grad():
                # termination_head is Sequential: [Linear, ReLU, Linear]
                # Index 2 is the last Linear layer
                last_layer = ctrl.termination_head[2]
                
                # Initialize weights to be small so output ≈ bias initially
                # Use Xavier init scaled by 0.01
                nn.init.xavier_uniform_(last_layer.weight)
                last_layer.weight.mul_(0.01)
                
                # Set bias to switch_init_bias
                last_layer.bias.fill_(bias_value)
                
                num_initialized += 1
        
        if self.accelerator.is_main_process:
            expected_switch_prob = 1.0 / (1.0 + math.exp(-bias_value))
            print(f"[INIT] Initialized termination_head bias for {num_initialized} layers to {bias_value:.2f}")
            print(f"[INIT] Expected initial switch probability: {expected_switch_prob:.4f} ({expected_switch_prob*100:.2f}%)")
    
    @torch.no_grad()
    def generate_rollout(
        self,
        queries: torch.Tensor,
    ) -> ControllerRollout:
        """
        Generate a rollout using the RNN controller (for data collection).
        
        The activation controllers will be trained on the collected data.
        """
        self.model.eval()
        device = queries.device
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] GPU {self.accelerator.process_index}: batch_size={queries.shape[0]}", flush=True)
        
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
        
        # Compute response lengths and termination flags
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = responses.shape[0]
        response_lengths = torch.zeros(batch_size, device=responses.device, dtype=torch.long)
        terminated = torch.zeros(batch_size, device=responses.device, dtype=torch.bool)
        
        for i in range(batch_size):
            resp = responses[i]
            eos_positions = (resp == eos_token_id).nonzero(as_tuple=True)[0]
            pad_positions = (resp == pad_token_id).nonzero(as_tuple=True)[0] if pad_token_id is not None else torch.tensor([], device=resp.device)
            
            if len(eos_positions) > 0:
                response_lengths[i] = eos_positions[0].item() + 1
                terminated[i] = True
            elif len(pad_positions) > 0:
                response_lengths[i] = pad_positions[0].item()
                terminated[i] = False
            else:
                response_lengths[i] = resp.shape[0]
                terminated[i] = False
        
        if self.accelerator.is_main_process:
            num_terminated = terminated.sum().item()
            print(f"  [ROLLOUT] Response lengths: mean={response_lengths.float().mean().item():.1f}")
            print(f"  [ROLLOUT] Terminated (EOS): {num_terminated}/{batch_size}")
        
        # Get recorded controller actions
        recorded_actions = controller_runtime.get("record_actions", {})
        
        # Compute rewards
        rewards, base_rewards, per_token_kl_list = self._compute_rewards(
            queries, responses, recorded_actions, query_len, response_lengths
        )
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] Rewards: mean={rewards.mean().item():.4f}")
        
        # Convert per_token_kl list to padded tensor
        per_token_kl_tensor = None
        if per_token_kl_list is not None:
            max_len = response_lengths.max().item()
            per_token_kl_tensor = torch.zeros(batch_size, int(max_len), dtype=torch.float32)
            for i, kl in enumerate(per_token_kl_list):
                if kl is not None:
                    length = min(len(kl), int(max_len))
                    per_token_kl_tensor[i, :length] = kl[:length]
            per_token_kl_tensor = per_token_kl_tensor.to(queries.device)
        
        return ControllerRollout(
            layer_data=recorded_actions,
            queries=queries,
            responses=responses,
            rewards=rewards,
            base_rewards=base_rewards,
            response_lengths=response_lengths,
            pad_token_id=self.tokenizer.pad_token_id,
            per_token_kl=per_token_kl_tensor,
            terminated=terminated,
        )
    
    def _compute_rewards(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        recorded_actions: Dict[int, Dict[str, torch.Tensor]],
        query_len: int,
        response_lengths: torch.Tensor,
    ):
        """Compute rewards for the rollout.
        
        Matches the original ControllerTrainer._compute_rewards exactly.
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
        
        # Get base rewards from reward function (matches original signature)
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
        
        # Debug: Print model output and reward score (first sample only)
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
    
    def _process_single_layer_activation_controller(
        self,
        layer_idx: int,
        controller: nn.Module,  # GptOssActivationController
        llm_hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        router_logits: torch.Tensor,  # [batch, seq_len, num_experts]
        switches: torch.Tensor,  # [batch, seq_len]
        selected_indices: torch.Tensor,  # [batch, seq_len, k]
        per_token_base_reward: torch.Tensor,  # [batch, seq_len]
        deliberation_cost: float,
        gamma: float,
        valid_mask: torch.Tensor,  # [batch, seq_len]
        valid_end: torch.Tensor,  # [batch, 1]
        terminated: torch.Tensor,  # [batch]
    ) -> Dict[str, Any]:
        """
        Process a single layer with activation-based controller.
        
        BATCHED VERSION: Processes all tokens at once instead of sequentially.
        This is possible because the activation controller has no sequential dependencies.
        """
        device = llm_hidden_states.device
        batch_size, seq_len, hidden_size = llm_hidden_states.shape
        num_experts = router_logits.shape[-1]
        top_k = selected_indices.shape[-1]
        
        # =========================================================================
        # Phase 1: Precompute "current option" at each timestep (no gradients needed)
        # =========================================================================
        # current_option[t] = the expert set active BEFORE potential switch at t
        # - t=0: router top-k
        # - t>0: if switch[t-1], then selected_indices[t-1]; else current_option[t-1]
        with torch.no_grad():
            router_logits_t0 = router_logits[:, 0, :]
            initial_option = torch.topk(router_logits_t0, top_k, dim=-1).indices  # [batch, k]
            
            # Build current_option_per_t: [batch, seq_len, k]
            current_option_per_t = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long, device=device)
            current_option_per_t[:, 0, :] = initial_option
            
            current_option = initial_option
            for t in range(1, seq_len):
                # If switch happened at t-1, update to selected_indices[t-1]
                switch_at_prev = switches[:, t-1].bool()  # [batch]
                current_option = torch.where(
                    switch_at_prev.unsqueeze(-1).expand(-1, top_k),
                    selected_indices[:, t-1, :],
                    current_option,
                )
                current_option_per_t[:, t, :] = current_option
        
        # =========================================================================
        # Phase 2: BATCHED forward pass - compute V, Q_U_old, Q_U_new, logits
        # =========================================================================
        # Flatten [batch, seq_len, ...] to [batch*seq_len, ...] for batched processing
        h_flat = llm_hidden_states.view(batch_size * seq_len, hidden_size)  # [B*T, H]
        current_option_flat = current_option_per_t.view(batch_size * seq_len, top_k)  # [B*T, k]
        new_option_flat = selected_indices.view(batch_size * seq_len, top_k)  # [B*T, k]
        router_logits_flat = router_logits.view(batch_size * seq_len, num_experts)  # [B*T, E]
        
        # Forward through activation controller (batched)
        switch_logits_flat, candidate_logits_flat, V_flat, _ = controller(
            hidden_states=h_flat,
            current_expert_indices=current_option_flat,
        )
        
        # Compute Q_U_old (batched): Q for current option
        Q_U_old_flat = controller.compute_q_option(h_flat, current_option_flat)
        
        # Compute Q_U_new (batched): Q for new option (selected_indices)
        Q_U_new_flat = controller.compute_q_option(h_flat, new_option_flat)
        
        # NO residual connection: selection head is initialized with router weights,
        # so it already outputs router-like logits. Adding router_logits again would
        # double them and cause a mismatch with inference.
        # Just clamp for numerical stability (same as inference)
        candidate_logits_flat = candidate_logits_flat.clamp(-20, 20)
        
        # Apply sampling temperature (MUST match inference exactly)
        # Get temperature from the MLP layer that owns this controller
        moe_layers = self._get_moe_layers()
        if layer_idx in moe_layers:
            sampling_temp = moe_layers[layer_idx].controller_sampling_temperature
            if sampling_temp != 1.0:
                candidate_logits_flat = candidate_logits_flat / sampling_temp
        
        # Reshape back to [batch, seq_len, ...]
        V_values = V_flat.view(batch_size, seq_len)
        Q_U_old_values = Q_U_old_flat.view(batch_size, seq_len)
        Q_U_new_values = Q_U_new_flat.view(batch_size, seq_len)
        switch_logits = switch_logits_flat.clamp(-20, 20).view(batch_size, seq_len)
        candidate_logits_all = candidate_logits_flat.view(batch_size, seq_len, num_experts)
        
        # =========================================================================
        # Phase 2: Per-token rewards (no deliberation cost - faithful to Harb et al.)
        # =========================================================================
        per_token_reward = per_token_base_reward
        
        # =========================================================================
        # Phase 3: Compute TD targets using GAE
        # =========================================================================
        gae_lambda = self.config.gae_lambda
        beta_probs = torch.sigmoid(switch_logits)
        
        gae_V = torch.zeros(batch_size, device=device, dtype=torch.float32)
        gae_Q = torch.zeros(batch_size, device=device, dtype=torch.float32)
        V_advantages = torch.zeros_like(V_values)
        Q_advantages = torch.zeros_like(Q_U_old_values)
        
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            V_t = V_values[:, t].detach()
            Q_t = Q_U_old_values[:, t].detach()
            
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)
                V_next = V_values[:, t+1].detach()
                Q_next = Q_U_old_values[:, t+1].detach()
                beta_next = beta_probs[:, t+1].detach()
                
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated
                
                V_bootstrap = torch.where(
                    next_is_valid,
                    gamma * V_next,
                    torch.where(should_bootstrap_at_boundary, gamma * V_next, torch.zeros_like(V_next))
                )
                delta_V = r_t + V_bootstrap - V_t
                
                soft_bootstrap = gamma * (beta_next * V_next + (1 - beta_next) * Q_next)
                Q_bootstrap = torch.where(
                    next_is_valid,
                    soft_bootstrap,
                    torch.where(should_bootstrap_at_boundary, soft_bootstrap, torch.zeros_like(soft_bootstrap))
                )
                delta_Q = r_t + Q_bootstrap - Q_t
                
                should_continue_gae = next_is_valid | should_bootstrap_at_boundary
                gae_V = torch.where(should_continue_gae, delta_V + gamma * gae_lambda * gae_V, delta_V)
                gae_Q = torch.where(should_continue_gae, delta_Q + gamma * gae_lambda * gae_Q, delta_Q)
            else:
                delta_V = r_t - V_t
                delta_Q = r_t - Q_t
                gae_V = delta_V
                gae_Q = delta_Q
            
            V_advantages[:, t] = gae_V
            Q_advantages[:, t] = gae_Q
        
        V_targets = V_values.detach() + V_advantages
        Q_U_old_targets = Q_U_old_values.detach() + Q_advantages
        
        # Debug logging for layer 0
        if layer_idx == 0 and self.accelerator.is_main_process:
            print(f"  [ACT-LAYER0] V: mean={V_values[valid_mask].mean().item():.4f}, "
                  f"Q_U_old: mean={Q_U_old_values[valid_mask].mean().item():.4f}")
        
        # =========================================================================
        # Phase 4: Compute advantages
        # =========================================================================
        adv_term = Q_U_old_values.detach() - V_values.detach() + deliberation_cost
        A_select = Q_U_new_values.detach() - V_values.detach()
        
        # =========================================================================
        # Phase 5: Compute losses
        # =========================================================================
        # Value loss: MSE
        V_loss = (V_values - V_targets.detach()) ** 2
        Q_loss = (Q_U_old_values - Q_U_old_targets.detach()) ** 2
        
        # Termination loss (direct gradient, no log-prob - Harb et al. 2017)
        # Skip t=0 (no switch decision)
        t_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        t_mask[0] = False
        t_mask = t_mask.unsqueeze(0).expand(batch_size, -1)
        
        beta_probs_clamped = beta_probs.clamp(1e-6, 1 - 1e-6)
        term_loss = torch.where(
            valid_mask & t_mask,
            adv_term * beta_probs_clamped,
            torch.zeros_like(adv_term)
        )
        
        # Selection loss (REINFORCE with log-prob)
        # Normalize A_select
        select_mask = switches.bool() & valid_mask & t_mask
        A_select_used = A_select[select_mask]
        if A_select_used.numel() > 1:
            A_select_mean = A_select_used.mean()
            A_select_std = A_select_used.std().clamp(min=1e-8)
            A_select_norm = (A_select - A_select_mean) / A_select_std
        else:
            A_select_norm = A_select
        
        # Compute log-prob of selected experts (Plackett-Luce)
        selection_log_probs = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        for t in range(seq_len):
            logits_t = candidate_logits_all[:, t, :]
            indices_t = selected_indices[:, t, :]
            log_prob_t = _plackett_luce_logprob_batched(logits_t, indices_t)
            selection_log_probs[:, t] = log_prob_t
        
        select_loss = torch.where(
            select_mask,
            -A_select_norm * selection_log_probs,
            torch.zeros_like(selection_log_probs)
        )
        
        # Aggregate losses
        num_valid = valid_mask[:, 1:].sum()
        num_switch = (switches.bool() & valid_mask & t_mask).sum()
        
        mean_term_loss = term_loss.sum() / max(num_valid, 1)
        mean_select_loss = select_loss.sum() / max(num_switch, 1)
        policy_loss = mean_term_loss + mean_select_loss
        
        # Value loss: average over valid positions
        v_mask = valid_mask.float()
        mean_V_loss = (V_loss * v_mask).sum() / max(v_mask.sum(), 1)
        mean_Q_loss = (Q_loss * v_mask).sum() / max(v_mask.sum(), 1)
        value_loss = mean_V_loss + mean_Q_loss
        
        # Total loss
        total_loss = policy_loss + self.config.value_coef * value_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_V_loss": mean_V_loss.item(),
            "mean_Q_loss": mean_Q_loss.item(),
            "mean_term_loss": mean_term_loss.item(),
            "mean_select_loss": mean_select_loss.item(),
            "switch_rate": (switches.bool() & valid_mask & t_mask).float().sum().item() / max(num_valid.item(), 1),
            "V_mean": V_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "Q_U_old_mean": Q_U_old_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "adv_term_mean": adv_term[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
        }
    
    def _compute_single_rollout_loss(
        self,
        rollout: ControllerRollout,
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute loss for a single rollout (without optimizer step).
        
        Used by train_step_with_accumulation to accumulate gradients across rollouts.
        
        Args:
            rollout: The rollout data
            scale_factor: Factor to scale loss by (1/grad_accum_steps)
            
        Returns:
            Dict with loss (tensor), policy_loss, value_loss, switch_rate, etc.
        """
        device = rollout.queries.device
        batch_size = rollout.batch_size
        query_len = rollout.queries.shape[1]
        response_lengths = rollout.response_lengths
        
        # Get per-token KL for dense rewards
        per_token_kl = rollout.per_token_kl
        if per_token_kl is None:
            raise ValueError("Activation controller requires per_token_kl for Option-Critic training")
        
        # Get full seq_len from first layer's router_logits (query_len + response_len)
        first_layer_idx = next(iter(self.activation_controllers.keys()))
        first_layer_data = rollout.layer_data.get(first_layer_idx)
        if first_layer_data is None:
            raise ValueError(f"No layer data for layer {first_layer_idx}")
        seq_len = first_layer_data["router_logits"].shape[1]
        
        # Create per_token_base_reward for FULL sequence (query + response)
        # Only response tokens get rewards (query tokens get 0)
        per_token_base_reward = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        kl_response_len = per_token_kl.shape[1]
        
        # Place per_token_kl at response positions (starting at query_len)
        for i in range(batch_size):
            resp_start = query_len
            available_space = seq_len - resp_start
            actual_resp_len = min(kl_response_len, int(response_lengths[i].item()), available_space)
            if actual_resp_len > 0:
                per_token_base_reward[i, resp_start:resp_start+actual_resp_len] = -per_token_kl[i, :actual_resp_len]
        
        # Normalize rewards by response length
        response_len_for_norm = response_lengths.float().clamp(min=1.0)
        per_token_base_reward = per_token_base_reward / response_len_for_norm.unsqueeze(1)
        
        if self.accelerator.is_main_process:
            print(f"  [OC-FWD] seq_len={seq_len} (query={query_len}, response={seq_len - query_len})", flush=True)
            print(f"  [OC-FWD] per_token_base_reward: min={per_token_base_reward.min().item():.4f}, max={per_token_base_reward.max().item():.4f}", flush=True)
        
        # Valid mask: positions within [left_padding, query_len + response_length)
        # Compute attention mask for query (1 for real tokens, 0 for padding)
        query_attention_mask = (rollout.queries != rollout.pad_token_id).long()
        real_query_lengths = query_attention_mask.sum(dim=1)  # Number of real query tokens
        left_padding_lengths = query_len - real_query_lengths  # Number of left-padding tokens
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        start = left_padding_lengths.unsqueeze(1)  # [batch, 1]
        end = (query_len + response_lengths).unsqueeze(1)  # [batch, 1]
        valid_mask = (positions >= start) & (positions < end)
        valid_end = end
        
        # Get terminated flags
        terminated = rollout.terminated if rollout.terminated is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Process each layer
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        layer_metrics = {}
        
        gamma = self.config.gamma
        deliberation_cost = self.config.option_critic_deliberation_cost
        
        for layer_idx, ctrl in self.activation_controllers.items():
            layer_data = rollout.layer_data.get(layer_idx)
            if layer_data is None:
                if self.accelerator.is_main_process:
                    print(f"  [WARN] No data for layer {layer_idx}", flush=True)
                continue
            
            # Get layer data
            llm_hidden_states = layer_data.get("llm_hidden_states")
            if llm_hidden_states is None:
                if self.accelerator.is_main_process:
                    print(f"  [WARN] No llm_hidden_states for layer {layer_idx}", flush=True)
                continue
            
            router_logits = layer_data["router_logits"].to(device)
            switches = layer_data["switches"].to(device)
            selected_indices = layer_data["selected_indices"].to(device)
            llm_hidden_states = llm_hidden_states.to(device)
            
            # Ensure matching sequence lengths
            layer_seq_len = router_logits.shape[1]
            if layer_seq_len != seq_len:
                if self.accelerator.is_main_process:
                    print(f"  [WARN] Seq len mismatch: layer {layer_idx} has {layer_seq_len}, expected {seq_len}", flush=True)
                continue
            
            # Process layer
            layer_result = self._process_single_layer_activation_controller(
                layer_idx=layer_idx,
                controller=ctrl,
                llm_hidden_states=llm_hidden_states,
                router_logits=router_logits,
                switches=switches,
                selected_indices=selected_indices,
                per_token_base_reward=per_token_base_reward,
                deliberation_cost=deliberation_cost,
                gamma=gamma,
                valid_mask=valid_mask,
                valid_end=valid_end,
                terminated=terminated,
            )
            
            total_loss = total_loss + layer_result["loss"]
            layer_metrics[layer_idx] = layer_result
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss * scale_factor
        
        # Aggregate metrics
        avg_policy_loss = sum(m["policy_loss"] for m in layer_metrics.values()) / max(len(layer_metrics), 1)
        avg_value_loss = sum(m["value_loss"] for m in layer_metrics.values()) / max(len(layer_metrics), 1)
        avg_switch_rate = sum(m["switch_rate"] for m in layer_metrics.values()) / max(len(layer_metrics), 1)
        
        return {
            "loss": scaled_loss,  # Tensor for backward
            "loss_value": total_loss.item(),  # Unscaled loss for logging
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "switch_rate": avg_switch_rate,
            "layer_metrics": layer_metrics,
        }
    
    def train_step_with_accumulation(
        self,
        batch_queries: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Training step with gradient accumulation (matches original ControllerTrainer).
        
        Args:
            batch_queries: List of [batch, query_len] tensors, one per accumulation step
            
        Returns:
            Dict with metrics
        """
        self.global_step += 1
        step_start = time.time()
        grad_accum_steps = len(batch_queries)
        
        # Set controllers to training mode
        for ctrl in self.activation_controllers.values():
            ctrl.train()
        
        if self.accelerator.is_main_process:
            print(f"  [DEBUG-STEP-START] Step {self.global_step}: Processing {grad_accum_steps} rollouts with Option-Critic", flush=True)
        
        # =====================================================================
        # Phase 1: Generate all rollouts (no gradients)
        # =====================================================================
        rollout_start = time.time()
        rollouts = []
        all_local_rewards = []
        all_base_rewards = []
        all_response_lengths = []
        
        # Reset reward scorer stats for this training step
        if hasattr(self, 'ppl_scorer') and self.ppl_scorer is not None:
            if hasattr(self.ppl_scorer, 'reset_batch_stats'):
                self.ppl_scorer.reset_batch_stats()
        
        for accum_idx, queries in enumerate(batch_queries):
            if self.accelerator.is_main_process:
                print(f"  [ROLLOUT {accum_idx+1}/{grad_accum_steps}] Generating rollout...", flush=True)
            
            with torch.no_grad():
                rollout = self.generate_rollout(queries)
            
            rollouts.append(rollout)
            all_local_rewards.append(rollout.rewards)
            all_base_rewards.append(rollout.base_rewards)
            all_response_lengths.append(rollout.response_lengths)
        
        rollout_time = time.time() - rollout_start
        
        # Concatenate for logging
        local_rewards = torch.cat(all_local_rewards, dim=0)
        local_base_rewards = torch.cat(all_base_rewards, dim=0)
        local_response_lengths = torch.cat(all_response_lengths, dim=0)
        
        if self.accelerator.is_main_process:
            print(f"  [BATCH-ACCUM] Collected {local_rewards.shape[0]} local samples", flush=True)
            print(f"  [BATCH-ACCUM] Rewards: mean={local_rewards.mean().item():.4f}, std={local_rewards.std().item() if local_rewards.numel() > 1 else 0:.4f}", flush=True)
        
        # =====================================================================
        # Phase 2: Gradient accumulation
        # =====================================================================
        update_start = time.time()
        scale_factor = 1.0 / grad_accum_steps
        
        # Accumulators
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_switch_rate = 0.0
        
        # Multiple update epochs on the same batch (like PPO)
        for epoch_idx in range(self.config.num_update_epochs):
            if self.accelerator.is_main_process and epoch_idx == 0:
                print(f"  [UPDATE] Running {self.config.num_update_epochs} update epochs with {grad_accum_steps} accumulation steps...", flush=True)
            
            # Zero gradients at start of each epoch
            self.optimizer.zero_grad()
            
            for accum_idx, rollout in enumerate(rollouts):
                # Compute loss (scaled for accumulation)
                result = self._compute_single_rollout_loss(rollout, scale_factor=scale_factor)
                
                # Check for NaN
                loss_finite = torch.isfinite(result["loss"]).item()
                if self.accelerator.is_main_process and epoch_idx == 0 and accum_idx == 0:
                    print(f"  [UPDATE-OC] policy_loss={result['policy_loss']:.4f}, value_loss={result['value_loss']:.4f}, switch_rate={result['switch_rate']:.4f}, finite={loss_finite}", flush=True)
                
                if not loss_finite:
                    if self.accelerator.is_main_process:
                        print(f"  [WARNING] Skipping backward due to non-finite loss", flush=True)
                    continue
                
                # Backward - accumulate gradients
                result["loss"].backward()
                
                # Accumulate metrics (only on first epoch)
                if epoch_idx == 0:
                    total_loss += result["loss_value"]
                    total_policy_loss += result["policy_loss"]
                    total_value_loss += result["value_loss"]
                    total_switch_rate += result["switch_rate"]
            
            # =========================================================
            # Sync gradients across all GPUs before optimizer step
            # =========================================================
            if torch.distributed.is_initialized():
                for p in self.controller_params:
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            
            # Check gradients before clipping (first epoch only)
            if self.accelerator.is_main_process and epoch_idx == 0:
                grad_norms = []
                nan_count = 0
                for p in self.controller_params:
                    if p.grad is not None:
                        g = p.grad.float()
                        if not torch.isfinite(g).all():
                            nan_count += 1
                        grad_norms.append(g.norm().item())
                
                if grad_norms:
                    print(f"  [UPDATE] grad_norm: mean={sum(grad_norms)/len(grad_norms):.4e}, max={max(grad_norms):.4e}, nan_params={nan_count}/{len(grad_norms)}", flush=True)
            
            # Gradient clipping
            if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.controller_params,
                    self.config.max_grad_norm,
                )
            
            # Optimizer step (after all accumulation steps)
            self.optimizer.step()
            
            # Clamp switch biases to prevent collapse
            min_switch_bias = -10.0
            max_switch_bias = 5.0
            num_clamped = 0
            for ctrl in self.activation_controllers.values():
                # The termination head is a Sequential: [Linear, ReLU, Linear]
                # The bias is in the last Linear layer (index 2)
                bias_param = ctrl.termination_head[2].bias
                old_val = bias_param.data.clone()
                bias_param.data.clamp_(min_switch_bias, max_switch_bias)
                if not torch.allclose(old_val, bias_param.data):
                    num_clamped += 1
            
            if self.accelerator.is_main_process and epoch_idx == 0 and num_clamped > 0:
                print(f"  [CLAMP] Clamped termination_head bias for {num_clamped} layers to [{min_switch_bias}, {max_switch_bias}]", flush=True)
        
        update_time = time.time() - update_start
        step_time = time.time() - step_start
        
        # Aggregate metrics across all GPUs
        device = local_rewards.device
        local_reward_sum = local_rewards.sum()
        local_count = torch.tensor(float(local_rewards.numel()), device=device)
        
        total_reward_sum = self.accelerator.reduce(local_reward_sum, reduction='sum')
        total_count = self.accelerator.reduce(local_count, reduction='sum')
        
        global_reward_mean = (total_reward_sum / total_count).item()
        total_batch_size = int(total_count.item())
        
        # Compute global switch_rate
        num_gpus = self.accelerator.num_processes
        global_switch_rate = total_switch_rate / grad_accum_steps
        
        # Average metrics over accumulation steps
        n_accum = float(grad_accum_steps)
        metrics = {
            "loss": total_loss / n_accum,
            "policy_loss": total_policy_loss / n_accum,
            "value_loss": total_value_loss / n_accum,
            "reward_mean": global_reward_mean,
            "switch_rate": global_switch_rate,
            "batch_size": total_batch_size,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "step_time": step_time,
            "gradient_accumulation_steps": grad_accum_steps,
        }
        
        return metrics
    
    def train(self) -> None:
        """
        Main training loop with gradient accumulation (matches original ControllerTrainer).
        """
        config = self.config
        accelerator = self.accelerator
        grad_accum_steps = config.gradient_accumulation_steps
        
        # Count total optimizer steps (not total batches)
        total_batches = len(self.train_dataloader) * config.num_train_epochs
        total_optimizer_steps = total_batches // grad_accum_steps
        
        if accelerator.is_main_process:
            print("=" * 60)
            print("Starting Activation Controller Training")
            print("=" * 60)
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Per-device batch size: {config.per_device_train_batch_size}")
            print(f"  Gradient accumulation steps: {grad_accum_steps}")
            print(f"  Effective batch size: {config.per_device_train_batch_size * accelerator.num_processes * grad_accum_steps}")
            print(f"  Update epochs per batch: {config.num_update_epochs}")
            print(f"  Response length: {config.response_length}")
            print(f"  Value coefficient: {config.value_coef}")
            print(f"  Total batches: {total_batches}")
            print(f"  Total optimizer steps: {total_optimizer_steps}")
            print("=" * 60)
        
        # Training loop
        start_time = time.time()
        
        # Running metrics for epoch-level logging (same as ControllerTrainer)
        running_metrics = {"loss": 0.0, "reward": 0.0, "switches": 0.0}
        running_count = 0
        
        for epoch in range(config.num_train_epochs):
            self.epoch = epoch
            
            # Reset running metrics at start of epoch
            running_metrics = {"loss": 0.0, "reward": 0.0, "switches": 0.0}
            running_count = 0
            
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
                if isinstance(batch, dict):
                    queries = batch["input_ids"]
                else:
                    queries = batch[0]
                
                queries = queries.to(accelerator.device)
                batch_queries.append(queries)
                
                # Check if we've collected enough batches for an optimizer step
                if len(batch_queries) < grad_accum_steps:
                    continue
                
                # Training step with accumulation
                metrics = self.train_step_with_accumulation(batch_queries)
                
                # Clear the batch buffer
                batch_queries = []
                
                # Accumulate metrics for epoch-level logging
                running_metrics["loss"] += metrics["loss"]
                running_metrics["reward"] += metrics["reward_mean"]
                running_metrics["switches"] += metrics["switch_rate"]
                running_count += 1
                
                # Update progress bar
                if accelerator.is_main_process:
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
                        print(f"\n  [STEP {self.global_step}] "
                              f"loss={metrics['loss']:.4f} "
                              f"policy_loss={metrics['policy_loss']:.4f} "
                              f"value_loss={metrics['value_loss']:.4f} "
                              f"reward={metrics['reward_mean']:.4f} "
                              f"switch_rate={metrics['switch_rate']:.4f} "
                              f"batch={metrics['batch_size']} "
                              f"time={elapsed:.0f}s", flush=True)
                
                # Log to wandb (same as ControllerTrainer)
                if self.wandb_run is not None:
                    log_dict = {
                        "train/loss": metrics["loss"],
                        "train/policy_loss": metrics["policy_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/reward_mean": metrics["reward_mean"],
                        "train/switch_rate": metrics["switch_rate"],
                        "train/batch_size": metrics["batch_size"],
                        "timing/step_time": metrics["step_time"],
                        "timing/rollout_time": metrics["rollout_time"],
                        "timing/update_time": metrics["update_time"],
                        "progress/epoch": epoch,
                        "progress/global_step": self.global_step,
                    }
                    # Add reward-specific metrics if available
                    if self.ppl_scorer is not None:
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
                if self.global_step % config.save_steps == 0:
                    checkpoint_path = f"{config.output_dir}/activation_controller_step_{self.global_step}.pt"
                    self.save_checkpoint(checkpoint_path)
                    # Also save as latest
                    latest_path = f"{config.output_dir}/activation_controller_latest.pt"
                    self.save_checkpoint(latest_path)
            
            if accelerator.is_main_process:
                pbar.close()
                
                # Log epoch-level metrics (same as ControllerTrainer)
                if running_count > 0:
                    avg_loss = running_metrics["loss"] / running_count
                    avg_reward = running_metrics["reward"] / running_count
                    avg_switches = running_metrics["switches"] / running_count
                    
                    print(f"\n[Epoch {epoch+1}] Avg loss: {avg_loss:.4f}, Avg reward: {avg_reward:.4f}, Avg switches: {avg_switches:.4f}")
                    
                    if self.wandb_run is not None:
                        wandb.log({
                            "epoch/avg_loss": avg_loss,
                            "epoch/avg_reward": avg_reward,
                            "epoch/avg_switches": avg_switches,
                        }, step=self.global_step)
            
            # Synchronize all ranks at epoch boundary
            accelerator.wait_for_everyone()
        
        # Final save
        checkpoint_path = f"{config.output_dir}/activation_controller_final.pt"
        self.save_checkpoint(checkpoint_path)
        
        if accelerator.is_main_process:
            total_time = time.time() - start_time
            print("\n" + "="*60)
            print("Training Complete")
            print("="*60)
            print(f"Total time: {total_time:.1f}s")
            print(f"Total optimizer steps: {self.global_step}")
            
            # Finish wandb
            if self.wandb_run is not None:
                wandb.finish()
    
    def save_checkpoint(self, path: str):
        """Save activation controller checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint = {
                "activation_controllers": {
                    layer_idx: ctrl.state_dict() 
                    for layer_idx, ctrl in self.activation_controllers.items()
                },
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.global_step,
                "epoch": self.epoch,
                "config": self.config,
            }
            torch.save(checkpoint, path)
            print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load activation controller checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        for layer_idx, state_dict in checkpoint.get("activation_controllers", {}).items():
            if layer_idx in self.activation_controllers:
                self.activation_controllers[layer_idx].load_state_dict(state_dict)
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"  [WARN] Failed to load optimizer: {e}")
                import traceback
                traceback.print_exc()
        
        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        if self.accelerator.is_main_process:
            print(f"[CHECKPOINT] Loaded from {path}, step={self.global_step}")

