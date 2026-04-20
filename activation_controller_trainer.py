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

import math
import os
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

# Try to import peft for LoRA (intra-option policy update)
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

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
        
        # Joint option mode: create a single shared controller across all layers
        self.joint_option = getattr(config, 'joint_option', False)
        self.joint_controller = None
        
        if self.joint_option:
            if accelerator.is_main_process:
                print(f"[INIT] Joint option mode enabled - creating shared controller across {len(self.activation_controllers)} layers")
            
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssJointOptionController
            
            # Collect router weights/biases from each MoE layer for selection head init
            moe_layers = self._get_moe_layers()
            router_weights = {}
            router_biases = {}
            for layer_idx, mlp in moe_layers.items():
                router_weights[layer_idx] = mlp.router.weight.data.clone()
                if mlp.router.bias is not None:
                    router_biases[layer_idx] = mlp.router.bias.data.clone()
            
            # Get the model config
            unwrapped = self.accelerator.unwrap_model(self.model)
            policy = unwrapped.policy if hasattr(unwrapped, 'policy') else unwrapped
            if hasattr(policy, 'base_model') and hasattr(policy.base_model, 'model'):
                model_config = policy.base_model.model.config
            else:
                model_config = policy.config
            
            # Set joint option config on model config so the controller picks it up
            model_config.joint_set_embed_dim = getattr(config, 'joint_set_embed_dim', 3072)
            model_config.joint_controller_mlp_hidden = getattr(config, 'joint_controller_mlp_hidden', 4096)
            
            self.joint_controller = GptOssJointOptionController(
                config=model_config,
                moe_layer_indices=sorted(self.activation_controllers.keys()),
                router_weights=router_weights,
                router_biases=router_biases,
            )
            
            # Move to same device as model
            device = next(iter(self.activation_controllers.values())).termination_head[0].weight.device
            self.joint_controller = self.joint_controller.to(device)
            
            # Convert to float32
            for param in self.joint_controller.parameters():
                param.data = param.data.float()
            
            if accelerator.is_main_process:
                num_joint_params = sum(p.numel() for p in self.joint_controller.parameters())
                print(f"[INIT] Joint controller parameters: {num_joint_params:,}")
                print(f"[INIT] Joint controller total_experts: {self.joint_controller.total_experts}")
                print(f"[INIT] Joint controller joint_set_embed_dim: {self.joint_controller.joint_set_embed_dim}")
            
            # Initialize termination bias on joint controller
            import math as _math
            with torch.no_grad():
                last_layer = self.joint_controller.termination_head[2]
                nn.init.xavier_uniform_(last_layer.weight)
                last_layer.weight.mul_(0.01)
                last_layer.bias.fill_(config.switch_init_bias)
            if accelerator.is_main_process:
                expected_switch_prob = 1.0 / (1.0 + _math.exp(-config.switch_init_bias))
                print(f"[INIT] Joint controller termination bias: {config.switch_init_bias:.2f} (switch prob: {expected_switch_prob:.4f})")
            
            # Optimizer for joint controller only (per-layer controllers are NOT trained)
            self.controller_params = list(self.joint_controller.parameters())
            if accelerator.is_main_process:
                num_params = sum(p.numel() for p in self.controller_params)
                print(f"[INIT] Joint controller optimizer parameters: {num_params:,}")
            
            self.optimizer = torch.optim.AdamW(
                self.controller_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            # Per-layer mode (original behavior)
            # =========================================================================
            # Convert activation controller parameters to float32 AFTER model loading
            # =========================================================================
            if accelerator.is_main_process:
                sample_param = next(iter(self.activation_controllers.values())).termination_head[0].weight
                print(f"[INIT] Activation controller dtype BEFORE conversion: {sample_param.dtype}")
            
            self._convert_activation_controllers_to_fp32()
            
            if accelerator.is_main_process:
                sample_param = next(iter(self.activation_controllers.values())).termination_head[0].weight
                print(f"[INIT] Activation controller dtype AFTER conversion: {sample_param.dtype}")
            
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
        
        # =====================================================================
        # Intra-option policy update: LoRA on experts + trainable router
        # (Harb et al. 2017, Algorithm 1: intra-option policy gradient)
        # LoRA is applied in train_controller_standalone.py before model loading
        # =====================================================================
        self.llm_optimizer = None
        self.lora_enabled = False
        self.peft_model = None
        if config.intra_option_update:
            if not HAS_PEFT:
                raise ImportError("peft is required for intra_option_update. Install with: pip install peft")
            
            if accelerator.is_main_process:
                print(f"[INTRA-OPTION] Intra-option policy update enabled")
                print(f"[INTRA-OPTION] LLM learning rate: {config.intra_option_lr}")
            
            # LoRA already applied in train_controller_standalone.py
            # Check if model is a PeftModel
            from peft import PeftModel
            unwrapped = self.accelerator.unwrap_model(self.model)
            if isinstance(unwrapped, PeftModel):
                self.peft_model = unwrapped
                self.lora_enabled = True
                if accelerator.is_main_process:
                    print(f"[INTRA-OPTION] Found PeftModel - LoRA is active")
            else:
                if accelerator.is_main_process:
                    print(f"[INTRA-OPTION] Model type: {type(unwrapped)}")
                    # Check if it's wrapped differently
                    if hasattr(unwrapped, 'base_model'):
                        print(f"[INTRA-OPTION] Has base_model - treating as PEFT")
                        self.lora_enabled = True
                    else:
                        print(f"[INTRA-OPTION] WARNING: Model may not have LoRA active")
            
            # Create second optimizer for LLM (LoRA + router) parameters
            llm_params = self._get_llm_trainable_params()
            if llm_params:
                self.llm_optimizer = torch.optim.AdamW(
                    llm_params,
                    lr=config.intra_option_lr,
                    weight_decay=config.weight_decay,
                )
                if accelerator.is_main_process:
                    num_llm_params = sum(p.numel() for p in llm_params)
                    print(f"[INTRA-OPTION] LLM optimizer: {len(llm_params)} param tensors, {num_llm_params:,} total params")
            else:
                if accelerator.is_main_process:
                    print(f"[INTRA-OPTION] WARNING: No LLM trainable parameters found!")
        
        # Generation config - token sampling strategy depends on intra-option update
        # A2OC requires stochastic sampling: the intra-option policy gradient
        # ∇log π(a|s) * (G - Q) is only unbiased when actions are sampled from π
        # But if we're NOT doing intra-option updates, greedy decoding is fine
        # output_logits=True saves raw unprocessed logits at each step to avoid recomputation in KL reward
        if config.intra_option_update:
            self.generation_config = GenerationConfig(
                max_new_tokens=config.response_length,
                do_sample=True,  # Stochastic sampling for valid policy gradient
                temperature=config.temperature,  # Control exploration
                top_p=0.95,  # Nucleus sampling - exclude extremely low probability tokens
                top_k=0,  # Disable top-k filtering (default is 50, which sets non-top-k to -inf)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,  # Keep for backwards compat, but we use logits
                output_logits=True,  # Raw unprocessed logits (no -inf from top_k filtering)
                return_dict_in_generate=True,
            )
            if accelerator.is_main_process:
                print(f"[GEN-CONFIG] Stochastic token sampling enabled (temp={config.temperature}, top_p=0.95) for intra-option policy gradient")
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=config.response_length,
                do_sample=True,
                temperature=config.temperature,
                top_k=0,  # Disable top-k filtering (default is 50, which causes -inf in scores)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,  # Keep for backwards compat
                output_logits=True,  # Raw unprocessed logits (no -inf from top_k filtering)
                return_dict_in_generate=True,
            )
            if accelerator.is_main_process:
                print(f"[GEN-CONFIG] Token sampling enabled (temp={config.temperature}) without intra-option update")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Epsilon annealing state for Plackett-Luce selection
        self._epsilon_start = getattr(config, 'selection_epsilon_start', None)
        self._epsilon_end = getattr(config, 'selection_epsilon_end', 0.05)
        self._epsilon_anneal_steps = getattr(config, 'selection_epsilon_anneal_steps', 200)
        self._epsilon_fixed = getattr(config, 'selection_epsilon', 0.0)
        
        # Epsilon annealing state for Q-based selection
        self._q_epsilon_start = getattr(config, 'q_selection_epsilon_start', None)
        self._q_epsilon_end = getattr(config, 'q_selection_epsilon_end', 0.05)
        self._q_epsilon_anneal_steps = getattr(config, 'q_selection_epsilon_anneal_steps', 200)
        self._q_epsilon_fixed = getattr(config, 'q_selection_epsilon', 0.1)
        
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
                    "token_decoding": "stochastic" if config.intra_option_update else "stochastic",
                    "value_coef": config.value_coef,
                    "latency_cost_per_switch": config.latency_cost_per_switch,
                    "switch_init_bias": config.switch_init_bias,
                    "controller_type": "activation",
                    "intra_option_update": config.intra_option_update,
                    "intra_option_lr": config.intra_option_lr,
                    "intra_option_warmup_steps": config.intra_option_warmup_steps,
                },
            )
            print(f"[WANDB] Initialized in offline mode: {self.wandb_run.dir}")
    
    def _get_moe_layers(self):
        """Get all MoE layers from the model.
        
        Uses the same pattern as ControllerTrainer to handle PEFT-wrapped models.
        Structure for PEFT: peft_model.base_model.model.model.layers
        Structure for standard: model.model.layers
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        # Handle PEFT-wrapped models: peft_model.base_model.model.model.layers
        # (Same pattern as ControllerTrainer._convert_controller_to_fp32, _initialize_switch_bias, etc.)
        layers = None
        if hasattr(policy, 'base_model') and hasattr(policy.base_model, 'model'):
            # PEFT wrapped model
            inner = policy.base_model.model
            if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
                layers = inner.model.layers
        elif hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
            # Standard model
            layers = policy.model.layers
        
        if layers is None:
            raise AttributeError(
                f"Could not find transformer layers in model. "
                f"Policy type: {type(policy)}, has 'model': {hasattr(policy, 'model')}, "
                f"has 'base_model': {hasattr(policy, 'base_model')}"
            )
        
        if self.accelerator.is_main_process:
            print(f"[INIT] Found {len(layers)} transformer layers")
        
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
        
        Uses the same pattern as ControllerTrainer for handling PEFT-wrapped models.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        # Handle PEFT-wrapped models: peft_model.base_model.model.config
        # (Same pattern as ControllerTrainer)
        config = None
        if hasattr(policy, 'base_model') and hasattr(policy.base_model, 'model'):
            # PEFT wrapped model
            inner = policy.base_model.model
            if hasattr(inner, 'config'):
                config = inner.config
        elif hasattr(policy, 'config'):
            # Standard model
            config = policy.config
        
        if config is None:
            raise AttributeError(
                f"Could not find model config. "
                f"Policy type: {type(policy)}, has 'config': {hasattr(policy, 'config')}"
            )
        
        ctrl_type = getattr(config, 'controller_type', 'rnn')
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
    
    def _get_llm_trainable_params(self) -> List[nn.Parameter]:
        """Get all LLM trainable parameters (LoRA + router).
        
        Returns list of parameters for the LLM optimizer.
        Excludes controller parameters (handled by separate optimizer).
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        policy = model
        if hasattr(policy, 'policy'):
            policy = policy.policy
        
        params = []
        
        # Check if policy is a PEFT model (might have been wrapped before passing to trainer)
        from peft import PeftModel
        is_peft = isinstance(policy, PeftModel)
        
        if is_peft:
            # Get all trainable params from PEFT model (LoRA + router)
            # Exclude controller params (they have separate optimizer)
            for name, param in policy.named_parameters():
                if param.requires_grad and "controller" not in name:
                    params.append(param)
            if self.accelerator.is_main_process:
                print(f"[LLM-PARAMS] Found {len(params)} trainable non-controller params from PeftModel")
        else:
            # Fallback: just router parameters
            if hasattr(policy, 'model') and hasattr(policy.model, 'layers'):
                for layer in policy.model.layers:
                    # Check both 'feed_forward' and 'mlp' naming conventions
                    for mlp_name in ['feed_forward', 'mlp']:
                        if hasattr(layer, mlp_name):
                            mlp = getattr(layer, mlp_name)
                            if hasattr(mlp, 'router'):
                                for param in mlp.router.parameters():
                                    if param.requires_grad:
                                        params.append(param)
            if self.accelerator.is_main_process:
                print(f"[LLM-PARAMS] Fallback: Found {len(params)} router params (non-PEFT)")
        
        return params
    
    def _get_current_epsilon(self) -> float:
        """Compute current Plackett-Luce exploration epsilon based on annealing schedule.
        
        If selection_epsilon_start is set, uses linear annealing:
            ε(step) = max(ε_end, ε_start - (ε_start - ε_end) * step / anneal_steps)
        
        Otherwise, returns the fixed selection_epsilon value.
        
        Returns:
            Current epsilon value for Plackett-Luce ε-greedy exploration.
        """
        if self._epsilon_start is not None:
            # Linear annealing from start to end over anneal_steps
            progress = min(1.0, self.global_step / max(1, self._epsilon_anneal_steps))
            current_eps = self._epsilon_start - (self._epsilon_start - self._epsilon_end) * progress
            return max(self._epsilon_end, current_eps)
        else:
            # Fixed epsilon (backward compatibility)
            return self._epsilon_fixed
    
    def _get_current_q_epsilon(self) -> float:
        """Compute current Q-based selection exploration epsilon based on annealing schedule.
        
        If q_selection_epsilon_start is set, uses linear annealing:
            ε(step) = max(ε_end, ε_start - (ε_start - ε_end) * step / anneal_steps)
        
        Otherwise, returns the fixed q_selection_epsilon value.
        
        Returns:
            Current epsilon value for Q-based ε-greedy exploration.
        """
        if self._q_epsilon_start is not None:
            # Linear annealing from start to end over anneal_steps
            progress = min(1.0, self.global_step / max(1, self._q_epsilon_anneal_steps))
            current_eps = self._q_epsilon_start - (self._q_epsilon_start - self._q_epsilon_end) * progress
            return max(self._q_epsilon_end, current_eps)
        else:
            # Fixed epsilon (backward compatibility)
            return self._q_epsilon_fixed
    
    def _convert_activation_controllers_to_fp32(self) -> None:
        """Convert all activation controller parameters to float32 AFTER model loading.
        
        This must be called after from_pretrained() because:
        - from_pretrained(..., torch_dtype=bfloat16) loads ALL weights in bfloat16
        - This overrides any dtype set in __init__
        
        bfloat16 has only 7 bits of mantissa, giving precision of ~0.02 at magnitude 3.0.
        This means small gradient updates (e.g., lr=1e-3 * grad=1e-3 = 1e-6) get rounded to 0.
        """
        num_converted = 0
        for layer_idx, ctrl in self.activation_controllers.items():
            # Convert each parameter to float32
            for param in ctrl.parameters():
                param.data = param.data.float()
            num_converted += 1
        
        if self.accelerator.is_main_process:
            print(f"[FP32] Converted {num_converted} activation controller modules to float32")
    
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
    
    def _set_controller_enabled(self, model, enabled: bool) -> int:
        """Set controller enabled state for all MoE blocks.
        
        Returns number of modules modified.
        """
        count = 0
        for module in model.modules():
            if hasattr(module, 'controller_enabled'):
                module.controller_enabled = enabled
                count += 1
        return count
    
    @torch.no_grad()
    def _generate_with_teacher_mix(
        self,
        queries: torch.Tensor,
        controller_runtime: dict,
        policy,
        max_new_tokens: int,
        temperature: float,
        teacher_mix_alpha: float,
    ) -> tuple:
        """
        Generate tokens using teacher-mixed sampling (MiniLLM-style).
        
        At each step:
        1. Get student logits (controller enabled) with student's KV cache
        2. Get teacher logits (controller disabled) with teacher's KV cache
        3. Mix: p_mixed = (1 - α) * p_student + α * p_teacher
        4. Sample from p_mixed
        5. Compute importance weight: w = p_student / p_mixed
        
        Uses TWO SEPARATE KV CACHES (like MiniLLM) for O(n) complexity:
        - student_past_kv: KV cache for forward passes with controller enabled
        - teacher_past_kv: KV cache for forward passes with controller disabled
        
        References:
        - Paper: https://arxiv.org/pdf/2306.08543 (MiniLLM, Section 2.2)
        - Generation: LMOps/dpkd/transformers/src/transformers/generation/utils.py line 2997
        - Importance weights: LMOps/minillm/minillm/sampler.py lines 112-115
        
        Args:
            queries: [batch, query_len] input token IDs
            controller_runtime: dict for recording controller actions
            policy: the model to use for generation
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            teacher_mix_alpha: α in p_mixed = (1-α)*p_student + α*p_teacher
            
        Returns:
            sequences: [batch, query_len + response_len] generated sequences
            student_logits_tensor: [batch, response_len, vocab] student logits at each step
            importance_weights_tensor: [batch, response_len] importance weights
        """
        device = queries.device
        batch_size = queries.shape[0]
        query_len = queries.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        # Check if model is PEFT-wrapped (for disabling LoRA on teacher)
        try:
            from peft import PeftModel
            is_peft = isinstance(policy, PeftModel)
        except ImportError:
            is_peft = False
        
        # Initialize - start with just the query
        current_ids = queries.clone()  # [batch, current_len]
        
        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Storage for outputs
        student_logits_list = []
        importance_weights_list = []
        
        # TWO SEPARATE KV CACHES - key to O(n) efficiency
        student_past_kv = None  # KV cache for controller-enabled forward passes
        teacher_past_kv = None  # KV cache for controller-disabled forward passes
        
        # CRITICAL: Must maintain controller_states across generation steps!
        # Without this, each token is treated as "first token" and gets fresh top-k experts,
        # meaning NO expert restriction persists across the sequence.
        student_controller_states = None
        
        # Storage to accumulate controller actions incrementally
        accumulated_actions = {}
        
        if self.accelerator.is_main_process:
            print(f"  [TEACHER-MIX] Starting generation with α={teacher_mix_alpha}, temp={temperature}", flush=True)
            print(f"  [TEACHER-MIX] Using dual KV-cache approach (O(n) complexity)", flush=True)
        
        # Joint option state management
        joint_option_state = None
        joint_moe_layers = None
        joint_k_experts = None
        if self.joint_option and self.joint_controller is not None:
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssJointOptionState, _mixed_policy_sample
            joint_option_state = GptOssJointOptionState()
            # Cache MoE layer info BEFORE disabling controllers
            # (_get_moe_layers filters by controller_enabled, so must be called first)
            joint_moe_layers = self._get_moe_layers()
            joint_k_experts = next(iter(joint_moe_layers.values())).controller_allowed_experts
            # Now disable per-layer controllers - we manage routing externally
            self._set_controller_enabled(policy, False)
            if self.accelerator.is_main_process:
                print(f"  [TEACHER-MIX] Joint option mode: per-layer controllers disabled, k_experts={joint_k_experts}", flush=True)
        
        # =====================================================================
        # Joint option mode: sequential prefill for tokens 0..query_len-2.
        # We process all but the LAST query token here. The last query token
        # is left for the generation loop's first iteration, which processes
        # it through both student and teacher to get logits for the first
        # generated token — exactly the same as the non-joint path.
        # =====================================================================
        joint_mode = (self.joint_option and self.joint_controller is not None)
        
        if joint_mode and query_len > 1:
            moe_layer_indices = self.joint_controller.moe_layer_indices
            num_experts = self.joint_controller.num_experts
            k_experts = joint_k_experts
            num_moe_layers = len(moe_layer_indices)
            selection_epsilon = controller_runtime.get("selection_epsilon", self._get_current_epsilon())
            
            # Collect per-token option indices during prefill for accurate replay
            # Each entry is [batch, num_moe_layers, k] — the option used at that prefill token
            # None means vanilla routing (no mask) was used (q_idx=0 before option init)
            prefill_option_list = []
            
            if self.accelerator.is_main_process:
                print(f"  [TEACHER-MIX] Sequential prefill for {query_len - 1} query tokens (last token deferred to gen loop)", flush=True)
            
            for q_idx in range(query_len - 1):
                prefill_token = current_ids[:, q_idx:q_idx+1]  # [batch, 1]
                if student_past_kv is None:
                    prefill_position_ids = None
                else:
                    prefill_position_ids = torch.tensor([[q_idx]], device=device).expand(batch_size, 1)
                prefill_attn_mask = (current_ids[:, :q_idx+1] != pad_token_id).long()
                
                prefill_runtime = {
                    "record_actions": {},
                    "sampling": True,
                    "selection_epsilon": selection_epsilon,
                    "joint_option_mode": True,
                }
                
                if joint_option_state.current_expert_indices_all is not None:
                    joint_masks = {}
                    for pos, layer_idx in enumerate(moe_layer_indices):
                        layer_indices = joint_option_state.current_expert_indices_all[:, pos, :]
                        mask = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
                        mask.scatter_(1, layer_indices, True)
                        joint_masks[layer_idx] = mask
                    prefill_runtime["joint_option_masks"] = joint_masks
                    prefill_option_list.append(joint_option_state.current_expert_indices_all.detach().clone())
                else:
                    # q_idx=0: no option yet, vanilla routing — record None placeholder
                    prefill_option_list.append(None)
                
                prefill_out = policy(
                    input_ids=prefill_token,
                    attention_mask=prefill_attn_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=student_past_kv,
                    use_cache=True,
                    controller_runtime=prefill_runtime,
                    controller_states=student_controller_states,
                )
                student_past_kv = prefill_out.past_key_values
                student_controller_states = getattr(prefill_out, 'controller_states', None)
                
                prefill_actions = prefill_runtime.get("record_actions", {})
                last_layer_hidden = prefill_actions.get("_last_layer_post_mlp_hidden", None)
                last_layer_hidden_t = last_layer_hidden[:, -1, :] if last_layer_hidden is not None else None
                
                if joint_option_state.current_expert_indices_all is None:
                    # q_idx=0: init from router top-k
                    all_indices = []
                    for pos, layer_idx in enumerate(moe_layer_indices):
                        layer_data = prefill_actions.get(layer_idx, {})
                        layer_router_logits = layer_data.get("router_logits", None)
                        if layer_router_logits is not None:
                            top_k_indices = torch.topk(layer_router_logits[:, -1, :], k_experts, dim=-1).indices
                        else:
                            top_k_indices = torch.zeros(batch_size, k_experts, dtype=torch.long, device=device)
                        all_indices.append(top_k_indices)
                    joint_option_state.current_expert_indices_all = torch.stack(all_indices, dim=1)
                    joint_option_state.last_layer_hidden = last_layer_hidden_t
                else:
                    # q_idx > 0: termination/selection decision
                    h_for_decision = joint_option_state.last_layer_hidden
                    current_all = joint_option_state.current_expert_indices_all
                    
                    switch_logits_jt, _, _ = self.joint_controller(h_for_decision, current_all)
                    switch_logits_jt = switch_logits_jt.clamp(-20, 20)
                    switch_probs = torch.sigmoid(switch_logits_jt)
                    bernoulli_p = switch_probs.clamp(min=1e-6, max=1 - 1e-6)
                    rand = torch.rand(switch_probs.shape, device=device, dtype=switch_probs.dtype)
                    switch_decision = rand < bernoulli_p
                    
                    if switch_decision.any():
                        sel_indices_all = []
                        for pos, layer_idx in enumerate(moe_layer_indices):
                            layer_data = prefill_actions.get(layer_idx, {})
                            layer_hidden = layer_data.get("llm_hidden_states", None)
                            if layer_hidden is not None:
                                h_layer = layer_hidden[:, -1, :]
                            else:
                                raise RuntimeError(
                                    f"Joint option prefill: layer {layer_idx} has no recorded llm_hidden_states."
                                )
                            candidate_logits = self.joint_controller.compute_selection_logits(layer_idx, h_layer)
                            candidate_logits = candidate_logits.clamp(-20, 20)
                            selected = _mixed_policy_sample(candidate_logits, k_experts, epsilon=selection_epsilon, generator=None)
                            sel_indices_all.append(selected)
                        sel_indices_all = torch.stack(sel_indices_all, dim=1)
                        
                        new_all = torch.where(
                            switch_decision.unsqueeze(-1).unsqueeze(-1).expand_as(sel_indices_all),
                            sel_indices_all,
                            current_all,
                        )
                        joint_option_state.current_expert_indices_all = new_all
                    
                    joint_option_state.last_layer_hidden = last_layer_hidden_t
                
                # Accumulate per-layer actions from prefill
                for layer_idx, layer_data in prefill_actions.items():
                    if isinstance(layer_idx, str) and layer_idx.startswith("_"):
                        continue
                    if layer_idx not in accumulated_actions:
                        accumulated_actions[layer_idx] = {}
                    for key, value_t in layer_data.items():
                        if key not in accumulated_actions[layer_idx]:
                            accumulated_actions[layer_idx][key] = value_t
                        else:
                            accumulated_actions[layer_idx][key] = torch.cat(
                                [accumulated_actions[layer_idx][key], value_t], dim=1
                            )
                
                if self.accelerator.is_main_process and (q_idx + 1) % 100 == 0:
                    print(f"  [TEACHER-MIX] Prefill {q_idx+1}/{query_len-1} tokens", flush=True)
            
            # Build prefill_indices_all tensor: [batch, query_len-1, num_moe_layers, k]
            # For q_idx=0 (None entry), we need to fill with the router top-k indices
            # that were determined after q_idx=0's forward pass (stored in prefill_option_list[1] or later).
            # Since q_idx=0 used vanilla routing, we use the option that was initialized
            # from router top-k (which is what joint_option_state got set to after q_idx=0).
            # The first non-None entry is the option initialized from router top-k.
            first_valid_option = None
            for opt in prefill_option_list:
                if opt is not None:
                    first_valid_option = opt
                    break
            
            stacked = []
            for opt in prefill_option_list:
                if opt is None:
                    if first_valid_option is not None:
                        stacked.append(first_valid_option)
                    else:
                        stacked.append(torch.zeros(batch_size, num_moe_layers, k_experts, dtype=torch.long, device=device))
                else:
                    stacked.append(opt)
            prefill_indices_all = torch.stack(stacked, dim=1)  # [batch, query_len-1, num_moe_layers, k]
            
            # Store in accumulated_actions for later use in replay
            joint_key = "_joint_option"
            if joint_key not in accumulated_actions:
                accumulated_actions[joint_key] = {}
            accumulated_actions[joint_key]["prefill_indices_all"] = prefill_indices_all
            
            if self.accelerator.is_main_process:
                print(f"  [TEACHER-MIX] Sequential prefill complete. Recorded {prefill_indices_all.shape[1]} prefill option snapshots.", flush=True)
        
        for step in range(max_new_tokens):
            # Determine input for this step
            if student_past_kv is None:
                # First step: process full query (non-joint mode, or joint with query_len==1)
                input_ids = current_ids
                position_ids = None
            else:
                # Process the last token in current_ids
                input_ids = current_ids[:, -1:]
                position_ids = torch.tensor([[current_ids.shape[1] - 1]], device=device).expand(batch_size, 1)
            
            attention_mask = (current_ids != pad_token_id).long()
            
            # =============================================================
            # Step 1: Student forward pass
            # =============================================================
            step_runtime = {
                "record_actions": {},
                "sampling": True,
                "selection_epsilon": controller_runtime.get("selection_epsilon", self._get_current_epsilon()),
            }
            
            if getattr(self.config, 'q_based_selection', False):
                step_runtime["q_based_selection"] = True
                step_runtime["q_selection_steps"] = getattr(self.config, 'q_selection_steps', 10)
                step_runtime["q_selection_lr"] = getattr(self.config, 'q_selection_lr', 1.0)
                step_runtime["q_selection_epsilon"] = controller_runtime.get("q_selection_epsilon", self._get_current_q_epsilon())
                step_runtime["q_selection_debug"] = getattr(self.config, 'q_selection_debug', False)
                step_runtime["q_selection_init_w"] = getattr(self.config, 'q_selection_init_w', 2.0)
            
            # Joint option: set per-layer masks from current joint state
            if joint_mode:
                step_runtime["joint_option_mode"] = True
                moe_layer_indices = self.joint_controller.moe_layer_indices
                num_experts = self.joint_controller.num_experts
                k_experts = joint_k_experts
                
                if joint_option_state.current_expert_indices_all is not None:
                    joint_masks = {}
                    for pos, layer_idx in enumerate(moe_layer_indices):
                        layer_indices = joint_option_state.current_expert_indices_all[:, pos, :]
                        mask = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
                        mask.scatter_(1, layer_indices, True)
                        joint_masks[layer_idx] = mask
                    step_runtime["joint_option_masks"] = joint_masks
            
            student_outputs = policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=student_past_kv,
                use_cache=True,
                controller_runtime=step_runtime,
                controller_states=student_controller_states,
            )
            student_logits = student_outputs.logits[:, -1, :]  # [batch, vocab]
            student_past_kv = student_outputs.past_key_values
            student_controller_states = getattr(student_outputs, 'controller_states', None)
            
            # =============================================================
            # Joint option: termination / selection decision after forward
            # =============================================================
            if joint_mode:
                step_actions = step_runtime.get("record_actions", {})
                selection_epsilon = controller_runtime.get("selection_epsilon", self._get_current_epsilon())
                
                last_layer_hidden = step_actions.get("_last_layer_post_mlp_hidden", None)
                last_layer_hidden_t = last_layer_hidden[:, -1, :] if last_layer_hidden is not None else None
                
                # Capture the option that was EXECUTED in the forward pass (before any switch)
                # and the hidden state used for the decision (h_{t-1})
                executed_option_t = joint_option_state.current_expert_indices_all  # may be None for first token
                decision_hidden_t = joint_option_state.last_layer_hidden  # h_{t-1}, None for first token
                
                if executed_option_t is None:
                    # Very first token: init from router top-k (query_len==1 case)
                    all_indices = []
                    for pos, layer_idx in enumerate(moe_layer_indices):
                        layer_data = step_actions.get(layer_idx, {})
                        layer_router_logits = layer_data.get("router_logits", None)
                        if layer_router_logits is not None:
                            top_k_indices = torch.topk(layer_router_logits[:, -1, :], k_experts, dim=-1).indices
                        else:
                            top_k_indices = torch.zeros(batch_size, k_experts, dtype=torch.long, device=device)
                        all_indices.append(top_k_indices)
                    joint_option_state.current_expert_indices_all = torch.stack(all_indices, dim=1)
                    joint_option_state.last_layer_hidden = last_layer_hidden_t
                    executed_option_t = joint_option_state.current_expert_indices_all
                    
                    switch_decision = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    switch_logprob = torch.zeros(batch_size, dtype=torch.float32, device=device)
                    value = torch.zeros(batch_size, dtype=torch.float32, device=device)
                    q_u_old = torch.zeros(batch_size, dtype=torch.float32, device=device)
                    selected_indices_all = joint_option_state.current_expert_indices_all.clone()
                else:
                    h_for_decision = decision_hidden_t
                    current_all = executed_option_t
                    
                    switch_logits_jt, value, _ = self.joint_controller(h_for_decision, current_all)
                    q_u_old = self.joint_controller.compute_q_option(h_for_decision, current_all)
                    
                    switch_logits_jt = switch_logits_jt.clamp(-20, 20)
                    switch_probs = torch.sigmoid(switch_logits_jt)
                    bernoulli_p = switch_probs.clamp(min=1e-6, max=1 - 1e-6)
                    rand = torch.rand(switch_probs.shape, device=device, dtype=switch_probs.dtype)
                    switch_decision = rand < bernoulli_p
                    
                    switch_logprob = torch.where(
                        switch_decision,
                        torch.log(bernoulli_p),
                        torch.log(1 - bernoulli_p),
                    )
                    
                    selected_indices_all = []
                    for pos, layer_idx in enumerate(moe_layer_indices):
                        layer_data = step_actions.get(layer_idx, {})
                        layer_hidden = layer_data.get("llm_hidden_states", None)
                        if layer_hidden is not None:
                            h_layer = layer_hidden[:, -1, :]
                        else:
                            raise RuntimeError(
                                f"Joint option: layer {layer_idx} has no recorded llm_hidden_states. "
                                f"This should not happen -- forward_joint_option should always record them."
                            )
                        candidate_logits = self.joint_controller.compute_selection_logits(layer_idx, h_layer)
                        candidate_logits = candidate_logits.clamp(-20, 20)
                        selected = _mixed_policy_sample(
                            candidate_logits, k_experts,
                            epsilon=selection_epsilon, generator=None,
                        )
                        selected_indices_all.append(selected)
                    selected_indices_all = torch.stack(selected_indices_all, dim=1)
                    
                    new_all = torch.where(
                        switch_decision.unsqueeze(-1).unsqueeze(-1).expand_as(selected_indices_all),
                        selected_indices_all,
                        current_all,
                    )
                    joint_option_state.current_expert_indices_all = new_all
                    joint_option_state.last_layer_hidden = last_layer_hidden_t
                
                # Record joint option actions
                # executed_indices_all[t]: option used in the forward pass at step t (pre-switch)
                # selected_indices_all[t]: newly selected option (if switch happened)
                # current_indices_all[t]: option after switch decision (= executed at step t+1)
                # decision_hidden[t]: h_{t-1} used for the termination/V/Q decision
                # last_layer_hidden[t]: h_t from this step's forward pass
                joint_record = {
                    "switches": switch_decision.unsqueeze(1),
                    "switch_logprobs": switch_logprob.unsqueeze(1),
                    "values": value.unsqueeze(1),
                    "q_u_old_values": q_u_old.unsqueeze(1),
                    "selected_indices_all": selected_indices_all.unsqueeze(1) if isinstance(selected_indices_all, torch.Tensor) else joint_option_state.current_expert_indices_all.unsqueeze(1),
                    "executed_indices_all": executed_option_t.unsqueeze(1),
                    "current_indices_all": joint_option_state.current_expert_indices_all.unsqueeze(1),
                }
                if last_layer_hidden_t is not None:
                    joint_record["last_layer_hidden"] = last_layer_hidden_t.unsqueeze(1)
                if decision_hidden_t is not None:
                    joint_record["decision_hidden"] = decision_hidden_t.unsqueeze(1)
                else:
                    # First token: no decision was made, use h_t as placeholder
                    joint_record["decision_hidden"] = last_layer_hidden_t.unsqueeze(1) if last_layer_hidden_t is not None else torch.zeros(batch_size, 1, self.joint_controller.hidden_size, device=device)
                
                joint_key = "_joint_option"
                if joint_key not in accumulated_actions:
                    accumulated_actions[joint_key] = {}
                for key, val in joint_record.items():
                    if key not in accumulated_actions[joint_key]:
                        accumulated_actions[joint_key][key] = val
                    else:
                        accumulated_actions[joint_key][key] = torch.cat(
                            [accumulated_actions[joint_key][key], val], dim=1
                        )
                
                # Accumulate per-layer actions
                for layer_idx, layer_data in step_actions.items():
                    if isinstance(layer_idx, str) and layer_idx.startswith("_"):
                        continue
                    if layer_idx not in accumulated_actions:
                        accumulated_actions[layer_idx] = {}
                    for key, value_t in layer_data.items():
                        if key not in accumulated_actions[layer_idx]:
                            accumulated_actions[layer_idx][key] = value_t
                        else:
                            accumulated_actions[layer_idx][key] = torch.cat(
                                [accumulated_actions[layer_idx][key], value_t], dim=1
                            )
            else:
                # Per-layer mode: accumulate actions
                step_actions = step_runtime.get("record_actions", {})
                for layer_idx, layer_data in step_actions.items():
                    if layer_idx not in accumulated_actions:
                        accumulated_actions[layer_idx] = {}
                    for key, value_t in layer_data.items():
                        if key not in accumulated_actions[layer_idx]:
                            accumulated_actions[layer_idx][key] = value_t
                        else:
                            accumulated_actions[layer_idx][key] = torch.cat(
                                [accumulated_actions[layer_idx][key], value_t], dim=1
                            )
            
            # =============================================================
            # Step 2: Teacher forward pass
            # For joint mode, the teacher was not processed during prefill,
            # so on the first gen step we feed the full query (single pass).
            # =============================================================
            if teacher_past_kv is None:
                teacher_input_ids = current_ids[:, :query_len]
                teacher_position_ids = None
                teacher_attn_mask = (current_ids[:, :query_len] != pad_token_id).long()
            else:
                teacher_input_ids = input_ids
                teacher_position_ids = position_ids
                teacher_attn_mask = attention_mask
            
            self._set_controller_enabled(policy, False)
            
            if hasattr(self, 'ppl_scorer') and self.ppl_scorer is not None:
                self.ppl_scorer._swap_router_weights(policy, use_original=True)
            
            if is_peft:
                with policy.disable_adapter():
                    teacher_outputs = policy(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attn_mask,
                        position_ids=teacher_position_ids,
                        past_key_values=teacher_past_kv,
                        use_cache=True,
                    )
            else:
                teacher_outputs = policy(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attn_mask,
                    position_ids=teacher_position_ids,
                    past_key_values=teacher_past_kv,
                    use_cache=True,
                )
            teacher_logits = teacher_outputs.logits[:, -1, :]  # [batch, vocab]
            teacher_past_kv = teacher_outputs.past_key_values
            
            if hasattr(self, 'ppl_scorer') and self.ppl_scorer is not None:
                self.ppl_scorer._swap_router_weights(policy, use_original=False)
            if not joint_mode:
                self._set_controller_enabled(policy, True)
            
            # =========================================================================
            # Step 3: Mix distributions and sample
            # p_mixed = (1 - α) * p_student + α * p_teacher
            # Matches MiniLLM: LMOps/dpkd/transformers/.../generation/utils.py line 2997
            # =========================================================================
            student_logits_scaled = student_logits / temperature
            teacher_logits_scaled = teacher_logits / temperature
            
            student_probs = F.softmax(student_logits_scaled.float(), dim=-1)
            teacher_probs = F.softmax(teacher_logits_scaled.float(), dim=-1)
            
            # Match MiniLLM exactly: (1 - alpha) * student + alpha * teacher
            mixed_probs = (1 - teacher_mix_alpha) * student_probs + teacher_mix_alpha * teacher_probs
            
            # Sample from mixed distribution
            next_token = torch.multinomial(mixed_probs, num_samples=1).squeeze(-1)
            
            # =========================================================================
            # Step 4: Compute importance weights
            # w = p_student / p_mixed
            # Matches MiniLLM: LMOps/minillm/minillm/sampler.py lines 112-115
            # =========================================================================
            student_prob_sampled = student_probs.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
            mixed_prob_sampled = mixed_probs.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
            
            # MiniLLM doesn't clamp importance weights - match exactly
            importance_weight = student_prob_sampled / mixed_prob_sampled.clamp(min=1e-8)
            
            # =========================================================================
            # Step 5: Handle EOS and update sequence
            # =========================================================================
            next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)
            finished = finished | (next_token == eos_token_id)
            
            student_logits_list.append(student_logits)
            importance_weights_list.append(importance_weight)
            
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            
            # Progress logging every 100 tokens
            if self.accelerator.is_main_process and (step + 1) % 100 == 0:
                print(f"  [TEACHER-MIX] Generated {step+1}/{max_new_tokens} tokens, "
                      f"finished: {finished.sum().item()}/{batch_size}", flush=True)
            
            if finished.all():
                if self.accelerator.is_main_process:
                    print(f"  [TEACHER-MIX] All sequences finished at step {step+1}", flush=True)
                break
        
        # Re-enable per-layer controllers if we disabled them for joint option
        if self.joint_option and self.joint_controller is not None:
            self._set_controller_enabled(policy, True)
        
        # Copy accumulated actions to controller_runtime
        controller_runtime["record_actions"] = accumulated_actions
        
        # Stack outputs
        student_logits_tensor = torch.stack(student_logits_list, dim=1)
        importance_weights_tensor = torch.stack(importance_weights_list, dim=1)
        
        responses = current_ids[:, query_len:]
        
        if self.accelerator.is_main_process:
            num_generated = responses.shape[1]
            print(f"  [TEACHER-MIX] Generated {num_generated} tokens total", flush=True)
            print(f"  [TEACHER-MIX] Importance weights: mean={importance_weights_tensor.mean().item():.4f}, "
                  f"std={importance_weights_tensor.std().item():.4f}, "
                  f"min={importance_weights_tensor.min().item():.4f}, "
                  f"max={importance_weights_tensor.max().item():.4f}", flush=True)
            
            if importance_weights_tensor.mean().item() < 0.5 or importance_weights_tensor.mean().item() > 2.0:
                print(f"  [TEACHER-MIX] WARNING: Unusual importance weight mean. "
                      f"Check if student/teacher distributions are very different.", flush=True)
            
            # Validate recorded actions
            # Note: rec_seq_len = current_ids - 1 is expected, because the last generated
            # token is appended to current_ids but never fed back as input, so no controller
            # action is recorded for it.
            if accumulated_actions:
                first_layer = next(iter(accumulated_actions.keys()))
                rec_seq_len = accumulated_actions[first_layer].get("switches", torch.tensor([])).shape[1] if "switches" in accumulated_actions[first_layer] else 0
                expected_len = current_ids.shape[1] - 1
                if rec_seq_len != expected_len:
                    print(f"  [TEACHER-MIX] WARNING: recorded_actions seq_len={rec_seq_len} != expected {expected_len} (current_ids={current_ids.shape[1]})", flush=True)
                else:
                    print(f"  [TEACHER-MIX] Recorded actions validated: {len(accumulated_actions)} layers, seq_len={rec_seq_len}", flush=True)
        
        return current_ids, student_logits_tensor, importance_weights_tensor
    
    @torch.no_grad()
    def generate_rollout(
        self,
        queries: torch.Tensor,
        ground_truth_answers: Optional[List[str]] = None,
    ) -> ControllerRollout:
        """
        Generate a rollout using the controller (for data collection).
        
        Args:
            queries: [batch, query_len] token IDs
            ground_truth_answers: Optional list of ground truth answer strings for correctness checking
        """
        self.model.eval()
        device = queries.device
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] GPU {self.accelerator.process_index}: batch_size={queries.shape[0]}", flush=True)
        
        # Set up controller runtime to record actions
        # Include selection_epsilon for mixed policy exploration (ε-greedy)
        # Uses annealing schedule if configured (high early, low late)
        current_epsilon = self._get_current_epsilon()
        current_q_epsilon = self._get_current_q_epsilon()
        
        if self.accelerator.is_main_process:
            # Log epsilon annealing progress (for whichever is enabled)
            if self._epsilon_start is not None:
                print(f"  [ROLLOUT] PL ε-annealing: ε={current_epsilon:.4f} "
                      f"(step {self.global_step}/{self._epsilon_anneal_steps}, "
                      f"start={self._epsilon_start:.2f}, end={self._epsilon_end:.2f})", flush=True)
            if self._q_epsilon_start is not None:
                print(f"  [ROLLOUT] Q ε-annealing: ε={current_q_epsilon:.4f} "
                      f"(step {self.global_step}/{self._q_epsilon_anneal_steps}, "
                      f"start={self._q_epsilon_start:.2f}, end={self._q_epsilon_end:.2f})", flush=True)
        
        controller_runtime = {
            "sampling": True,
            "record_actions": {},
            "selection_epsilon": current_epsilon,  # ε-greedy mixture for PL exploration
        }
        
        # Add Q-based selection parameters if enabled
        if getattr(self.config, 'q_based_selection', False):
            controller_runtime["q_based_selection"] = True
            controller_runtime["q_selection_steps"] = getattr(self.config, 'q_selection_steps', 10)
            controller_runtime["q_selection_lr"] = getattr(self.config, 'q_selection_lr', 1.0)
            controller_runtime["q_selection_epsilon"] = current_q_epsilon  # Use annealed value
            controller_runtime["q_selection_debug"] = getattr(self.config, 'q_selection_debug', False)
            controller_runtime["q_selection_init_w"] = getattr(self.config, 'q_selection_init_w', 2.0)
        
        # Get the unwrapped model for generation
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            policy = unwrapped.policy
        else:
            policy = unwrapped
        
        # Check if teacher-mixed sampling is enabled
        teacher_mix_alpha = getattr(self.config, 'teacher_mix_alpha', 0.0)
        importance_weights = None
        
        gen_start = time.time()
        
        if self.joint_option and teacher_mix_alpha <= 0:
            # Joint option mode requires teacher-mix generation loop (manual token-by-token)
            # Force teacher_mix_alpha=0 through the same code path
            if self.accelerator.is_main_process:
                print(f"  [ROLLOUT] Joint option mode: using manual generation loop (α=0)", flush=True)
            teacher_mix_alpha = 0.0
        
        if teacher_mix_alpha > 0 or self.joint_option:
            # =========================================================================
            # Teacher-mixed sampling (MiniLLM-style) OR joint option manual generation
            # =========================================================================
            if self.accelerator.is_main_process:
                print(f"  [ROLLOUT] Using teacher-mixed sampling with α={teacher_mix_alpha}", flush=True)
            
            outputs, student_logits, importance_weights = self._generate_with_teacher_mix(
                queries=queries,
                controller_runtime=controller_runtime,
                policy=policy,
                max_new_tokens=self.config.response_length,
                temperature=self.config.temperature,
                teacher_mix_alpha=teacher_mix_alpha,
            )
        else:
            # =========================================================================
            # Standard generation using HuggingFace's generate()
            # =========================================================================
            # Generate responses
            attention_mask = (queries != self.tokenizer.pad_token_id).long()
            
            gen_output = policy.generate(
                input_ids=queries,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                controller_runtime=controller_runtime,
            )
            
            # Extract sequences and logits from generation output
            # With return_dict_in_generate=True, output is GenerateDecoderOnlyOutput
            outputs = gen_output.sequences  # [batch, query_len + response_len]
            
            # IMPORTANT: Use logits (unprocessed) instead of scores (processed)!
            # When do_sample=True, HuggingFace applies top_k=50 by default, which sets
            # non-top-k tokens to -inf in scores. This causes NaN in KL divergence.
            # The logits are raw model output without any processing.
            generation_logits = getattr(gen_output, 'logits', None)  # Unprocessed (preferred)
            generation_scores = gen_output.scores if generation_logits is None else None  # Fallback
            
            # Stack into [batch, response_len, vocab_size] tensor
            if generation_logits is not None:
                student_logits = torch.stack(generation_logits, dim=1)  # [batch, response_len, vocab]
            elif generation_scores is not None:
                student_logits = torch.stack(generation_scores, dim=1)  # [batch, response_len, vocab]
            else:
                student_logits = None
        
        gen_time = time.time() - gen_start
        
        # Extract responses (remove query prefix)
        query_len = queries.shape[1]
        responses = outputs[:, query_len:]
        
        if self.accelerator.is_main_process:
            print(f"  [ROLLOUT] Generated {responses.shape[1]} tokens in {gen_time:.1f}s")
            if student_logits is not None:
                print(f"  [ROLLOUT] Saved student_logits shape: {student_logits.shape} for KL speedup")
        
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
        
        # Compute rewards - pass student_logits to avoid recomputing in KL reward
        rewards, base_rewards, per_token_kl_list, correctness_list = self._compute_rewards(
            queries, responses, recorded_actions, query_len, response_lengths, student_logits,
            ground_truth_answers=ground_truth_answers,
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
        
        # Convert correctness list to tensor
        correctness_tensor = None
        if correctness_list is not None:
            correctness_tensor = torch.tensor(correctness_list, dtype=torch.bool, device=queries.device)
            if self.accelerator.is_main_process:
                num_correct = correctness_tensor.sum().item()
                print(f"  [ROLLOUT] Correctness: {num_correct}/{batch_size} correct", flush=True)
        
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
            student_logits=student_logits,
            correctness=correctness_tensor,
            importance_weights=importance_weights,
        )
    
    def _compute_rewards(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        recorded_actions: Dict[int, Dict[str, torch.Tensor]],
        query_len: int,
        response_lengths: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        ground_truth_answers: Optional[List[str]] = None,
    ):
        """Compute rewards for the rollout.
        
        Args:
            student_logits: [batch, response_len, vocab_size] - pre-computed logits from generation
                           If provided, KL reward computation skips student forward pass.
            ground_truth_answers: Optional list of ground truth answer strings for correctness checking.
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
        # Pass original token IDs and recorded_actions for KL reward to ensure exact alignment
        # (fixes D_gen != D_reward bug by avoiding re-tokenization)
        # Also pass student_logits to skip student forward pass in KL computation (speedup)
        input_ids = torch.cat([queries, responses], dim=1)  # [batch, query_len + response_len]
        base_rewards = self.reward_fn(
            query_texts, response_texts,
            recorded_actions=recorded_actions,
            input_ids=input_ids,
            left_padding_lengths=left_padding_lengths,
            response_lengths=response_lengths,
            query_len=query_len,
            student_logits=student_logits,
            ground_truth_answers=ground_truth_answers,
        )
        
        # Ensure base_rewards is on the same device
        if not isinstance(base_rewards, torch.Tensor):
            base_rewards = torch.tensor(base_rewards, dtype=torch.float32)
        base_rewards = base_rewards.to(device=queries.device, dtype=torch.float32)
        
        # Apply latency penalty normalized by sequence length AND number of layers
        # This penalizes switch RATE (fraction of token-layer pairs that switched)
        # In joint mode there is one shared switch decision per token, so num_layers=1
        if "_joint_option" in recorded_actions:
            num_layers = 1
        else:
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
        
        # Get correctness flags from scorer if available
        correctness = None
        if hasattr(self, 'ppl_scorer') and self.ppl_scorer is not None:
            if hasattr(self.ppl_scorer, 'last_batch_correctness'):
                correctness = self.ppl_scorer.last_batch_correctness
        
        return rewards, base_rewards, per_token_kl, correctness
    
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
        importance_weights: Optional[torch.Tensor] = None,  # [batch, seq_len] for off-policy correction
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
        # Convert hidden states to float32 to ensure controller computation is in float32
        # (LLM hidden states may be bf16, which would cause bf16 outputs even with fp32 params)
        h_flat = llm_hidden_states.view(batch_size * seq_len, hidden_size).float()  # [B*T, H]
        current_option_flat = current_option_per_t.view(batch_size * seq_len, top_k)  # [B*T, k]
        new_option_flat = selected_indices.view(batch_size * seq_len, top_k)  # [B*T, k]
        router_logits_flat = router_logits.view(batch_size * seq_len, num_experts).float()  # [B*T, E]
        
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
        
        # Reshape back to [batch, seq_len, ...] and convert to float32 for numerical stability
        # Even with fp32 controller params, the computation may involve bf16 intermediates
        # from the LLM hidden states. Force float32 for all value-related tensors.
        V_values = V_flat.view(batch_size, seq_len).float()
        Q_U_old_values = Q_U_old_flat.view(batch_size, seq_len).float()
        Q_U_new_values = Q_U_new_flat.view(batch_size, seq_len).float()
        switch_logits = switch_logits_flat.clamp(-20, 20).view(batch_size, seq_len).float()
        candidate_logits_all = candidate_logits_flat.view(batch_size, seq_len, num_experts).float()
        
        # =========================================================================
        # Per-token reward: PURE KL (no deliberation cost)
        # 
        # We use the λ=0 regime from Harb et al. "When Waiting is not an Option":
        # - Critic learns Q-values for the ORIGINAL reward (KL divergence)
        # - Deliberation cost η appears ONLY in termination advantage: (A + η)
        # - This keeps the LM policy gradient optimizing pure KL
        # - Only termination is affected by switching cost
        #
        # See paper Section "Computational Horizon" and Equation (10):
        #   ∂J/∂θ_β = γE[-∂β/∂θ (A_θ + η)]
        # where A_θ is advantage from ORIGINAL reward (not transformed).
        # =========================================================================
        # Ensure reward is float32 for numerical stability
        per_token_reward = per_token_base_reward.float()
        
        # =========================================================================
        # FIX #2: Use Q_exec (executed option's Q) instead of always Q_U_old
        #
        # When switches[t]=True, the reward at t is generated under the NEW option.
        # Training Q_U_old to match returns from a different option is inconsistent.
        # Q_exec = Q_U_new if switch, Q_U_old otherwise
        # =========================================================================
        Q_exec_values = torch.where(switches.bool(), Q_U_new_values, Q_U_old_values)
        
        # =========================================================================
        # Phase 3: Compute TD targets using GAE
        # 
        # IMPORTANT: Use explicit float32 for all GAE tensors to avoid bf16 precision loss.
        # V_values may be in bf16 if model is loaded in bf16, but GAE accumulation needs
        # higher precision to avoid numeric errors in TD targets.
        # =========================================================================
        gae_lambda = self.config.gae_lambda
        beta_probs = torch.sigmoid(switch_logits)
        
        gae_V = torch.zeros(batch_size, device=device, dtype=torch.float32)
        gae_Q_exec = torch.zeros(batch_size, device=device, dtype=torch.float32)
        # Explicitly use float32 for advantages to avoid bf16 precision loss
        V_advantages = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        Q_exec_advantages = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            V_t = V_values[:, t].detach()
            Q_exec_t = Q_exec_values[:, t].detach()  # Use executed option's Q
            
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)
                V_next = V_values[:, t+1].detach()
                beta_next = beta_probs[:, t+1].detach()
                
                # =====================================================================
                # Q bootstrap must use Q of the SAME option at next state
                # Q_U_old_values[t+1] = Q(s_{t+1}, o_current_at_{t+1})
                #                     = Q(s_{t+1}, o_executed_at_t)
                # because the "current" option at t+1 is whatever was executed at t.
                # =====================================================================
                Q_continue_next = Q_U_old_values[:, t+1].detach()  # Q of SAME option at next state
                
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated
                
                V_bootstrap = torch.where(
                    next_is_valid,
                    gamma * V_next,
                    torch.where(should_bootstrap_at_boundary, gamma * V_next, torch.zeros_like(V_next))
                )
                delta_V = r_t + V_bootstrap - V_t
                
                # Q_exec bootstrap: U(s', o) = β·V(s') + (1-β)·Q(s', o_same)
                soft_bootstrap = gamma * (beta_next * V_next + (1 - beta_next) * Q_continue_next)
                Q_bootstrap = torch.where(
                    next_is_valid,
                    soft_bootstrap,
                    torch.where(should_bootstrap_at_boundary, soft_bootstrap, torch.zeros_like(soft_bootstrap))
                )
                delta_Q_exec = r_t + Q_bootstrap - Q_exec_t
                
                should_continue_gae = next_is_valid | should_bootstrap_at_boundary
                gae_V = torch.where(should_continue_gae, delta_V + gamma * gae_lambda * gae_V, delta_V)
                gae_Q_exec = torch.where(should_continue_gae, delta_Q_exec + gamma * gae_lambda * gae_Q_exec, delta_Q_exec)
            else:
                delta_V = r_t - V_t
                delta_Q_exec = r_t - Q_exec_t
                gae_V = delta_V
                gae_Q_exec = delta_Q_exec
            
            V_advantages[:, t] = gae_V
            Q_exec_advantages[:, t] = gae_Q_exec
        
        V_targets = V_values.detach() + V_advantages
        Q_exec_targets = Q_exec_values.detach() + Q_exec_advantages
        
        if self.accelerator.is_main_process and layer_idx == 0:
            valid_V_adv = V_advantages[valid_mask]
            valid_Q_adv = Q_exec_advantages[valid_mask]
            valid_reward = per_token_reward[valid_mask]
            print(f"  [ACT-LAYER0] per_token_reward: min={valid_reward.min().item():.6f}, max={valid_reward.max().item():.6f}, "
                  f"mean={valid_reward.mean().item():.6f}, std={valid_reward.std().item():.6f}", flush=True)
            print(f"  [ACT-LAYER0] V_advantages: min={valid_V_adv.min().item():.6f}, max={valid_V_adv.max().item():.6f}, "
                  f"mean={valid_V_adv.mean().item():.6f}, std={valid_V_adv.std().item():.6f}", flush=True)
            print(f"  [ACT-LAYER0] Q_exec_advantages: min={valid_Q_adv.min().item():.6f}, max={valid_Q_adv.max().item():.6f}, "
                  f"mean={valid_Q_adv.mean().item():.6f}, std={valid_Q_adv.std().item():.6f}", flush=True)
            print(f"  [ACT-LAYER0] V_targets: min={V_targets[valid_mask].min().item():.6f}, "
                  f"max={V_targets[valid_mask].max().item():.6f}, "
                  f"mean={V_targets[valid_mask].mean().item():.6f}", flush=True)
            print(f"  [ACT-LAYER0] Q_exec_targets: min={Q_exec_targets[valid_mask].min().item():.6f}, "
                  f"max={Q_exec_targets[valid_mask].max().item():.6f}, "
                  f"mean={Q_exec_targets[valid_mask].mean().item():.6f}", flush=True)
        
        # =========================================================================
        # Compute intra-option returns (for LLM policy gradient)
        # This is G (discounted return) WITHOUT Q baseline, computed separately
        # from GAE advantages because intra-option update needs raw returns.
        #
        # For truncated (non-terminated) sequences, we bootstrap with V(s_{t+1})
        # at the boundary, matching the GAE computation above.
        # =========================================================================
        intra_option_returns = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        G_accumulator = torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated
                
                V_next = V_values[:, t+1].detach()
                
                G_accumulator = torch.where(
                    next_is_valid,
                    r_t + gamma * G_accumulator,
                    torch.where(
                        should_bootstrap_at_boundary,
                        r_t + gamma * V_next,
                        r_t
                    )
                )
            else:
                # Last position in tensor (t = seq_len - 1).
                # This is typically padding; the real boundary bootstrap
                # is handled above when t+1 crosses valid_end.
                G_accumulator = r_t
            
            intra_option_returns[:, t] = G_accumulator
        
        # Debug logging for layer 0 (basic stats before t_mask is defined)
        if layer_idx == 0 and self.accelerator.is_main_process:
            print(f"  [ACT-LAYER0] Intra-option return G (no Q baseline): mean={intra_option_returns[valid_mask].mean().item():.4f}", flush=True)
            print(f"  [ACT-LAYER0] V: mean={V_values[valid_mask].mean().item():.4f}, "
                  f"Q_exec: mean={Q_exec_values[valid_mask].mean().item():.4f}, "
                  f"Q_U_old (for term): mean={Q_U_old_values[valid_mask].mean().item():.4f}, "
                  f"Q_U_new (for select): mean={Q_U_new_values[valid_mask].mean().item():.4f}", flush=True)
        
        # =========================================================================
        # Phase 4: Compute advantages
        # =========================================================================
        # Define t_mask early (needed for RMS norm and losses)
        # Skip t=0 (no switch decision at first timestep)
        t_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        t_mask[0] = False
        t_mask = t_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Termination advantage: A + η (λ=0 regime from Harb et al.)
        # Q and V are trained on PURE KL reward (no deliberation cost).
        # The +η margin appears ONLY here in the termination gradient.
        # This makes termination more conservative: switch only if Q-V > η.
        adv_term = Q_U_old_values.detach() - V_values.detach() + deliberation_cost
        A_select = Q_U_new_values.detach() - V_values.detach()
        
        # Optionally apply RMS normalization to termination advantages
        # This helps when advantage variance collapses during training
        if getattr(self.config, 'term_adv_rms_norm', False):
            term_mask = valid_mask & t_mask
            adv_term_used = adv_term[term_mask]
            if adv_term_used.numel() > 1:
                # RMS = sqrt(mean(x^2)) - preserves sign and mean, just scales magnitude
                adv_term_rms = torch.sqrt((adv_term_used ** 2).mean()).clamp(min=1e-8)
                adv_term = adv_term / adv_term_rms
                if layer_idx == 0 and self.accelerator.is_main_process:
                    print(f"  [ACT-LAYER0] adv_term RMS norm applied: rms={adv_term_rms.item():.6f}", flush=True)
        
        # =========================================================================
        # Phase 5: Compute losses
        # =========================================================================
        # Value loss: MSE
        # V_loss: trains V to predict state value
        # Q_loss: trains Q_exec (the executed option) to predict option value
        V_loss = (V_values - V_targets.detach()) ** 2
        Q_loss = (Q_exec_values - Q_exec_targets.detach()) ** 2
        
        # Termination loss (direct gradient, no log-prob - Harb et al. 2017)
        # t_mask already defined in Phase 4 (skips t=0)
        # 
        # Apply importance sampling weights for off-policy correction (teacher-mixed rollouts).
        # The policy gradient should be weighted by w_t = p_student / p_mixed to correct for
        # the distribution mismatch. This follows MiniLLM's per-token approximation.
        beta_probs_clamped = beta_probs.clamp(1e-6, 1 - 1e-6)
        term_loss_raw = adv_term * beta_probs_clamped
        
        # Apply importance weights if available
        if importance_weights is not None:
            term_loss_raw = importance_weights * term_loss_raw
        
        term_loss = torch.where(
            valid_mask & t_mask,
            term_loss_raw,
            torch.zeros_like(adv_term)
        )
        
        # Selection policy loss: -A_select * log_prob_option (only when switching, t > 0)
        # Skip if using Q-based selection (no selection policy to train)
        q_based_selection = getattr(self.config, 'q_based_selection', False)
        
        if q_based_selection:
            # Q-based selection: no selection policy loss
            # The "policy" is implicitly argmax Q, trained via TD on Q network
            selection_log_probs = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
            select_loss = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        else:
            # Plackett-Luce selection: compute log prob
            # RMS-normalize A_select (no centering — it's already an advantage)
            # Consistent with termination advantage normalization
            select_mask = switches.bool() & valid_mask & t_mask
            A_select_used = A_select[select_mask]
            if A_select_used.numel() > 1:
                A_select_rms = torch.sqrt((A_select_used ** 2).mean()).clamp(min=1e-8)
                A_select_norm = A_select / A_select_rms
            else:
                A_select_norm = A_select
            
            # Compute log-prob of selected experts (Plackett-Luce)
            selection_log_probs = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
            for t in range(seq_len):
                logits_t = candidate_logits_all[:, t, :]
                indices_t = selected_indices[:, t, :]
                log_prob_t = _plackett_luce_logprob_batched(logits_t, indices_t)
                selection_log_probs[:, t] = log_prob_t
            
            if layer_idx == 0 and self.accelerator.is_main_process:
                valid_lp = selection_log_probs[select_mask]
                valid_a = A_select_norm[select_mask]
                if valid_lp.numel() > 0:
                    print(f"  [PERLAYER-SELECT-DEBUG] layer {layer_idx}: PL_logprob at switch positions: "
                          f"mean={valid_lp.mean().item():.4f}, min={valid_lp.min().item():.4f}, max={valid_lp.max().item():.4f}", flush=True)
                    print(f"  [PERLAYER-SELECT-DEBUG] A_select_norm at switch positions: "
                          f"mean={valid_a.mean().item():.4f}, std={valid_a.std().item():.4f}, "
                          f"num_switch={select_mask.sum().item()}", flush=True)
            
            # Selection loss with importance weighting for off-policy correction
            select_loss_raw = -A_select_norm * selection_log_probs
            
            # Apply importance weights if available
            if importance_weights is not None:
                select_loss_raw = importance_weights * select_loss_raw
            
            select_loss = torch.where(
                select_mask,
                select_loss_raw,
                torch.zeros_like(selection_log_probs)
            )
        
        # Aggregate losses
        # When using importance weights, normalize by sum of weights instead of count
        # This gives: mean = sum(w * loss) / sum(w) for importance-weighted losses
        num_valid = valid_mask[:, 1:].sum()
        num_switch = (switches.bool() & valid_mask & t_mask).sum()
        
        if importance_weights is not None:
            # Termination: normalize by sum of weights for valid positions (excluding t=0)
            term_weight_sum = (importance_weights * (valid_mask & t_mask).float()).sum().clamp(min=1e-8)
            mean_term_loss = term_loss.sum() / term_weight_sum
            
            # Selection: normalize by sum of weights for switch positions
            select_mask = switches.bool() & valid_mask & t_mask
            select_weight_sum = (importance_weights * select_mask.float()).sum().clamp(min=1e-8)
            mean_select_loss = select_loss.sum() / select_weight_sum if num_switch > 0 else select_loss.sum()
        else:
            mean_term_loss = term_loss.sum() / max(num_valid, 1)
            mean_select_loss = select_loss.sum() / max(num_switch, 1)
        
        policy_loss = mean_term_loss + mean_select_loss
        
        # Value loss: sum over valid positions per batch (matches RNN controller)
        # RNN: value_loss = (V_loss + Q_loss).sum(dim=1)  # [batch]
        # Normalization happens later in _compute_single_rollout_loss
        #
        # IMPORTANT: Critics must learn on ALL valid timesteps including t=0.
        # t_mask skips t=0 which is correct for policies (can't switch at t=0),
        # but critics need to learn V(s_0) and Q(s_0, o) for proper TD bootstrap.
        # Without this, V[0] stays random noise, corrupting TD targets at t=1, t=2, etc.
        V_loss_masked = torch.where(valid_mask, V_loss, torch.zeros_like(V_loss))
        Q_loss_masked = torch.where(valid_mask, Q_loss, torch.zeros_like(Q_loss))
        value_loss = (V_loss_masked + Q_loss_masked).sum(dim=1)  # [batch]
        
        # For logging: compute mean values
        v_mask = valid_mask.float()
        mean_V_loss = (V_loss * v_mask).sum() / max(v_mask.sum(), 1)
        mean_Q_loss = (Q_loss * v_mask).sum() / max(v_mask.sum(), 1)
        
        # Compute expert entropy (for logging and optional entropy bonus)
        # Higher entropy = more uniform distribution = more diverse expert selection
        expert_softmax = torch.softmax(candidate_logits_all, dim=-1)  # [batch, seq_len, num_experts]
        expert_log_softmax = torch.log_softmax(candidate_logits_all, dim=-1)  # numerically stable
        entropy_per_position = -(expert_softmax * expert_log_softmax).sum(dim=-1)  # [batch, seq_len]
        # Mean over valid positions only
        valid_entropy_mask = valid_mask & t_mask
        expert_entropy = (entropy_per_position * valid_entropy_mask).sum() / valid_entropy_mask.sum().clamp(min=1)
        
        # NOTE: Entropy bonus and value_coef scaling are applied at the aggregate level
        # (in _compute_single_rollout_loss), matching the RNN controller pattern.
        # Each layer just returns policy_loss (scalar) and value_loss ([batch] tensor).
        
        # Compute termination binariness metrics (same as RNN controller)
        # Binariness = how close switch_probs are to 0 or 1
        with torch.no_grad():
            valid_switch_probs = beta_probs[valid_mask & t_mask]
            if valid_switch_probs.numel() > 0:
                # Mean switch prob
                switch_prob_mean = valid_switch_probs.mean().item()
                # Std of switch probs (lower = more concentrated)
                switch_prob_std = valid_switch_probs.std().item() if valid_switch_probs.numel() > 1 else 0.0
                # Fraction that are "binary" (< 0.1 or > 0.9)
                binary_frac = ((valid_switch_probs < 0.1) | (valid_switch_probs > 0.9)).float().mean().item()
                # Entropy of switch probs: H = -p*log(p) - (1-p)*log(1-p)
                # Lower entropy = more binary
                p = valid_switch_probs.clamp(1e-7, 1 - 1e-7)
                switch_entropy = (-p * p.log() - (1 - p) * (1 - p).log()).mean().item()
                
                # High switch probability metrics
                switch_prob_max = valid_switch_probs.max().item()
                switch_prob_p90 = torch.quantile(valid_switch_probs, 0.90).item() if valid_switch_probs.numel() >= 10 else switch_prob_max
                switch_prob_p95 = torch.quantile(valid_switch_probs, 0.95).item() if valid_switch_probs.numel() >= 20 else switch_prob_max
                frac_gt_0p1 = (valid_switch_probs > 0.1).float().mean().item()
                frac_gt_0p5 = (valid_switch_probs > 0.5).float().mean().item()
            else:
                switch_prob_mean = 0.0
                switch_prob_std = 0.0
                binary_frac = 0.0
                switch_entropy = 0.0
                switch_prob_max = 0.0
                switch_prob_p90 = 0.0
                switch_prob_p95 = 0.0
                frac_gt_0p1 = 0.0
                frac_gt_0p5 = 0.0
        
        # Debug logging for layer 0 (termination metrics)
        if layer_idx == 0 and self.accelerator.is_main_process:
            print(f"  [ACT-LAYER0] switch_prob: mean={switch_prob_mean:.4f}, std={switch_prob_std:.4f}, "
                  f"max={switch_prob_max:.4f}, p90={switch_prob_p90:.4f}", flush=True)
            print(f"  [ACT-LAYER0] switch_binary_frac={binary_frac:.4f}, entropy={switch_entropy:.4f}, "
                  f"frac>0.1={frac_gt_0p1:.4f}, frac>0.5={frac_gt_0p5:.4f}", flush=True)
        
        return {
            # Losses for aggregation (matches RNN controller pattern)
            "policy_loss": policy_loss,  # Tensor (scalar) - already averaged within layer
            "value_loss": value_loss,    # Tensor [batch] - summed over seq, to be normalized later
            "expert_entropy": expert_entropy,  # Tensor (scalar) - for entropy bonus
            # Counts for normalization
            "num_valid_timesteps": num_valid.item(),
            "num_switch_timesteps": num_switch.item(),
            # For logging
            "mean_V_loss": mean_V_loss.item(),
            "mean_Q_loss": mean_Q_loss.item(),
            "mean_term_loss": mean_term_loss.item(),
            "mean_select_loss": mean_select_loss.item(),
            "switch_rate": (switches.bool() & valid_mask & t_mask).float().sum().item() / max(num_valid.item(), 1),
            "V_mean": V_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "Q_exec_mean": Q_exec_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "Q_U_old_mean": Q_U_old_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "Q_U_new_mean": Q_U_new_values[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            "adv_term_mean": adv_term[valid_mask].mean().item() if valid_mask.sum() > 0 else 0,
            # For intra-option policy update (Harb et al. 2017, Algorithm 1)
            "intra_option_advantage": intra_option_returns.detach(),  # [batch, seq_len] - G (raw return)
            "intra_option_q_values": Q_exec_values.detach(),  # [batch, seq_len] - Q(s,o) for optional baseline
            "valid_mask": valid_mask,  # [batch, seq_len]
            # Switch probabilities for TopK regularization (WITH gradients)
            "switch_probs": beta_probs,  # [batch, seq_len] - keep gradients for TopK loss
            "t_mask": t_mask,  # [batch, seq_len] - skip t=0
            # Termination binariness metrics
            "switch_prob_mean": switch_prob_mean,
            "switch_prob_std": switch_prob_std,
            "switch_binary_frac": binary_frac,  # Fraction of switch_probs that are < 0.1 or > 0.9
            "switch_entropy": switch_entropy,  # Lower = more binary
            "switch_prob_max": switch_prob_max,
            "switch_prob_p90": switch_prob_p90,
            "switch_prob_p95": switch_prob_p95,
            "frac_gt_0p1": frac_gt_0p1,  # Fraction with switch_prob > 0.1
            "frac_gt_0p5": frac_gt_0p5,  # Fraction with switch_prob > 0.5
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
        
        # =========================================================================
        # FIX #1: Place reward at PREDICTION position, not TARGET position
        # 
        # For causal LM: the distribution that generates response token j lives at
        # position (query_len - 1 + j), NOT (query_len + j).
        # - Response token 0 (at position query_len) is predicted at position query_len - 1
        # - Response token j (at position query_len + j) is predicted at query_len - 1 + j
        #
        # The reward for an action should be placed where the action was taken.
        # =========================================================================
        for i in range(batch_size):
            reward_start = query_len - 1  # Position where first response token is predicted
            available_space = seq_len - reward_start
            actual_resp_len = min(kl_response_len, int(response_lengths[i].item()), available_space)
            if actual_resp_len > 0:
                per_token_base_reward[i, reward_start:reward_start+actual_resp_len] = -per_token_kl[i, :actual_resp_len]
        
        if self.accelerator.is_main_process:
            print(f"  [OC-FWD] seq_len={seq_len} (query={query_len}, response={seq_len - query_len})", flush=True)
            print(f"  [OC-FWD] per_token_kl shape={per_token_kl.shape}, response_lengths={response_lengths.tolist()}", flush=True)
            print(f"  [OC-FWD] Reward placed at indices [{query_len-1}:{query_len-1}+resp_len] (prediction positions)", flush=True)
            # Count tokens with non-zero rewards
            num_rewarded = (per_token_base_reward != 0).sum().item()
            print(f"  [OC-FWD] Tokens with rewards: {num_rewarded} (expected: batch_size * avg_resp_len)", flush=True)
        
        # =========================================================================
        # Correctness reward bonus: add +alpha uniformly for correct trajectories
        # =========================================================================
        correctness_alpha = self.config.correctness_reward_alpha
        if correctness_alpha > 0 and rollout.correctness is not None:
            num_correct = 0
            for i in range(batch_size):
                if rollout.correctness[i]:
                    reward_start = query_len - 1
                    available_space = seq_len - reward_start
                    actual_resp_len = min(kl_response_len, int(response_lengths[i].item()), available_space)
                    if actual_resp_len > 0:
                        per_token_base_reward[i, reward_start:reward_start+actual_resp_len] += correctness_alpha
                    num_correct += 1
            if self.accelerator.is_main_process:
                print(f"  [OC-FWD] Correctness bonus: alpha={correctness_alpha}, "
                      f"correct={num_correct}/{batch_size}", flush=True)
        
        # =========================================================================
        # Repetition metrics and penalty (distance-based)
        # For each token at position t, find distance d to previous occurrence
        # Penalty = c * λ^d (c should be negative, λ < 1 so nearby repeats penalized more)
        # Always compute metrics for logging, only apply penalty if c != 0
        # =========================================================================
        rep_c = self.config.repetition_penalty_c
        rep_decay = self.config.repetition_penalty_decay
        
        # Compute repetition metrics for each sample in batch
        responses = rollout.responses  # [batch, response_len] - token IDs
        rep_penalty_tensor = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Metrics to track
        batch_rep_rates = []  # repetition rate per sample: 1 - unique/total
        batch_num_repeats = []  # number of repeat tokens per sample
        batch_total_tokens = []  # total tokens per sample
        
        for i in range(batch_size):
            resp_len = int(response_lengths[i].item())
            if resp_len <= 0:
                continue
            
            # Get response tokens for this sample
            resp_tokens = responses[i, :resp_len].cpu().tolist()
            
            # Track last position of each token
            last_pos = {}  # token_id -> last position
            num_repeats = 0
            
            for t, token_id in enumerate(resp_tokens):
                if token_id in last_pos:
                    num_repeats += 1
                    # Compute penalty if enabled
                    if rep_c != 0.0:
                        d = t - last_pos[token_id]
                        penalty = rep_c * (rep_decay ** d)
                        
                        # Place penalty at the prediction position (query_len - 1 + t)
                        pred_pos = query_len - 1 + t
                        if pred_pos < seq_len:
                            rep_penalty_tensor[i, pred_pos] = penalty
                
                # Update last position
                last_pos[token_id] = t
            
            # Compute repetition rate: 1 - unique/total
            unique_count = len(last_pos)
            rep_rate = 1.0 - unique_count / resp_len if resp_len > 0 else 0.0
            batch_rep_rates.append(rep_rate)
            batch_num_repeats.append(num_repeats)
            batch_total_tokens.append(resp_len)
        
        # Store metrics for wandb logging
        if batch_rep_rates:
            self._last_rep_rate_mean = sum(batch_rep_rates) / len(batch_rep_rates)
            self._last_rep_rate_max = max(batch_rep_rates)
            self._last_num_repeats_mean = sum(batch_num_repeats) / len(batch_num_repeats)
            self._last_repeat_frac_mean = sum(batch_num_repeats) / sum(batch_total_tokens) if sum(batch_total_tokens) > 0 else 0.0
        else:
            self._last_rep_rate_mean = 0.0
            self._last_rep_rate_max = 0.0
            self._last_num_repeats_mean = 0.0
            self._last_repeat_frac_mean = 0.0
        
        # Add repetition penalty to base reward BEFORE normalization (matches RNN controller)
        if rep_c != 0.0:
            per_token_base_reward = per_token_base_reward + rep_penalty_tensor
            
            # Compute total repetition penalty per sample (for wandb logging)
            # Sum over sequence dimension, then mean over batch
            rep_penalty_sum_per_sample = rep_penalty_tensor.sum(dim=1)  # [batch]
            self._last_rep_penalty_sum_mean = rep_penalty_sum_per_sample.mean().item()
            
            if self.accelerator.is_main_process:
                nonzero_penalty = rep_penalty_tensor[rep_penalty_tensor != 0]
                if nonzero_penalty.numel() > 0:
                    self._last_rep_penalty_mean = nonzero_penalty.mean().item()
                    print(f"  [OC-FWD] Repetition penalty (c={rep_c}, λ={rep_decay}): "
                          f"num_penalized={nonzero_penalty.numel()}, "
                          f"mean={nonzero_penalty.mean().item():.6f}, "
                          f"min={nonzero_penalty.min().item():.6f}, max={nonzero_penalty.max().item():.6f}", flush=True)
                else:
                    self._last_rep_penalty_mean = 0.0
                    print(f"  [OC-FWD] Repetition penalty (c={rep_c}, λ={rep_decay}): no repeats found", flush=True)
        else:
            self._last_rep_penalty_mean = 0.0
            self._last_rep_penalty_sum_mean = 0.0
        
        # Normalize rewards by a CONSTANT for numerical stability AFTER adding all components
        # (matches RNN controller order: KL reward + repetition penalty, THEN normalize)
        # We use a fixed constant (512) instead of actual response length so that
        # per-token rewards don't depend on how many other tokens are in the sequence.
        # This keeps hyperparameters calibrated while fixing the length bias issue.
        REWARD_NORMALIZATION_CONSTANT = 512.0
        per_token_base_reward = per_token_base_reward / REWARD_NORMALIZATION_CONSTANT
        
        if self.accelerator.is_main_process:
            print(f"  [OC-FWD] per_token_base_reward (normalized by {REWARD_NORMALIZATION_CONSTANT}): min={per_token_base_reward.min().item():.4f}, max={per_token_base_reward.max().item():.4f}", flush=True)
        
        if self.accelerator.is_main_process:
            print(f"  [OC-FWD] Repetition metrics: rep_rate_mean={self._last_rep_rate_mean:.4f}, "
                  f"rep_rate_max={self._last_rep_rate_max:.4f}, "
                  f"num_repeats_mean={self._last_num_repeats_mean:.1f}, "
                  f"repeat_frac={self._last_repeat_frac_mean:.4f}", flush=True)
        
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
        
        # Process each layer (matches RNN controller pattern)
        # Accumulate: policy_loss (scalars summed), value_loss ([batch] tensors summed)
        total_policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_value_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
        total_expert_entropy = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_timesteps = 0
        num_switch_timesteps = 0
        num_layers_with_entropy = 0
        layer_metrics = {}
        
        # For intra-option policy update
        intra_option_advantages = {}  # layer_idx -> [batch, seq_len]
        intra_option_q_values = {}  # layer_idx -> [batch, seq_len] (for optional Q baseline)
        valid_masks_per_layer = {}  # layer_idx -> [batch, seq_len]
        
        gamma = self.config.gamma
        deliberation_cost = self.config.option_critic_deliberation_cost
        
        # =========================================================================
        # Create full-sequence importance weights for off-policy correction
        # rollout.importance_weights: [batch, response_len] from teacher-mixed sampling
        # We expand to [batch, seq_len] with query positions having weight=1.0
        #
        # Alignment: importance_weights[j] corrects for response token j being
        # sampled from the mixed distribution. The reward for response token j is
        # placed at position (query_len - 1 + j) (prediction position), so the
        # importance weight must be placed at the same position.
        # =========================================================================
        full_seq_importance_weights = None
        if rollout.importance_weights is not None:
            iw_response = rollout.importance_weights  # [batch, response_len]
            full_seq_importance_weights = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
            
            iw_response_len = iw_response.shape[1]
            start_idx = query_len - 1
            available_space = seq_len - start_idx
            actual_iw_len = min(iw_response_len, available_space)
            
            if actual_iw_len > 0:
                full_seq_importance_weights[:, start_idx:start_idx + actual_iw_len] = iw_response[:, :actual_iw_len].to(device)
            
            if self.accelerator.is_main_process:
                valid_iw = full_seq_importance_weights[valid_mask]
                print(f"  [OC-FWD] Importance weights for controller: mean={valid_iw.mean().item():.4f}, "
                      f"std={valid_iw.std().item() if valid_iw.numel() > 1 else 0:.4f}", flush=True)
        
        if self.accelerator.is_main_process:
            num_experts = first_layer_data["router_logits"].shape[-1]
            print(f"  [OC-FWD] batch_size={batch_size}, num_layers={len(self.activation_controllers)}, num_experts={num_experts}", flush=True)
            print(f"  [OC-FWD] gamma={gamma}, deliberation_cost={deliberation_cost}", flush=True)
            print(f"  [OC-FWD] valid_mask: {valid_mask.sum().item()} / {valid_mask.numel()} ({100*valid_mask.sum().item()/valid_mask.numel():.1f}%)", flush=True)
        
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
                importance_weights=full_seq_importance_weights,
            )
            
            # Accumulate losses (matches RNN controller)
            total_policy_loss = total_policy_loss + layer_result["policy_loss"]
            total_value_loss = total_value_loss + layer_result["value_loss"]
            num_valid_timesteps += layer_result["num_valid_timesteps"]
            num_switch_timesteps += layer_result["num_switch_timesteps"]
            
            # Accumulate expert entropy (keep gradients for entropy bonus)
            if "expert_entropy" in layer_result:
                total_expert_entropy = total_expert_entropy + layer_result["expert_entropy"]
                num_layers_with_entropy += 1
            
            layer_metrics[layer_idx] = layer_result
            
            # Collect intra-option advantages for LLM update
            if "intra_option_advantage" in layer_result:
                intra_option_advantages[layer_idx] = layer_result["intra_option_advantage"]
            if "intra_option_q_values" in layer_result:
                intra_option_q_values[layer_idx] = layer_result["intra_option_q_values"]
            if "valid_mask" in layer_result:
                valid_masks_per_layer[layer_idx] = layer_result["valid_mask"]
            
            # Collect switch probabilities for TopK regularization
            if "switch_probs" in layer_result and "t_mask" in layer_result:
                if not hasattr(self, '_all_switch_probs'):
                    self._all_switch_probs = []
                sp = layer_result["switch_probs"]  # [batch, seq_len]
                vm = layer_result["valid_mask"]    # [batch, seq_len]
                tm = layer_result["t_mask"]        # [batch, seq_len]
                valid_sp = sp[vm & tm]  # Flatten to 1D, valid positions only
                self._all_switch_probs.append(valid_sp)
        
        # =========================================================================
        # TopK termination regularization loss
        # Loss = λ * (1 - mean(TopK β))²
        # Prevents termination head from collapsing to uniform-low
        # =========================================================================
        term_topk_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        self._last_term_topk_mean = 0.0
        self._last_term_topk_loss = 0.0
        if self.config.term_topk_lambda > 0 and hasattr(self, '_all_switch_probs') and self._all_switch_probs:
            # Concatenate all switch probs from all layers
            all_sp = torch.cat(self._all_switch_probs, dim=0)
            
            # Get top K
            K = min(self.config.term_topk_k, all_sp.numel())
            if K > 0:
                topk_sp, _ = torch.topk(all_sp, K)
                # Loss = λ * mean((1 - β_i)² for i in topK)
                # This pushes EACH of the top K towards 1, not just their average
                term_topk_loss = self.config.term_topk_lambda * ((1.0 - topk_sp) ** 2).mean()
                
                # Store for wandb logging
                self._last_term_topk_mean = topk_sp.mean().item()
                self._last_term_topk_loss = term_topk_loss.item()
                
                if self.accelerator.is_main_process:
                    print(f"  [OC-FWD] TopK term reg: K={K}, mean_topk={topk_sp.mean().item():.4f}, "
                          f"loss={term_topk_loss.item():.6f}", flush=True)
            
            # Clear for next forward pass
            self._all_switch_probs = []
        
        # Add TopK loss to policy loss (matches RNN controller)
        total_policy_loss = total_policy_loss + term_topk_loss
        
        # Normalize policy loss by number of layers (matches RNN controller)
        num_layers = len(layer_metrics)
        total_policy_loss = total_policy_loss / max(num_layers, 1)
        
        # Compute mean expert entropy (as tensor for gradient flow)
        mean_expert_entropy = total_expert_entropy / max(num_layers_with_entropy, 1)
        
        # Value loss: sum across batch, then normalize by num_decisions
        # (matches RNN controller pattern in compute_loss)
        value_loss_sum = total_value_loss.sum()
        mean_value_loss = value_loss_sum / max(num_valid_timesteps, 1)
        
        # Entropy bonus: -entropy_coef * entropy
        entropy_coef = getattr(self.config, 'entropy_coef', 0.0)
        entropy_bonus = -entropy_coef * mean_expert_entropy
        
        # Total loss = policy + value_coef * value + entropy_bonus
        # (matches RNN controller compute_loss)
        total_loss = total_policy_loss + self.config.value_coef * mean_value_loss + entropy_bonus
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss * scale_factor
        
        # Aggregate metrics for logging
        avg_policy_loss = total_policy_loss.item()
        avg_value_loss = mean_value_loss.item()
        avg_switch_rate = sum(m["switch_rate"] for m in layer_metrics.values()) / max(len(layer_metrics), 1)
        
        # Store entropy for wandb logging
        self._last_expert_entropy = mean_expert_entropy.item()
        
        # Aggregate termination binariness metrics
        if num_layers > 0 and "switch_prob_mean" in next(iter(layer_metrics.values())):
            avg_switch_prob_mean = sum(m["switch_prob_mean"] for m in layer_metrics.values()) / num_layers
            avg_switch_prob_std = sum(m["switch_prob_std"] for m in layer_metrics.values()) / num_layers
            avg_switch_binary_frac = sum(m["switch_binary_frac"] for m in layer_metrics.values()) / num_layers
            avg_switch_entropy = sum(m["switch_entropy"] for m in layer_metrics.values()) / num_layers
            max_switch_prob_max = max(m["switch_prob_max"] for m in layer_metrics.values())
            max_switch_prob_p90 = max(m["switch_prob_p90"] for m in layer_metrics.values())
            max_switch_prob_p95 = max(m["switch_prob_p95"] for m in layer_metrics.values())
            avg_frac_gt_0p1 = sum(m["frac_gt_0p1"] for m in layer_metrics.values()) / num_layers
            avg_frac_gt_0p5 = sum(m["frac_gt_0p5"] for m in layer_metrics.values()) / num_layers
        else:
            avg_switch_prob_mean = 0.0
            avg_switch_prob_std = 0.0
            avg_switch_binary_frac = 0.0
            avg_switch_entropy = 0.0
            max_switch_prob_max = 0.0
            max_switch_prob_p90 = 0.0
            max_switch_prob_p95 = 0.0
            avg_frac_gt_0p1 = 0.0
            avg_frac_gt_0p5 = 0.0
        
        # Summary print (matches RNN controller)
        if self.accelerator.is_main_process:
            switch_rate = num_switch_timesteps / max(num_valid_timesteps, 1)
            print(f"  [OC-FWD] num_valid_timesteps={num_valid_timesteps}, num_switch_timesteps={num_switch_timesteps} ({switch_rate:.2%})", flush=True)
            print(f"  [OC-FWD] total_policy_loss={avg_policy_loss:.4f}, mean_value_loss={avg_value_loss:.4f}", flush=True)
            print(f"  [OC-FWD] expert_entropy={self._last_expert_entropy:.4f} (max={math.log(num_experts):.2f})", flush=True)
        
        avg_term_loss = sum(m["mean_term_loss"] for m in layer_metrics.values()) / max(num_layers, 1) if layer_metrics else 0.0
        avg_select_loss = sum(m["mean_select_loss"] for m in layer_metrics.values()) / max(num_layers, 1) if layer_metrics else 0.0
        avg_V_loss_per_layer = sum(m["mean_V_loss"] for m in layer_metrics.values()) / max(num_layers, 1) if layer_metrics else 0.0
        avg_Q_loss_per_layer = sum(m["mean_Q_loss"] for m in layer_metrics.values()) / max(num_layers, 1) if layer_metrics else 0.0
        
        if self.accelerator.is_main_process:
            per_layer_select = [m["mean_select_loss"] for m in layer_metrics.values()]
            per_layer_term = [m["mean_term_loss"] for m in layer_metrics.values()]
            print(f"  [OC-FWD] avg_term_loss={avg_term_loss:.4f}, avg_select_loss={avg_select_loss:.4f}", flush=True)
            print(f"  [OC-FWD] per-layer select_loss (first 6): {[f'{x:.4f}' for x in per_layer_select[:6]]}", flush=True)
            print(f"  [OC-FWD] per-layer select_loss min={min(per_layer_select):.4f}, max={max(per_layer_select):.4f}, "
                  f"std={torch.tensor(per_layer_select).std().item():.4f}", flush=True)
        
        return {
            "loss": scaled_loss,
            "loss_value": total_loss.item(),
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "mean_term_loss": avg_term_loss,
            "mean_select_loss": avg_select_loss,
            "mean_V_loss": avg_V_loss_per_layer,
            "mean_Q_loss": avg_Q_loss_per_layer,
            "switch_rate": avg_switch_rate,
            "layer_metrics": layer_metrics,
            "intra_option_advantages": intra_option_advantages,
            "intra_option_q_values": intra_option_q_values,
            "valid_masks_per_layer": valid_masks_per_layer,
            "switch_prob_mean": avg_switch_prob_mean,
            "switch_prob_std": avg_switch_prob_std,
            "switch_binary_frac": avg_switch_binary_frac,
            "switch_entropy": avg_switch_entropy,
            "switch_prob_max": max_switch_prob_max,
            "switch_prob_p90": max_switch_prob_p90,
            "switch_prob_p95": max_switch_prob_p95,
            "frac_gt_0p1": avg_frac_gt_0p1,
            "frac_gt_0p5": avg_frac_gt_0p5,
        }
    
    def _compute_single_rollout_loss_joint(
        self,
        rollout: ControllerRollout,
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute loss for a single rollout in JOINT OPTION mode.
        
        Key differences from per-layer mode:
        - Single GAE computation using joint V/Q
        - Single termination loss
        - Per-layer selection losses (using per-layer selection heads)
        - Intra-option returns computed once
        """
        device = rollout.queries.device
        batch_size = rollout.batch_size
        query_len = rollout.queries.shape[1]
        response_lengths = rollout.response_lengths
        
        per_token_kl = rollout.per_token_kl
        if per_token_kl is None:
            raise ValueError("Joint option controller requires per_token_kl")
        
        # Get joint option data from recorded actions
        joint_data = rollout.layer_data.get("_joint_option")
        if joint_data is None:
            raise ValueError("No joint option data found in rollout. Was joint_option mode enabled during generation?")
        
        # Get sequence length from per-layer data
        moe_layer_indices = self.joint_controller.moe_layer_indices
        first_layer_idx = moe_layer_indices[0]
        first_layer_data = rollout.layer_data.get(first_layer_idx)
        if first_layer_data is None:
            raise ValueError(f"No layer data for layer {first_layer_idx}")
        seq_len = first_layer_data["router_logits"].shape[1]
        
        # Build per_token_base_reward (same as per-layer mode)
        per_token_base_reward = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        kl_response_len = per_token_kl.shape[1]
        for i in range(batch_size):
            reward_start = query_len - 1
            available_space = seq_len - reward_start
            actual_resp_len = min(kl_response_len, int(response_lengths[i].item()), available_space)
            if actual_resp_len > 0:
                per_token_base_reward[i, reward_start:reward_start+actual_resp_len] = -per_token_kl[i, :actual_resp_len]
        
        # Repetition penalty (same as per-layer mode)
        rep_c = self.config.repetition_penalty_c
        rep_decay = self.config.repetition_penalty_decay
        responses = rollout.responses
        rep_penalty_tensor = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        batch_rep_rates = []
        batch_num_repeats = []
        batch_total_tokens = []
        
        for i in range(batch_size):
            resp_len = int(response_lengths[i].item())
            if resp_len <= 0:
                continue
            resp_tokens = responses[i, :resp_len].cpu().tolist()
            last_pos = {}
            num_repeats = 0
            for t, token_id in enumerate(resp_tokens):
                if token_id in last_pos:
                    num_repeats += 1
                    if rep_c != 0.0:
                        d = t - last_pos[token_id]
                        penalty = rep_c * (rep_decay ** d)
                        pred_pos = query_len - 1 + t
                        if pred_pos < seq_len:
                            rep_penalty_tensor[i, pred_pos] = penalty
                last_pos[token_id] = t
            unique_count = len(last_pos)
            rep_rate = 1.0 - unique_count / resp_len if resp_len > 0 else 0.0
            batch_rep_rates.append(rep_rate)
            batch_num_repeats.append(num_repeats)
            batch_total_tokens.append(resp_len)
        
        if batch_rep_rates:
            self._last_rep_rate_mean = sum(batch_rep_rates) / len(batch_rep_rates)
            self._last_rep_rate_max = max(batch_rep_rates)
            self._last_num_repeats_mean = sum(batch_num_repeats) / len(batch_num_repeats)
            self._last_repeat_frac_mean = sum(batch_num_repeats) / sum(batch_total_tokens) if sum(batch_total_tokens) > 0 else 0.0
        else:
            self._last_rep_rate_mean = 0.0
            self._last_rep_rate_max = 0.0
            self._last_num_repeats_mean = 0.0
            self._last_repeat_frac_mean = 0.0
        
        if rep_c != 0.0:
            per_token_base_reward = per_token_base_reward + rep_penalty_tensor
            rep_penalty_sum_per_sample = rep_penalty_tensor.sum(dim=1)
            self._last_rep_penalty_sum_mean = rep_penalty_sum_per_sample.mean().item()
            if self.accelerator.is_main_process:
                nonzero_penalty = rep_penalty_tensor[rep_penalty_tensor != 0]
                if nonzero_penalty.numel() > 0:
                    self._last_rep_penalty_mean = nonzero_penalty.mean().item()
                    print(f"  [JOINT-OC] Repetition penalty (c={rep_c}, λ={rep_decay}): "
                          f"num_penalized={nonzero_penalty.numel()}, "
                          f"mean={nonzero_penalty.mean().item():.6f}, "
                          f"min={nonzero_penalty.min().item():.6f}, max={nonzero_penalty.max().item():.6f}", flush=True)
                else:
                    self._last_rep_penalty_mean = 0.0
                    print(f"  [JOINT-OC] Repetition penalty (c={rep_c}, λ={rep_decay}): no repeats found", flush=True)
        else:
            self._last_rep_penalty_mean = 0.0
            self._last_rep_penalty_sum_mean = 0.0
        
        if self.accelerator.is_main_process:
            print(f"  [JOINT-OC] Repetition metrics: rep_rate_mean={self._last_rep_rate_mean:.4f}, "
                  f"rep_rate_max={self._last_rep_rate_max:.4f}, "
                  f"num_repeats_mean={self._last_num_repeats_mean:.1f}, "
                  f"repeat_frac={self._last_repeat_frac_mean:.4f}", flush=True)
        
        REWARD_NORMALIZATION_CONSTANT = 512.0
        per_token_base_reward = per_token_base_reward / REWARD_NORMALIZATION_CONSTANT
        
        # Valid mask (covers full seq_len including query positions)
        query_attention_mask = (rollout.queries != rollout.pad_token_id).long()
        real_query_lengths = query_attention_mask.sum(dim=1)
        left_padding_lengths = query_len - real_query_lengths
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        start = left_padding_lengths.unsqueeze(1)
        end = (query_len + response_lengths).unsqueeze(1)
        valid_mask = (positions >= start) & (positions < end)
        valid_end = end
        terminated = rollout.terminated if rollout.terminated is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Importance weights
        full_seq_importance_weights = None
        if rollout.importance_weights is not None:
            iw_response = rollout.importance_weights
            full_seq_importance_weights = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
            iw_response_len = iw_response.shape[1]
            start_idx = query_len - 1
            available_space = seq_len - start_idx
            actual_iw_len = min(iw_response_len, available_space)
            if actual_iw_len > 0:
                full_seq_importance_weights[:, start_idx:start_idx + actual_iw_len] = iw_response[:, :actual_iw_len].to(device)
        
        # Extract joint option tensors (these have joint_seq_len = num generation steps,
        # which is shorter than seq_len = query_len + response_len from per-layer data)
        switches_joint = joint_data["switches"].to(device)  # [batch, joint_seq_len]
        selected_indices_all_joint = joint_data["selected_indices_all"].to(device)  # [batch, joint_seq_len, num_layers, k]
        executed_indices_all_joint = joint_data.get("executed_indices_all")
        if executed_indices_all_joint is not None:
            executed_indices_all_joint = executed_indices_all_joint.to(device)  # [batch, joint_seq_len, num_layers, k]
        else:
            # Backward compat
            executed_indices_all_joint = joint_data["current_indices_all"].to(device)
            if self.accelerator.is_main_process:
                print("  [JOINT-OC] WARNING: executed_indices_all not found, falling back to current_indices_all", flush=True)
        current_indices_all_joint = joint_data["current_indices_all"].to(device)  # [batch, joint_seq_len, num_layers, k]
        
        # decision_hidden[t] = h_{t-1}, the hidden state used for the termination/V/Q decision
        # last_layer_hidden[t] = h_t, the hidden state from step t's forward pass
        decision_hidden_joint = joint_data.get("decision_hidden", None)
        if decision_hidden_joint is not None:
            decision_hidden_joint = decision_hidden_joint.to(device)  # [batch, joint_seq_len, hidden]
        else:
            # Backward compat: fall back to last_layer_hidden (incorrect but avoids crash)
            decision_hidden_joint = joint_data.get("last_layer_hidden", None)
            if decision_hidden_joint is not None:
                decision_hidden_joint = decision_hidden_joint.to(device)
            if self.accelerator.is_main_process:
                print("  [JOINT-OC] WARNING: decision_hidden not found, falling back to last_layer_hidden", flush=True)
        joint_seq_len = switches_joint.shape[1]
        pad_len = seq_len - joint_seq_len
        
        gamma = self.config.gamma
        deliberation_cost = self.config.option_critic_deliberation_cost
        num_layers = len(moe_layer_indices)
        num_experts = self.joint_controller.num_experts
        
        if self.accelerator.is_main_process:
            print(f"  [JOINT-OC] batch_size={batch_size}, seq_len={seq_len}, joint_seq_len={joint_seq_len}, pad_len={pad_len}, num_layers={num_layers}", flush=True)
            print(f"  [JOINT-OC] gamma={gamma}, deliberation_cost={deliberation_cost}", flush=True)
        
        # =========================================================================
        # Batched forward through joint controller (only over joint_seq_len positions)
        # Use decision_hidden (h_{t-1}) and executed_indices_all (pre-switch option)
        # to match the actual (state, option) pair used during the rollout decision
        # =========================================================================
        h_flat = decision_hidden_joint.view(batch_size * joint_seq_len, -1).float()
        executed_all_flat = executed_indices_all_joint.view(batch_size * joint_seq_len, num_layers, -1)
        selected_all_flat = selected_indices_all_joint.view(batch_size * joint_seq_len, num_layers, -1)
        
        # Forward: termination + V using (h_{t-1}, executed_option_t)
        switch_logits_flat, V_flat, _ = self.joint_controller(h_flat, executed_all_flat)
        
        # Q for executed option (pre-switch) and newly selected option
        Q_U_old_flat = self.joint_controller.compute_q_option(h_flat, executed_all_flat)
        Q_U_new_flat = self.joint_controller.compute_q_option(h_flat, selected_all_flat)
        
        # Reshape to [batch, joint_seq_len]
        V_joint = V_flat.view(batch_size, joint_seq_len).float()
        Q_U_old_joint = Q_U_old_flat.view(batch_size, joint_seq_len).float()
        Q_U_new_joint = Q_U_new_flat.view(batch_size, joint_seq_len).float()
        switch_logits_joint = switch_logits_flat.clamp(-20, 20).view(batch_size, joint_seq_len).float()
        
        # Pad to full seq_len (prepend zeros for query positions so indexing aligns with valid_mask)
        if pad_len > 0:
            V_values = F.pad(V_joint, (pad_len, 0), value=0.0)
            Q_U_old_values = F.pad(Q_U_old_joint, (pad_len, 0), value=0.0)
            Q_U_new_values = F.pad(Q_U_new_joint, (pad_len, 0), value=0.0)
            switch_logits = F.pad(switch_logits_joint, (pad_len, 0), value=0.0)
            switches = F.pad(switches_joint.float(), (pad_len, 0), value=0.0).bool()
            selected_indices_all = F.pad(selected_indices_all_joint, (0, 0, 0, 0, pad_len, 0), value=0)
            current_indices_all = F.pad(current_indices_all_joint, (0, 0, 0, 0, pad_len, 0), value=0)
        else:
            V_values = V_joint
            Q_U_old_values = Q_U_old_joint
            Q_U_new_values = Q_U_new_joint
            switch_logits = switch_logits_joint
            switches = switches_joint
            selected_indices_all = selected_indices_all_joint
            current_indices_all = current_indices_all_joint
        
        # In joint mode, switches[t]=1 means "switch AFTER step t's forward pass".
        # The reward r_t was produced under executed_indices_all[t] (old option).
        # So Q_exec[t] is ALWAYS Q of the executed (old) option, regardless of switches[t].
        # This differs from per-layer mode where switches[t]=1 means the new option was
        # used AT step t.
        Q_exec_values = Q_U_old_values
        
        per_token_reward = per_token_base_reward.float()
        
        # Joint valid mask: only positions where joint option data exists (>= pad_len)
        # Query-only positions have zero-padded controller outputs and should not contribute to loss
        joint_valid_mask = valid_mask & (positions >= pad_len)
        
        # =========================================================================
        # GAE computation (single, not per-layer)
        # =========================================================================
        gae_lambda = self.config.gae_lambda
        beta_probs = torch.sigmoid(switch_logits)
        
        gae_V = torch.zeros(batch_size, device=device, dtype=torch.float32)
        gae_Q_exec = torch.zeros(batch_size, device=device, dtype=torch.float32)
        V_advantages = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        Q_exec_advantages = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            V_t = V_values[:, t].detach()
            Q_exec_t = Q_exec_values[:, t].detach()
            
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)
                V_next = V_values[:, t+1].detach()
                beta_next = beta_probs[:, t+1].detach()
                Q_continue_next = Q_U_old_values[:, t+1].detach()
                
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated
                
                V_bootstrap = torch.where(
                    next_is_valid, gamma * V_next,
                    torch.where(should_bootstrap_at_boundary, gamma * V_next, torch.zeros_like(V_next))
                )
                delta_V = r_t + V_bootstrap - V_t
                
                # In joint mode, switches[t]=1 means old option terminated AFTER step t.
                # If terminated: next-state value for Q of old option is just V(s_{t+1})
                #   (the old option is done; a new option was selected)
                # If not terminated: U(s_{t+1}, o) = β_{t+1}*V_{t+1} + (1-β_{t+1})*Q(s_{t+1}, o_same)
                switched_t = switches[:, t].bool() if t < switches.shape[1] else torch.zeros(batch_size, dtype=torch.bool, device=device)
                soft_bootstrap_continue = gamma * (beta_next * V_next + (1 - beta_next) * Q_continue_next)
                soft_bootstrap_term = gamma * V_next
                soft_bootstrap = torch.where(switched_t, soft_bootstrap_term, soft_bootstrap_continue)
                Q_bootstrap = torch.where(
                    next_is_valid, soft_bootstrap,
                    torch.where(should_bootstrap_at_boundary, soft_bootstrap, torch.zeros_like(soft_bootstrap))
                )
                delta_Q_exec = r_t + Q_bootstrap - Q_exec_t
                
                should_continue_gae = next_is_valid | should_bootstrap_at_boundary
                gae_V = torch.where(should_continue_gae, delta_V + gamma * gae_lambda * gae_V, delta_V)
                # Cut Q GAE trace when option terminates (switches[t]=1), because
                # the old option's trajectory ends here and future GAE terms belong
                # to a different option.
                q_should_continue = should_continue_gae & ~switched_t
                gae_Q_exec = torch.where(q_should_continue, delta_Q_exec + gamma * gae_lambda * gae_Q_exec, delta_Q_exec)
            else:
                gae_V = r_t - V_t
                gae_Q_exec = r_t - Q_exec_t
            
            V_advantages[:, t] = gae_V
            Q_exec_advantages[:, t] = gae_Q_exec
        
        V_targets = V_values.detach() + V_advantages
        Q_exec_targets = Q_exec_values.detach() + Q_exec_advantages
        
        if self.accelerator.is_main_process:
            valid_V_adv = V_advantages[joint_valid_mask]
            valid_Q_adv = Q_exec_advantages[joint_valid_mask]
            valid_reward = per_token_reward[joint_valid_mask]
            print(f"  [JOINT-OC] per_token_reward: min={valid_reward.min().item():.6f}, max={valid_reward.max().item():.6f}, "
                  f"mean={valid_reward.mean().item():.6f}, std={valid_reward.std().item():.6f}", flush=True)
            print(f"  [JOINT-OC] V_advantages: min={valid_V_adv.min().item():.6f}, max={valid_V_adv.max().item():.6f}, "
                  f"mean={valid_V_adv.mean().item():.6f}, std={valid_V_adv.std().item():.6f}", flush=True)
            print(f"  [JOINT-OC] Q_exec_advantages: min={valid_Q_adv.min().item():.6f}, max={valid_Q_adv.max().item():.6f}, "
                  f"mean={valid_Q_adv.mean().item():.6f}, std={valid_Q_adv.std().item():.6f}", flush=True)
            print(f"  [JOINT-OC] V_targets: min={V_targets[joint_valid_mask].min().item():.6f}, "
                  f"max={V_targets[joint_valid_mask].max().item():.6f}, "
                  f"mean={V_targets[joint_valid_mask].mean().item():.6f}", flush=True)
            print(f"  [JOINT-OC] Q_exec_targets: min={Q_exec_targets[joint_valid_mask].min().item():.6f}, "
                  f"max={Q_exec_targets[joint_valid_mask].max().item():.6f}, "
                  f"mean={Q_exec_targets[joint_valid_mask].mean().item():.6f}", flush=True)
        
        # Intra-option returns (for LLM policy gradient)
        intra_option_returns = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        G_accumulator = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for t in reversed(range(seq_len)):
            r_t = per_token_reward[:, t]
            if t + 1 < seq_len:
                next_is_valid = (t + 1) < valid_end.squeeze(1)
                at_boundary = ~next_is_valid
                should_bootstrap_at_boundary = at_boundary & ~terminated
                V_next = V_values[:, t+1].detach()
                G_accumulator = torch.where(
                    next_is_valid, r_t + gamma * G_accumulator,
                    torch.where(should_bootstrap_at_boundary, r_t + gamma * V_next, r_t)
                )
            else:
                G_accumulator = r_t
            intra_option_returns[:, t] = G_accumulator
        
        # =========================================================================
        # Losses
        # =========================================================================
        t_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        t_mask[0] = False
        t_mask = t_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Termination advantage
        adv_term = Q_U_old_values.detach() - V_values.detach() + deliberation_cost
        A_select = Q_U_new_values.detach() - V_values.detach()
        
        if getattr(self.config, 'term_adv_rms_norm', False):
            term_mask = joint_valid_mask & t_mask
            adv_term_used = adv_term[term_mask]
            if adv_term_used.numel() > 1:
                adv_term_rms = torch.sqrt((adv_term_used ** 2).mean()).clamp(min=1e-8)
                adv_term = adv_term / adv_term_rms
        
        # Value loss
        V_loss = (V_values - V_targets.detach()) ** 2
        Q_loss = (Q_exec_values - Q_exec_targets.detach()) ** 2
        
        # Termination loss
        beta_probs_clamped = beta_probs.clamp(1e-6, 1 - 1e-6)
        term_loss_raw = adv_term * beta_probs_clamped
        if full_seq_importance_weights is not None:
            term_loss_raw = full_seq_importance_weights * term_loss_raw
        term_loss = torch.where(joint_valid_mask & t_mask, term_loss_raw, torch.zeros_like(adv_term))
        
        num_valid = joint_valid_mask[:, 1:].sum()
        num_switch = (switches.bool() & joint_valid_mask & t_mask).sum()
        
        if full_seq_importance_weights is not None:
            term_weight_sum = (full_seq_importance_weights * (joint_valid_mask & t_mask).float()).sum().clamp(min=1e-8)
            mean_term_loss = term_loss.sum() / term_weight_sum
        else:
            mean_term_loss = term_loss.sum() / max(num_valid, 1)
        
        # Per-layer selection loss (using per-layer selection heads)
        total_select_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        select_mask = switches.bool() & joint_valid_mask & t_mask
        A_select_used = A_select[select_mask]
        if A_select_used.numel() > 1:
            A_select_rms = torch.sqrt((A_select_used ** 2).mean()).clamp(min=1e-8)
            A_select_norm = A_select / A_select_rms
        else:
            A_select_norm = A_select
        
        joint_per_layer_select_losses = []
        for pos, layer_idx in enumerate(moe_layer_indices):
            layer_data = rollout.layer_data.get(layer_idx)
            if layer_data is None:
                continue
            layer_hidden = layer_data.get("llm_hidden_states", None)
            if layer_hidden is None:
                continue
            layer_hidden = layer_hidden.to(device)
            
            # Get per-layer selected indices from joint option
            layer_selected = selected_indices_all[:, :, pos, :]  # [batch, seq_len, k]
            
            # Compute selection logits for this layer
            h_layer_flat = layer_hidden.view(batch_size * seq_len, -1).float()
            candidate_logits_flat = self.joint_controller.compute_selection_logits(layer_idx, h_layer_flat)
            candidate_logits_flat = candidate_logits_flat.clamp(-20, 20)
            candidate_logits_layer = candidate_logits_flat.view(batch_size, seq_len, num_experts)
            
            # Compute PL log-prob
            selection_log_probs = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
            for t in range(seq_len):
                logits_t = candidate_logits_layer[:, t, :]
                indices_t = layer_selected[:, t, :]
                log_prob_t = _plackett_luce_logprob_batched(logits_t, indices_t)
                selection_log_probs[:, t] = log_prob_t
            
            if self.accelerator.is_main_process and pos == 0:
                valid_lp = selection_log_probs[select_mask]
                valid_a = A_select_norm[select_mask]
                print(f"  [JOINT-SELECT-DEBUG] layer {layer_idx}: PL_logprob at switch positions: "
                      f"mean={valid_lp.mean().item():.4f}, min={valid_lp.min().item():.4f}, max={valid_lp.max().item():.4f}", flush=True)
                print(f"  [JOINT-SELECT-DEBUG] A_select_norm at switch positions: "
                      f"mean={valid_a.mean().item():.4f}, std={valid_a.std().item():.4f}, "
                      f"num_switch={select_mask.sum().item()}", flush=True)
            
            select_loss_raw = -A_select_norm * selection_log_probs
            if full_seq_importance_weights is not None:
                select_loss_raw = full_seq_importance_weights * select_loss_raw
            select_loss = torch.where(select_mask, select_loss_raw, torch.zeros_like(selection_log_probs))
            
            if full_seq_importance_weights is not None:
                select_weight_sum = (full_seq_importance_weights * select_mask.float()).sum().clamp(min=1e-8)
                mean_select_loss = select_loss.sum() / select_weight_sum if num_switch > 0 else select_loss.sum()
            else:
                mean_select_loss = select_loss.sum() / max(num_switch, 1)
            
            joint_per_layer_select_losses.append(mean_select_loss.item())
            total_select_loss = total_select_loss + mean_select_loss
        
        # Normalize selection loss by number of layers (matches per-layer controller)
        total_select_loss = total_select_loss / max(num_layers, 1)
        
        if self.accelerator.is_main_process and joint_per_layer_select_losses:
            import numpy as np
            arr = np.array(joint_per_layer_select_losses)
            print(f"  [JOINT-SELECT-DEBUG] per-layer select_loss (first 6): {[f'{x:.4f}' for x in arr[:6]]}", flush=True)
            print(f"  [JOINT-SELECT-DEBUG] per-layer select_loss: min={arr.min():.4f}, max={arr.max():.4f}, "
                  f"mean={arr.mean():.4f}, std={arr.std():.4f}", flush=True)
            print(f"  [JOINT-SELECT-DEBUG] total_select_loss (after /num_layers)={total_select_loss.item():.4f}", flush=True)
        
        policy_loss = mean_term_loss + total_select_loss
        
        V_loss_masked = torch.where(joint_valid_mask, V_loss, torch.zeros_like(V_loss))
        Q_loss_masked = torch.where(joint_valid_mask, Q_loss, torch.zeros_like(Q_loss))
        value_loss = (V_loss_masked + Q_loss_masked).sum(dim=1)
        
        jv_mask = joint_valid_mask.float()
        mean_V_loss = (V_loss * jv_mask).sum() / max(jv_mask.sum(), 1)
        mean_Q_loss = (Q_loss * jv_mask).sum() / max(jv_mask.sum(), 1)
        
        num_valid_timesteps = num_valid.item()
        value_loss_sum = value_loss.sum()
        mean_value_loss = value_loss_sum / max(num_valid_timesteps, 1)
        
        # Expert entropy (averaged across layers, matching per-layer mode)
        total_expert_entropy = torch.tensor(0.0, device=device, requires_grad=True)
        num_layers_with_entropy = 0
        for pos, layer_idx in enumerate(moe_layer_indices):
            layer_data = rollout.layer_data.get(layer_idx)
            if layer_data is None:
                continue
            layer_router_logits = layer_data.get("router_logits")
            if layer_router_logits is None:
                continue
            layer_router_logits = layer_router_logits.to(device)
            expert_softmax = torch.softmax(layer_router_logits, dim=-1)
            expert_log_softmax = torch.log_softmax(layer_router_logits, dim=-1)
            entropy_per_pos = -(expert_softmax * expert_log_softmax).sum(dim=-1)
            valid_entropy_mask = joint_valid_mask & t_mask
            layer_entropy = (entropy_per_pos * valid_entropy_mask).sum() / valid_entropy_mask.sum().clamp(min=1)
            total_expert_entropy = total_expert_entropy + layer_entropy
            num_layers_with_entropy += 1
        
        mean_expert_entropy = total_expert_entropy / max(num_layers_with_entropy, 1)
        self._last_expert_entropy = mean_expert_entropy.item()
        
        entropy_coef = getattr(self.config, 'entropy_coef', 0.0)
        entropy_bonus = -entropy_coef * mean_expert_entropy
        
        total_loss = policy_loss + self.config.value_coef * mean_value_loss + entropy_bonus
        scaled_loss = total_loss * scale_factor
        
        # Compute switch rate
        switch_rate = (switches.bool() & joint_valid_mask & t_mask).float().sum().item() / max(num_valid_timesteps, 1)
        
        # Termination binariness metrics (matching per-layer mode exactly)
        with torch.no_grad():
            valid_switch_probs = beta_probs[joint_valid_mask & t_mask]
            if valid_switch_probs.numel() > 0:
                switch_prob_mean = valid_switch_probs.mean().item()
                switch_prob_std = valid_switch_probs.std().item() if valid_switch_probs.numel() > 1 else 0.0
                binary_frac = ((valid_switch_probs < 0.1) | (valid_switch_probs > 0.9)).float().mean().item()
                p = valid_switch_probs.clamp(1e-7, 1 - 1e-7)
                switch_entropy = (-p * p.log() - (1 - p) * (1 - p).log()).mean().item()
                switch_prob_max = valid_switch_probs.max().item()
                switch_prob_p90 = torch.quantile(valid_switch_probs, 0.90).item() if valid_switch_probs.numel() >= 10 else switch_prob_max
                switch_prob_p95 = torch.quantile(valid_switch_probs, 0.95).item() if valid_switch_probs.numel() >= 20 else switch_prob_max
                frac_gt_0p1 = (valid_switch_probs > 0.1).float().mean().item()
                frac_gt_0p5 = (valid_switch_probs > 0.5).float().mean().item()
            else:
                switch_prob_mean = 0.0
                switch_prob_std = 0.0
                binary_frac = 0.0
                switch_entropy = 0.0
                switch_prob_max = 0.0
                switch_prob_p90 = 0.0
                switch_prob_p95 = 0.0
                frac_gt_0p1 = 0.0
                frac_gt_0p5 = 0.0
        
        if self.accelerator.is_main_process:
            print(f"  [JOINT-OC] policy_loss={policy_loss.item():.4f} (term={mean_term_loss.item():.4f}, "
                  f"select={total_select_loss.item():.4f}), value_loss={mean_value_loss.item():.4f}, "
                  f"switch_rate={switch_rate:.4f}", flush=True)
            print(f"  [JOINT-OC] V_mean={V_values[joint_valid_mask].mean().item():.4f}, "
                  f"Q_exec_mean={Q_exec_values[joint_valid_mask].mean().item():.4f}, "
                  f"expert_entropy={self._last_expert_entropy:.4f}", flush=True)
            print(f"  [JOINT-OC] switch_prob: mean={switch_prob_mean:.4f}, std={switch_prob_std:.4f}, "
                  f"max={switch_prob_max:.4f}, p90={switch_prob_p90:.4f}, p95={switch_prob_p95:.4f}", flush=True)
            print(f"  [JOINT-OC] binary_frac={binary_frac:.4f}, entropy={switch_entropy:.4f}, "
                  f"frac>0.1={frac_gt_0p1:.4f}, frac>0.5={frac_gt_0p5:.4f}", flush=True)
        
        # For intra-option update: single set of returns (not per-layer)
        intra_option_advantages = {"_joint": intra_option_returns.detach()}
        intra_option_q_values = {"_joint": Q_exec_values.detach()}
        valid_masks_per_layer = {"_joint": joint_valid_mask}
        
        return {
            "loss": scaled_loss,
            "loss_value": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": mean_value_loss.item(),
            "mean_term_loss": mean_term_loss.item(),
            "mean_select_loss": total_select_loss.item(),
            "mean_V_loss": mean_V_loss.item(),
            "mean_Q_loss": mean_Q_loss.item(),
            "switch_rate": switch_rate,
            "layer_metrics": {},
            "intra_option_advantages": intra_option_advantages,
            "intra_option_q_values": intra_option_q_values,
            "valid_masks_per_layer": valid_masks_per_layer,
            "switch_prob_mean": switch_prob_mean,
            "switch_prob_std": switch_prob_std,
            "switch_binary_frac": binary_frac,
            "switch_entropy": switch_entropy,
            "switch_prob_max": switch_prob_max,
            "switch_prob_p90": switch_prob_p90,
            "switch_prob_p95": switch_prob_p95,
            "frac_gt_0p1": frac_gt_0p1,
            "frac_gt_0p5": frac_gt_0p5,
        }
    
    # =========================================================================
    # Intra-Option Policy Update (Harb et al. 2017, Algorithm 1)
    # =========================================================================
    
    def _compute_intra_option_loss(
        self,
        rollout: ControllerRollout,
        intra_option_advantages: Dict[int, torch.Tensor],  # layer_idx -> [batch, seq_len]
        valid_masks: Dict[int, torch.Tensor],  # layer_idx -> [batch, seq_len]
        intra_option_q_values: Optional[Dict[int, torch.Tensor]] = None,  # layer_idx -> [batch, seq_len]
    ) -> torch.Tensor:
        """
        Compute intra-option policy gradient loss (Harb et al. 2017, Algorithm 1).
        
        This does a forward pass through the LLM with the recorded expert selections
        (via replay_actions), computes token log probabilities, and applies the
        intra-option advantage.
        
        When intra_option_q_baseline is enabled, uses G - Q(s,o) as the advantage
        (matching A2OC Algorithm 1). Otherwise uses raw G (on-policy distillation style).
        
        Loss: -sum(A_t * log P(token_t | context)) / num_valid_tokens
        
        Args:
            rollout: ControllerRollout with recorded actions
            intra_option_advantages: Per-layer discounted returns G [batch, seq_len]
            valid_masks: Per-layer validity masks
            intra_option_q_values: Per-layer Q(s,o) values for optional baseline subtraction
            
        Returns:
            Intra-option policy loss (scalar tensor with gradients)
        """
        device = rollout.queries.device
        
        # Concatenate queries and responses for the forward pass
        input_ids = torch.cat([rollout.queries, rollout.responses], dim=1)
        batch_size, total_len = input_ids.shape
        query_len = rollout.queries.shape[1]
        response_len = rollout.responses.shape[1]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != rollout.pad_token_id).long()
        
        # Build replay configuration from rollout.layer_data
        is_joint_mode = "_joint_option" in rollout.layer_data
        
        if is_joint_mode:
            # Joint mode: build per-token joint_option_masks for replay
            # Use executed_indices_all (the option actually used in each forward pass)
            # NOT current_indices_all (which is the post-switch option)
            joint_data = rollout.layer_data["_joint_option"]
            executed_indices_all = joint_data.get("executed_indices_all")
            if executed_indices_all is not None:
                executed_indices_all = executed_indices_all.to(device)  # [batch, joint_seq_len, num_layers, k]
            else:
                # Backward compat: fall back to current_indices_all if executed not recorded
                executed_indices_all = joint_data["current_indices_all"].to(device)
                if self.accelerator.is_main_process:
                    print("  [INTRA-OPT-DEBUG] WARNING: executed_indices_all not found, falling back to current_indices_all", flush=True)
            moe_layer_indices = self.joint_controller.moe_layer_indices
            num_experts = self.joint_controller.num_experts
            joint_seq_len = executed_indices_all.shape[1]
            
            # Per-layer data has seq_len = query_len + response_len - 1 (prefill + gen)
            first_layer_idx = moe_layer_indices[0]
            first_layer_data = rollout.layer_data.get(first_layer_idx)
            recorded_seq_len = first_layer_data["router_logits"].shape[1] if first_layer_data is not None else total_len
            prefill_indices_all = joint_data.get("prefill_indices_all")  # [batch, query_len-1, num_moe_layers, k] or None
            
            if self.accelerator.is_main_process:
                has_prefill = prefill_indices_all is not None
                prefill_shape = prefill_indices_all.shape if has_prefill else None
                print(f"  [INTRA-OPT-DEBUG] Joint mode: total_len={total_len}, recorded_seq_len={recorded_seq_len}, "
                      f"joint_seq_len={joint_seq_len}, query_len={query_len}, response_len={response_len}, "
                      f"has_prefill_masks={has_prefill}, prefill_shape={prefill_shape}", flush=True)
            
            # Build per-token masks for the full sequence [batch, total_len, num_experts]
            # Prefill positions: per-token options recorded during sequential prefill
            # Generation positions: executed_indices_all (the option that was actually used)
            
            joint_masks_per_layer = {}
            for pos, layer_idx in enumerate(moe_layer_indices):
                per_token_mask = torch.zeros(batch_size, total_len, num_experts, dtype=torch.bool, device=device)
                
                prefill_len = min(query_len - 1, total_len)
                if prefill_indices_all is not None and prefill_len > 0:
                    prefill_for_layer = prefill_indices_all[:, :prefill_len, pos, :]  # [batch, prefill_len, k]
                    per_token_mask[:, :prefill_len, :].scatter_(2, prefill_for_layer.to(device), True)
                    # Position 0 used vanilla routing (no mask) during generation,
                    # so allow all experts to match the original softmax distribution
                    per_token_mask[:, 0, :] = True
                elif prefill_len > 0:
                    # Fallback: no prefill data recorded (shouldn't happen in joint mode with query_len>1)
                    first_option_indices = executed_indices_all[:, 0, pos, :]  # [batch, k]
                    prefill_mask = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
                    prefill_mask.scatter_(1, first_option_indices, True)
                    per_token_mask[:, :prefill_len, :] = prefill_mask.unsqueeze(1).expand(-1, prefill_len, -1)
                
                gen_start = query_len - 1
                gen_len = min(joint_seq_len, total_len - gen_start)
                if gen_len > 0:
                    gen_indices = executed_indices_all[:, :gen_len, pos, :]  # [batch, gen_len, k]
                    per_token_mask[:, gen_start:gen_start + gen_len, :].scatter_(2, gen_indices, True)
                    # If no prefill occurred (query_len<=1), the first generation step used
                    # vanilla routing (no mask), so allow all experts at that position
                    if prefill_indices_all is None and gen_len > 0:
                        per_token_mask[:, gen_start, :] = True
                
                # Pad remaining positions (if total_len > gen_start + gen_len)
                if gen_len > 0 and gen_start + gen_len < total_len:
                    last_indices = executed_indices_all[:, gen_len - 1, pos, :]
                    last_mask = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
                    last_mask.scatter_(1, last_indices, True)
                    remaining = total_len - (gen_start + gen_len)
                    per_token_mask[:, gen_start + gen_len:, :] = last_mask.unsqueeze(1).expand(-1, remaining, -1)
                
                joint_masks_per_layer[layer_idx] = per_token_mask  # [batch, total_len, num_experts]
            
            controller_runtime = {
                "sampling": False,
                "joint_option_mode": True,
                "joint_option_masks": joint_masks_per_layer,
            }
        else:
            # Per-layer mode: build replay_actions as before
            replay_actions = {}
            for layer_idx, layer_data in rollout.layer_data.items():
                recorded_seq_len = layer_data["switches"].shape[1]
                
                if self.accelerator.is_main_process and layer_idx == 0:
                    print(f"  [INTRA-OPT-DEBUG] total_len={total_len}, recorded_seq_len={recorded_seq_len}, query_len={query_len}, response_len={response_len}", flush=True)
                
                switches = layer_data["switches"]
                selected_indices = layer_data["selected_indices"]
                
                if recorded_seq_len < total_len:
                    pad_size = total_len - recorded_seq_len
                    switches = F.pad(switches.float(), (0, pad_size), value=0.0).bool()
                    selected_indices = F.pad(selected_indices, (0, 0, 0, pad_size), value=0)
                    if self.accelerator.is_main_process and layer_idx == 0:
                        print(f"  [INTRA-OPT-DEBUG] Padded replay actions from {recorded_seq_len} to {total_len}", flush=True)
                
                replay_actions[layer_idx] = {
                    "switches": switches,
                    "selected_indices": selected_indices,
                }
            
            controller_runtime = {
                "sampling": False,
                "replay_actions": replay_actions,
            }
        
        # Get the unwrapped model for forward pass
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        policy = model
        if hasattr(policy, 'policy'):
            policy = policy.policy
        
        # Ensure model is in training mode for gradient tracking
        policy.train()
        
        # DEBUG: Check trainable params in the model we're using
        if self.accelerator.is_main_process:
            trainable_count = sum(1 for p in policy.parameters() if p.requires_grad)
            total_count = sum(1 for p in policy.parameters())
            print(f"  [INTRA-OPT-DEBUG] policy type: {type(policy).__name__}, "
                  f"trainable params: {trainable_count}/{total_count}", flush=True)
        
        # Joint mode: disable per-layer controllers for replay (masks are external)
        if is_joint_mode:
            self._set_controller_enabled(policy, False)
        
        # CRITICAL: Wrap forward pass with torch.enable_grad() to ensure gradient tracking
        # even if we're in a no_grad context or gradient checkpointing is interfering
        with torch.enable_grad():
            outputs = policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                controller_runtime=controller_runtime,
                use_cache=False,
            )
            logits = outputs.logits  # [batch, total_len, vocab_size]
        
        # Re-enable per-layer controllers after replay
        if is_joint_mode:
            self._set_controller_enabled(policy, True)
        
        if self.accelerator.is_main_process:
            print(f"  [INTRA-OPT-GRAD] logits.requires_grad={logits.requires_grad}, "
                  f"policy.training={policy.training}", flush=True)
        
        # Shift logits and labels for next-token prediction
        # logits[:, :-1, :] predicts tokens at positions 1:total_len
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, total_len-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()  # [batch, total_len-1]
        
        # Apply temperature scaling to match rollout sampling
        # During rollout, tokens are sampled with temperature T, so log-probs should be
        # computed under the same tempered distribution: log p_T(token) = logits/T - logsumexp(logits/T)
        token_temperature = self.config.temperature
        if token_temperature != 1.0:
            shift_logits = shift_logits / token_temperature
        
        # Compute log probabilities of the actual tokens (under tempered distribution)
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, total_len-1, vocab_size]
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, total_len-1]
        
        # The response tokens start at position query_len-1 in shift_labels
        # (because shift_labels starts at position 1 of input_ids)
        # Response positions in shift_labels: [query_len-1, query_len-1+response_len)
        response_start_idx = query_len - 1
        response_token_log_probs = token_log_probs[:, response_start_idx:response_start_idx + response_len]
        # [batch, response_len]
        
        # Aggregate intra-option advantages across all layers
        # When intra_option_q_baseline is enabled, use G - Q(s,o) (A2OC Algorithm 1)
        # Otherwise use raw G (on-policy distillation style)
        use_q_baseline = getattr(self.config, 'intra_option_q_baseline', False) and intra_option_q_values
        total_advantage = None
        total_valid_mask = None
        num_layers = 0
        
        # Determine available response length from first layer's advantage
        first_layer_idx = next(iter(intra_option_advantages.keys()))
        first_advantage = intra_option_advantages[first_layer_idx]
        adv_seq_len = first_advantage.shape[1]
        # Both advantages and masks start at query_len - 1 (the prediction position for
        # the first response token). The valid_mask at position p tells us whether the
        # action at position p is within a valid sequence, which aligns with the advantage
        # at position p.
        adv_start = query_len - 1
        mask_start = query_len - 1
        # available_response_len must account for BOTH advantage and mask slicing
        adv_available = adv_seq_len - adv_start if adv_seq_len > adv_start else 0
        mask_available = adv_seq_len - mask_start if adv_seq_len > mask_start else 0
        available_response_len = min(adv_available, mask_available, response_len)
        
        if available_response_len <= 0:
            if self.accelerator.is_main_process:
                print(f"  [INTRA-OPT-SHAPE] No response tokens with advantages (adv_seq_len={adv_seq_len}, query_len={query_len})", flush=True)
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Slice response_token_log_probs to match available_response_len
        assert response_token_log_probs.shape[1] >= available_response_len, \
            f"Not enough tokens: response_token_log_probs has {response_token_log_probs.shape[1]} but need {available_response_len}"
        response_token_log_probs = response_token_log_probs[:, :available_response_len]
        
        if self.accelerator.is_main_process:
            print(f"  [INTRA-OPT-SHAPE] query_len={query_len}, response_len={response_len}, "
                  f"available_response_len={available_response_len}, "
                  f"response_token_log_probs.shape={response_token_log_probs.shape}", flush=True)
        
        for layer_idx, advantage in intra_option_advantages.items():
            valid_mask = valid_masks[layer_idx]
            
            if self.accelerator.is_main_process and layer_idx == 0:
                print(f"  [INTRA-OPT-SHAPE] Layer {layer_idx}: advantage.shape={advantage.shape}, "
                      f"valid_mask.shape={valid_mask.shape}, adv_start={adv_start}", flush=True)
            
            adv_seq_len = advantage.shape[1]
            adv_start = query_len - 1
            
            # Calculate how many response tokens we can use
            layer_adv_available = adv_seq_len - adv_start if adv_seq_len > adv_start else 0
            layer_mask_available = adv_seq_len - mask_start if adv_seq_len > mask_start else 0
            layer_available_len = min(layer_adv_available, layer_mask_available, response_len)
            
            if layer_available_len <= 0:
                if self.accelerator.is_main_process and layer_idx == 0:
                    print(f"  [INTRA-OPT-SHAPE] Skipping layer {layer_idx}: no response tokens with advantages", flush=True)
                continue
            
            # Assert bounds before slicing
            assert adv_start + available_response_len <= advantage.shape[1], \
                f"Layer {layer_idx}: advantage slice out of bounds"
            assert mask_start + available_response_len <= valid_mask.shape[1], \
                f"Layer {layer_idx}: valid_mask slice out of bounds"
            
            adv_response = advantage[:, adv_start:adv_start + available_response_len]
            mask_response = valid_mask[:, mask_start:mask_start + available_response_len]
            
            # Subtract Q baseline if enabled (A2OC: advantage = G - Q(s,o))
            if use_q_baseline and layer_idx in intra_option_q_values:
                q_response = intra_option_q_values[layer_idx][:, adv_start:adv_start + available_response_len]
                adv_response = adv_response - q_response
            
            # Verify shapes after slicing
            assert adv_response.shape == response_token_log_probs.shape, \
                f"Layer {layer_idx}: adv_response {adv_response.shape} != response_token_log_probs {response_token_log_probs.shape}"
            assert mask_response.shape == response_token_log_probs.shape, \
                f"Layer {layer_idx}: mask_response {mask_response.shape} != response_token_log_probs {response_token_log_probs.shape}"
            
            if total_advantage is None:
                total_advantage = adv_response.clone()
                total_valid_mask = mask_response.clone()
            else:
                total_advantage = total_advantage + adv_response
                total_valid_mask = total_valid_mask & mask_response
            
            num_layers += 1
        
        if total_advantage is None or num_layers == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Average advantage across layers
        avg_advantage = total_advantage / num_layers  # [batch, available_response_len]
        
        # Shape assertions
        assert avg_advantage.shape == response_token_log_probs.shape
        assert total_valid_mask.shape == response_token_log_probs.shape
        
        # Normalize advantage (standard practice for policy gradients)
        valid_advs = avg_advantage[total_valid_mask]
        if valid_advs.numel() > 1:
            adv_mean = valid_advs.mean()
            adv_std = valid_advs.std().clamp(min=1e-8)
            normalized_advantage = (avg_advantage - adv_mean) / adv_std
        elif valid_advs.numel() == 1:
            normalized_advantage = avg_advantage
        else:
            if self.accelerator.is_main_process:
                print(f"  [INTRA-OPTION] WARNING: No valid tokens for intra-option loss", flush=True)
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # =========================================================================
        # Apply importance weights for teacher-mixed sampling (MiniLLM)
        # When teacher_mix_alpha > 0, tokens are sampled from mixed distribution:
        # p_mixed = α * p_teacher + (1-α) * p_student
        # Importance weight w_t = p_student(token_t) / p_mixed(token_t)
        # This corrects for the off-policy sampling
        # Reference: https://arxiv.org/pdf/2306.08543 (Section 2.2)
        # =========================================================================
        importance_weights = None
        if rollout.importance_weights is not None:
            # Slice importance weights to match available_response_len
            iw = rollout.importance_weights[:, :available_response_len]
            if iw.shape == response_token_log_probs.shape:
                importance_weights = iw
                if self.accelerator.is_main_process:
                    valid_iw = importance_weights[total_valid_mask]
                    print(f"  [INTRA-OPTION] Applying importance weights: mean={valid_iw.mean().item():.4f}, "
                          f"std={valid_iw.std().item() if valid_iw.numel() > 1 else 0:.4f}", flush=True)
            else:
                if self.accelerator.is_main_process:
                    print(f"  [INTRA-OPTION] Warning: importance_weights shape {iw.shape} != "
                          f"response_token_log_probs shape {response_token_log_probs.shape}, skipping", flush=True)
        
        # Compute intra-option policy loss: -w * A * log_prob (sum over valid tokens)
        # With importance weights: loss = sum(-w_t * A_t * log_prob_t) / sum(w_t)
        # Without: loss = sum(-A_t * log_prob_t) / num_valid
        if importance_weights is not None:
            weighted_loss = torch.where(
                total_valid_mask,
                -importance_weights * normalized_advantage * response_token_log_probs,
                torch.zeros_like(response_token_log_probs)
            )
            # Normalize by sum of weights for valid tokens
            weight_sum = (importance_weights * total_valid_mask.float()).sum().clamp(min=1e-8)
            intra_option_loss = weighted_loss.sum() / weight_sum
        else:
            masked_loss = torch.where(
                total_valid_mask,
                -normalized_advantage * response_token_log_probs,
                torch.zeros_like(response_token_log_probs)
            )
            num_valid_tokens = total_valid_mask.sum().clamp(min=1)
            intra_option_loss = masked_loss.sum() / num_valid_tokens
        
        if self.accelerator.is_main_process:
            baseline_str = "G - Q" if use_q_baseline else "G"
            iw_str = f", importance_weighted=True" if importance_weights is not None else ""
            valid_log_probs = response_token_log_probs[total_valid_mask]
            valid_norm_adv = normalized_advantage[total_valid_mask]
            print(f"  [INTRA-OPTION] Loss: {intra_option_loss.item():.6f}, "
                  f"avg_adv ({baseline_str}): mean={valid_advs.mean().item():.4f}, std={valid_advs.std().item() if valid_advs.numel() > 1 else 0:.4f}, "
                  f"token_temp: {token_temperature}{iw_str}", flush=True)
            print(f"  [INTRA-OPTION] log_probs: mean={valid_log_probs.mean().item():.4f}, "
                  f"min={valid_log_probs.min().item():.4f}, max={valid_log_probs.max().item():.4f}, "
                  f"std={valid_log_probs.std().item():.4f}", flush=True)
            print(f"  [INTRA-OPTION] norm_adv: mean={valid_norm_adv.mean().item():.4f}, "
                  f"min={valid_norm_adv.min().item():.4f}, max={valid_norm_adv.max().item():.4f}, "
                  f"std={valid_norm_adv.std().item():.4f}", flush=True)
            if importance_weights is not None:
                valid_iw_product = (importance_weights * normalized_advantage * response_token_log_probs)[total_valid_mask]
                print(f"  [INTRA-OPTION] |w*A*logp|: mean={valid_iw_product.abs().mean().item():.4f}, "
                      f"max={valid_iw_product.abs().max().item():.4f}, "
                      f"num_valid={total_valid_mask.sum().item()}", flush=True)
        
        return intra_option_loss
    
    def train_step_with_accumulation(
        self,
        batch_queries: List[torch.Tensor],
        batch_ground_truth_answers: Optional[List[Optional[List[str]]]] = None,
    ) -> Dict[str, float]:
        """
        Training step with gradient accumulation (matches original ControllerTrainer).
        
        Args:
            batch_queries: List of [batch, query_len] tensors, one per accumulation step
            batch_ground_truth_answers: Optional list of ground truth answer lists, one per accum step
            
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
            
            # Get ground truth answers for this accumulation step if available
            accum_answers = None
            if batch_ground_truth_answers is not None and accum_idx < len(batch_ground_truth_answers):
                accum_answers = batch_ground_truth_answers[accum_idx]
            
            with torch.no_grad():
                rollout = self.generate_rollout(queries, ground_truth_answers=accum_answers)
            
            rollouts.append(rollout)
            all_local_rewards.append(rollout.rewards)
            all_base_rewards.append(rollout.base_rewards)
            all_response_lengths.append(rollout.response_lengths)
        
        rollout_time = time.time() - rollout_start
        
        # Accumulate correctness stats across rollouts
        total_correct = 0
        total_with_answers = 0
        for rollout in rollouts:
            if rollout.correctness is not None:
                total_correct += rollout.correctness.sum().item()
                total_with_answers += rollout.correctness.numel()
        self._last_correctness_rate = total_correct / max(total_with_answers, 1)
        self._last_correctness_count = total_correct
        self._last_correctness_total = total_with_answers
        
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
        total_term_loss = 0.0
        total_select_loss = 0.0
        total_V_loss = 0.0
        total_Q_loss = 0.0
        
        # Multiple update epochs on the same batch (like PPO)
        for epoch_idx in range(self.config.num_update_epochs):
            if self.accelerator.is_main_process and epoch_idx == 0:
                print(f"  [UPDATE] Running {self.config.num_update_epochs} update epochs with {grad_accum_steps} accumulation steps...", flush=True)
            
            # Zero gradients at start of each epoch
            self.optimizer.zero_grad()
            if self.llm_optimizer is not None:
                self.llm_optimizer.zero_grad()
            
            for accum_idx, rollout in enumerate(rollouts):
                # Compute loss (scaled for accumulation)
                if self.joint_option and self.joint_controller is not None:
                    result = self._compute_single_rollout_loss_joint(rollout, scale_factor=scale_factor)
                else:
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
                
                # =========================================================
                # Intra-option policy update on LLM (Harb et al. 2017, Algorithm 1)
                # This is a SEPARATE backward pass for LoRA + router parameters
                # =========================================================
                intra_option_advantages = result.get("intra_option_advantages", {})
                intra_option_q_vals = result.get("intra_option_q_values", {})
                valid_masks_per_layer = result.get("valid_masks_per_layer", {})
                
                # Skip intra-option updates during warmup period (let value function warm up first)
                in_warmup = self.global_step <= self.config.intra_option_warmup_steps
                if self.lora_enabled and intra_option_advantages and self.llm_optimizer is not None and not in_warmup:
                    intra_option_loss = self._compute_intra_option_loss(
                        rollout, intra_option_advantages, valid_masks_per_layer,
                        intra_option_q_values=intra_option_q_vals,
                    )
                    # Scale by same factor as controller loss for gradient accumulation
                    scaled_intra_loss = intra_option_loss * scale_factor
                    scaled_intra_loss.backward()
                    
                    if self.accelerator.is_main_process and epoch_idx == 0 and accum_idx == 0:
                        print(f"  [INTRA-OPTION] Backward pass completed, loss={intra_option_loss.item():.6f}", flush=True)
                        
                        # Check LoRA and router gradients
                        lora_grad_norms = []
                        router_grad_norms = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if 'lora_' in name:
                                    lora_grad_norms.append(grad_norm)
                                elif 'router' in name:
                                    router_grad_norms.append(grad_norm)
                        
                        if lora_grad_norms:
                            avg_lora = sum(lora_grad_norms) / len(lora_grad_norms)
                            print(f"  [GRAD-CHECK] LoRA params with grads: {len(lora_grad_norms)}, avg grad norm: {avg_lora:.6f}", flush=True)
                        else:
                            print(f"  [GRAD-CHECK] WARNING: No LoRA params have gradients!", flush=True)
                        
                        if router_grad_norms:
                            avg_router = sum(router_grad_norms) / len(router_grad_norms)
                            print(f"  [GRAD-CHECK] Router params with grads: {len(router_grad_norms)}, avg grad norm: {avg_router:.6f}", flush=True)
                        else:
                            print(f"  [GRAD-CHECK] WARNING: No router params have gradients!", flush=True)
                elif in_warmup and self.lora_enabled and self.accelerator.is_main_process and epoch_idx == 0 and accum_idx == 0:
                    print(f"  [INTRA-OPTION] Skipping LLM update (warmup: step {self.global_step}/{self.config.intra_option_warmup_steps})", flush=True)
                
                # Accumulate metrics (only on first epoch)
                if epoch_idx == 0:
                    total_loss += result["loss_value"]
                    total_policy_loss += result["policy_loss"]
                    total_value_loss += result["value_loss"]
                    total_switch_rate += result["switch_rate"]
                    if "mean_term_loss" in result:
                        total_term_loss += result["mean_term_loss"]
                        total_select_loss += result["mean_select_loss"]
                        total_V_loss += result["mean_V_loss"]
                        total_Q_loss += result["mean_Q_loss"]
                    
                    # Store termination binariness metrics from last rollout (for logging)
                    if "switch_prob_mean" in result:
                        self._last_switch_prob_mean = result["switch_prob_mean"]
                        self._last_switch_prob_std = result["switch_prob_std"]
                        self._last_switch_binary_frac = result["switch_binary_frac"]
                        self._last_switch_entropy = result["switch_entropy"]
                        self._last_switch_prob_max = result["switch_prob_max"]
                        self._last_switch_prob_p90 = result["switch_prob_p90"]
                        self._last_switch_prob_p95 = result["switch_prob_p95"]
                        self._last_frac_gt_0p1 = result["frac_gt_0p1"]
                        self._last_frac_gt_0p5 = result["frac_gt_0p5"]
            
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
            
            # LLM optimizer step (for intra-option policy update)
            # Skip during warmup (same as skipping the backward pass)
            in_warmup_for_llm = self.global_step <= self.config.intra_option_warmup_steps
            if self.llm_optimizer is not None and not in_warmup_for_llm:
                # Sync LLM gradients across GPUs
                if torch.distributed.is_initialized():
                    llm_params = self._get_llm_trainable_params()
                    for p in llm_params:
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
                
                # Gradient clipping for LLM
                if self.config.max_grad_norm is not None:
                    llm_params = self._get_llm_trainable_params()
                    torch.nn.utils.clip_grad_norm_(llm_params, self.config.max_grad_norm)
                
                # Record param norms BEFORE optimizer step
                lora_norms_before = []
                router_norms_before = []
                if self.accelerator.is_main_process and epoch_idx == 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param_norm = param.data.norm().item()
                            if 'lora_' in name:
                                lora_norms_before.append(param_norm)
                            elif 'router' in name:
                                router_norms_before.append(param_norm)
                
                # DEBUG: Check if optimizer's params have gradients before step
                if self.accelerator.is_main_process and epoch_idx == 0:
                    # Get all params from optimizer
                    opt_param_ids = set()
                    for group in self.llm_optimizer.param_groups:
                        for p in group['params']:
                            opt_param_ids.add(id(p))
                    
                    # Get all params from model that have gradients
                    model_params_with_grad = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            model_params_with_grad[id(param)] = (name, param.grad.norm().item())
                    
                    # Check overlap
                    opt_params_with_grad = 0
                    opt_params_no_grad = 0
                    model_grads_not_in_opt = 0
                    
                    for group in self.llm_optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                opt_params_with_grad += 1
                            else:
                                opt_params_no_grad += 1
                    
                    for pid in model_params_with_grad:
                        if pid not in opt_param_ids:
                            name, grad_norm = model_params_with_grad[pid]
                            if 'controller' not in name:  # Exclude controller params
                                model_grads_not_in_opt += 1
                    
                    total_opt_params = sum(len(g['params']) for g in self.llm_optimizer.param_groups)
                    print(f"  [OPT-DEBUG] Optimizer: {opt_params_with_grad} have grad, {opt_params_no_grad} no grad, total={total_opt_params}", flush=True)
                    print(f"  [OPT-DEBUG] Model params with grad NOT in optimizer: {model_grads_not_in_opt}", flush=True)
                
                self.llm_optimizer.step()
                self.llm_optimizer.zero_grad()
                
                # Record param norms AFTER optimizer step and compute delta
                if self.accelerator.is_main_process and epoch_idx == 0:
                    lora_norms_after = []
                    router_norms_after = []
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param_norm = param.data.norm().item()
                            if 'lora_' in name:
                                lora_norms_after.append(param_norm)
                            elif 'router' in name:
                                router_norms_after.append(param_norm)
                    
                    if lora_norms_before and lora_norms_after:
                        avg_before = sum(lora_norms_before) / len(lora_norms_before)
                        avg_after = sum(lora_norms_after) / len(lora_norms_after)
                        delta = avg_after - avg_before
                        print(f"  [LLM-UPDATE] LoRA: before={avg_before:.8f}, after={avg_after:.8f}, delta={delta:.6e}", flush=True)
                    if router_norms_before and router_norms_after:
                        avg_before = sum(router_norms_before) / len(router_norms_before)
                        avg_after = sum(router_norms_after) / len(router_norms_after)
                        delta = avg_after - avg_before
                        max_delta = max(abs(a - b) for a, b in zip(router_norms_after, router_norms_before))
                        print(f"  [LLM-UPDATE] Router: before={avg_before:.8f}, after={avg_after:.8f}, delta={delta:.6e}, max_delta={max_delta:.6e}", flush=True)
                    print(f"  [LLM-UPDATE] LLM optimizer step completed", flush=True)
            
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
            "mean_term_loss": total_term_loss / n_accum,
            "mean_select_loss": total_select_loss / n_accum,
            "mean_V_loss": total_V_loss / n_accum,
            "mean_Q_loss": total_Q_loss / n_accum,
            "reward_mean": global_reward_mean,
            "switch_rate": global_switch_rate,
            "batch_size": total_batch_size,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "step_time": step_time,
            "gradient_accumulation_steps": grad_accum_steps,
        }
        
        # Add termination binariness metrics if available
        if hasattr(self, '_last_switch_prob_mean'):
            metrics["switch_prob_mean"] = self._last_switch_prob_mean
            metrics["switch_prob_std"] = self._last_switch_prob_std
            metrics["switch_binary_frac"] = self._last_switch_binary_frac
            metrics["switch_entropy"] = self._last_switch_entropy
            metrics["switch_prob_max"] = self._last_switch_prob_max
            metrics["switch_prob_p90"] = self._last_switch_prob_p90
            metrics["switch_prob_p95"] = self._last_switch_prob_p95
            metrics["frac_gt_0p1"] = self._last_frac_gt_0p1
            metrics["frac_gt_0p5"] = self._last_frac_gt_0p5
        
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
        
        # Calculate how many dataloader batches to skip when resuming
        resume_skip_batches = self.global_step * grad_accum_steps if self.global_step > 0 else 0
        if resume_skip_batches > 0 and accelerator.is_main_process:
            print(f"[RESUME] Skipping {resume_skip_batches} batches ({self.global_step} steps x {grad_accum_steps} accum) to resume from step {self.global_step}")
        
        for epoch in range(config.num_train_epochs):
            self.epoch = epoch
            
            # Reset running metrics at start of epoch
            running_metrics = {"loss": 0.0, "reward": 0.0, "switches": 0.0}
            running_count = 0
            
            # Create progress bar (only on main process)
            # Progress is per optimizer step, not per batch
            total_steps = len(self.train_dataloader) // grad_accum_steps
            if accelerator.is_main_process:
                pbar = tqdm(
                    total=total_steps,
                    desc=f"Epoch {epoch+1}/{config.num_train_epochs}",
                    dynamic_ncols=True,
                    leave=True,
                    initial=min(self.global_step, total_steps),
                )
            
            # Collect batches for gradient accumulation
            batch_queries = []
            batch_gt_answers = []
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Skip batches that were already processed before checkpoint
                if batch_idx < resume_skip_batches:
                    if batch_idx == 0 and accelerator.is_main_process:
                        print(f"[RESUME] Skipping batches...", flush=True)
                    continue
                if batch_idx == resume_skip_batches and resume_skip_batches > 0 and accelerator.is_main_process:
                    print(f"[RESUME] Done skipping, resuming training from batch {batch_idx}", flush=True)
                
                if isinstance(batch, dict):
                    queries = batch["input_ids"]
                    gt_answers = batch.get("ground_truth_answers", None)
                else:
                    queries = batch[0]
                    gt_answers = None
                
                queries = queries.to(accelerator.device)
                batch_queries.append(queries)
                batch_gt_answers.append(gt_answers)
                
                # Check if we've collected enough batches for an optimizer step
                if len(batch_queries) < grad_accum_steps:
                    continue
                
                # Training step with accumulation
                # Pass ground truth answers if any batch has them
                has_answers = any(a is not None for a in batch_gt_answers)
                metrics = self.train_step_with_accumulation(
                    batch_queries,
                    batch_ground_truth_answers=batch_gt_answers if has_answers else None,
                )
                
                # Clear the batch buffer
                batch_queries = []
                batch_gt_answers = []
                
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
                        "train/mean_term_loss": metrics["mean_term_loss"],
                        "train/mean_select_loss": metrics["mean_select_loss"],
                        "train/mean_V_loss": metrics["mean_V_loss"],
                        "train/mean_Q_loss": metrics["mean_Q_loss"],
                        "train/reward_mean": metrics["reward_mean"],
                        "train/switch_rate": metrics["switch_rate"],
                        "train/batch_size": metrics["batch_size"],
                        "timing/step_time": metrics["step_time"],
                        "timing/rollout_time": metrics["rollout_time"],
                        "timing/update_time": metrics["update_time"],
                        "progress/epoch": epoch,
                        "progress/global_step": self.global_step,
                        "exploration/pl_epsilon": self._get_current_epsilon(),
                        "exploration/q_epsilon": self._get_current_q_epsilon(),
                    }
                    # Add KL reward metrics if available
                    if self.ppl_scorer is not None and hasattr(self.ppl_scorer, 'last_batch_kl_mean'):
                        log_dict["reward/kl_mean"] = self.ppl_scorer.last_batch_kl_mean
                        log_dict["reward/kl_std"] = self.ppl_scorer.last_batch_kl_std
                        log_dict["reward/teacher_ppl_mean"] = self.ppl_scorer.last_batch_teacher_ppl_mean
                        log_dict["reward/student_ppl_mean"] = self.ppl_scorer.last_batch_student_ppl_mean
                    
                    # Add repetition metrics
                    if hasattr(self, '_last_rep_rate_mean'):
                        log_dict["repetition/rate_mean"] = self._last_rep_rate_mean
                        log_dict["repetition/rate_max"] = self._last_rep_rate_max
                        log_dict["repetition/num_repeats_mean"] = self._last_num_repeats_mean
                        log_dict["repetition/repeat_frac"] = self._last_repeat_frac_mean
                        log_dict["repetition/penalty_mean"] = self._last_rep_penalty_mean
                        # Total repetition penalty per sample (sum over sequence, mean over batch)
                        if hasattr(self, '_last_rep_penalty_sum_mean'):
                            log_dict["repetition/penalty_sum_mean"] = self._last_rep_penalty_sum_mean
                            # Reward with repetition penalty included
                            # reward_mean is base_reward (per-token mean KL) - latency_penalty
                            # rep_penalty_sum_mean is sum over sequence (before /512 normalization)
                            # Divide by 512 to put on same per-token scale as reward_mean
                            REWARD_NORMALIZATION_CONSTANT = 512.0
                            rep_penalty_per_token = self._last_rep_penalty_sum_mean / REWARD_NORMALIZATION_CONSTANT
                            log_dict["train/reward_with_rep_penalty"] = metrics["reward_mean"] + rep_penalty_per_token
                    
                    # Add TopK termination regularization metrics
                    if hasattr(self, '_last_term_topk_mean'):
                        log_dict["termination/topk_mean"] = self._last_term_topk_mean
                        log_dict["termination/topk_loss"] = self._last_term_topk_loss
                    
                    # Add correctness metrics
                    if hasattr(self, '_last_correctness_total') and self._last_correctness_total > 0:
                        log_dict["reward/correctness_rate"] = self._last_correctness_rate
                        log_dict["reward/correctness_count"] = self._last_correctness_count
                        log_dict["reward/correctness_total"] = self._last_correctness_total
                    
                    # Add expert entropy metric (mode collapse indicator)
                    if hasattr(self, '_last_expert_entropy'):
                        log_dict["train/expert_entropy"] = self._last_expert_entropy
                    
                    # Add termination binariness metrics
                    if "switch_prob_mean" in metrics:
                        log_dict["termination/switch_prob_mean"] = metrics["switch_prob_mean"]
                        log_dict["termination/switch_prob_std"] = metrics["switch_prob_std"]
                        log_dict["termination/binary_frac"] = metrics["switch_binary_frac"]
                        log_dict["termination/entropy"] = metrics["switch_entropy"]
                        # High switch probability metrics
                        log_dict["termination/switch_prob_max"] = metrics["switch_prob_max"]
                        log_dict["termination/switch_prob_p90"] = metrics["switch_prob_p90"]
                        log_dict["termination/switch_prob_p95"] = metrics["switch_prob_p95"]
                        log_dict["termination/frac_gt_0.1"] = metrics["frac_gt_0p1"]
                        log_dict["termination/frac_gt_0.5"] = metrics["frac_gt_0p5"]
                    
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
    
    def _get_router_state_dict(self):
        """Extract router weight state dict from the model."""
        router_state = {}
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
        for name, param in model.named_parameters():
            if '.router.' in name or 'router.weight' in name or 'router.bias' in name:
                router_state[name] = param.data.cpu().clone()
        return router_state
    
    def _load_router_state_dict(self, router_state):
        """Load router weights back into the model."""
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
        model_state = dict(model.named_parameters())
        loaded = 0
        for name, saved_param in router_state.items():
            if name in model_state:
                model_state[name].data.copy_(saved_param.to(model_state[name].device))
                loaded += 1
        return loaded
    
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
                "router_state_dict": self._get_router_state_dict(),
            }
            
            # Save joint controller if in joint option mode
            if self.joint_option and self.joint_controller is not None:
                checkpoint["joint_controller"] = self.joint_controller.state_dict()
                checkpoint["joint_option"] = True
            
            # Save LLM optimizer state if intra-option update is enabled
            if self.llm_optimizer is not None:
                checkpoint["llm_optimizer_state_dict"] = self.llm_optimizer.state_dict()
            
            torch.save(checkpoint, path)
            print(f"[CHECKPOINT] Saved to {path}")
            if self.joint_option:
                print(f"[CHECKPOINT] Includes joint controller state")
            
            # Save PEFT/LoRA weights if enabled
            if self.lora_enabled and self.peft_model is not None:
                import os
                checkpoint_dir = os.path.dirname(path)
                lora_dir = os.path.join(checkpoint_dir, f"lora_step_{self.global_step}")
                self.peft_model.save_pretrained(lora_dir)
                print(f"[CHECKPOINT] Saved LoRA to {lora_dir}")
                
                # Also save to a "latest" folder for easy access
                lora_latest_dir = os.path.join(checkpoint_dir, "lora_latest")
                self.peft_model.save_pretrained(lora_latest_dir)
                print(f"[CHECKPOINT] Saved LoRA (latest) to {lora_latest_dir}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load activation controller checkpoint.
        
        Restores: controller params, optimizer states, LoRA weights, global_step, epoch.
        The training loop uses global_step to skip already-completed steps.
        """
        import os
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        # Load joint controller if in joint option mode
        if self.joint_option and self.joint_controller is not None and "joint_controller" in checkpoint:
            self.joint_controller.load_state_dict(checkpoint["joint_controller"])
            if self.accelerator.is_main_process:
                print(f"[CHECKPOINT] Loaded joint controller state")
        elif self.joint_option and "joint_controller" not in checkpoint:
            if self.accelerator.is_main_process:
                print(f"  [WARN] Joint option mode enabled but no joint_controller in checkpoint. Starting fresh.")
        
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
        
        # Load LLM optimizer state if available and intra-option update is enabled
        if load_optimizer and self.llm_optimizer is not None and "llm_optimizer_state_dict" in checkpoint:
            try:
                self.llm_optimizer.load_state_dict(checkpoint["llm_optimizer_state_dict"])
            except Exception as e:
                print(f"  [WARN] Failed to load LLM optimizer: {e}")
                import traceback
                traceback.print_exc()
        
        # Load LoRA weights if available
        if self.lora_enabled and self.peft_model is not None:
            checkpoint_dir = os.path.dirname(path)
            step = checkpoint.get("step", 0)
            lora_dir = os.path.join(checkpoint_dir, f"lora_step_{step}")
            if not os.path.exists(lora_dir):
                lora_dir = os.path.join(checkpoint_dir, "lora_latest")
            if os.path.exists(lora_dir):
                try:
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file
                    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
                    bin_path = os.path.join(lora_dir, "adapter_model.bin")
                    if os.path.exists(safetensors_path):
                        adapter_state = load_file(safetensors_path)
                    elif os.path.exists(bin_path):
                        adapter_state = torch.load(bin_path, map_location="cpu", weights_only=True)
                    else:
                        raise FileNotFoundError(f"No adapter weights found in {lora_dir}")
                    set_peft_model_state_dict(self.peft_model, adapter_state)
                    if self.accelerator.is_main_process:
                        print(f"[CHECKPOINT] Loaded LoRA weights from {lora_dir} ({len(adapter_state)} tensors)")
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"  [WARN] Failed to load LoRA weights from {lora_dir}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                if self.accelerator.is_main_process:
                    print(f"  [WARN] No LoRA weights found at {lora_dir}, using fresh LoRA")
        
        # Load router weights if available
        if "router_state_dict" in checkpoint:
            loaded = self._load_router_state_dict(checkpoint["router_state_dict"])
            if self.accelerator.is_main_process:
                print(f"[CHECKPOINT] Loaded {loaded} router weight tensors")
        else:
            if self.accelerator.is_main_process:
                print(f"  [WARN] No router weights in checkpoint (pre-existing checkpoints won't have them)")
        
        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        if self.accelerator.is_main_process:
            print(f"[CHECKPOINT] Loaded from {path}, step={self.global_step}")

