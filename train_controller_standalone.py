#!/usr/bin/env python3
"""
Standalone Controller Training Script for gpt-oss MoE models.

This script trains only the controller using REINFORCE with value baseline,
without modifying the LLM weights. The controller is treated as an RNN-like
policy that makes sequential decisions across tokens and layers.

Usage:
    accelerate launch --config_file accelerate_config.yaml train_controller_standalone.py

Or via SLURM:
    sbatch run_controller.slurm
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from controller_trainer import ControllerTrainer, ControllerTrainerConfig
from activation_controller_trainer import ActivationControllerTrainer

# Add eval directory to path for answer checking utilities
import sys
sys.path.insert(0, str(Path(__file__).parent / "eval"))


# =============================================================================
# Answer Extraction and Correctness Checking
# =============================================================================

def extract_answer_tags(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags in model response.
    Uses the LAST complete pair where content doesn't contain <answer>."""
    import re
    pattern = r'<answer>((?:(?!<answer>).)*?)</answer>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    answer = re.sub(r'^\\[\(\[]?\s*\\displaystyle\s*', '', answer)
    answer = re.sub(r'\s*\\[\)\]]?$', '', answer)
    return answer if answer else None


def check_answer_correctness(response: str, ground_truth: Optional[str]) -> Optional[bool]:
    """Check if the response contains the correct answer.
    
    Returns True/False if ground_truth is available, None otherwise.
    """
    if ground_truth is None:
        return None
    predicted = extract_answer_tags(response)
    if predicted is None:
        return False
    try:
        from is_equiv import is_equiv
        return is_equiv(predicted, ground_truth)
    except ImportError:
        # Fallback to simple string comparison
        return predicted.strip() == ground_truth.strip()


# =============================================================================
# Environment Configuration
# =============================================================================

OFFLINE_VARS = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "KERNELS_AUTO_OFFLINE": "1",
    "KERNELS_LOCAL_ONLY": "1",
    "KERNELS_LOCAL_REPO": "/scratch/gpfs/HENDERSON/zs7353/rl_moe/local_kernels",
}

# Wandb vars - these are forced (not setdefault) to ensure correct paths
WANDB_VARS = {
    "WANDB_MODE": "offline",
    "WANDB_PROJECT": "rl_moe_controller",
    "WANDB_DIR": "/scratch/gpfs/HENDERSON/zs7353/legacy/wandb",
    "WANDB_CACHE_DIR": "/scratch/gpfs/HENDERSON/zs7353/legacy/wandb/cache",
    "WANDB_CONFIG_DIR": "/scratch/gpfs/HENDERSON/zs7353/legacy/wandb/config",
}


def configure_offline_env() -> None:
    """Configure environment for offline operation."""
    for key, value in OFFLINE_VARS.items():
        os.environ.setdefault(key, value)
    
    # Force wandb directories to correct paths (override any existing values)
    for key, value in WANDB_VARS.items():
        os.environ[key] = value
    
    # Create wandb directories
    wandb_dir = Path(os.environ["WANDB_DIR"])
    wandb_dir.mkdir(parents=True, exist_ok=True)
    for extra_key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        extra_path = os.environ.get(extra_key)
        if extra_path:
            Path(extra_path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reward Model (KL Divergence)
# =============================================================================

class KLReward:
    """Compute on-policy distillation reward using KL divergence.
    
    Based on: https://thinkingmachines.ai/blog/on-policy-distillation/
    
    The idea: we want the controller (student) to produce token distributions
    that match the full model (teacher). We compute the FULL reverse KL divergence
    over the vocabulary at each token position:
    
    KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
    
    This is computed using the FULL distributions, not just the sampled token.
    The KL divergence is always >= 0, with 0 meaning perfect match.
    
    Reward = -KL_sum * scale
    
    Higher reward = smaller KL = controller matches full model better.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        accelerator,
        max_length: int = 2048,
        reward_scale: float = 1.0,  # Scale factor for the KL reward
        temperature: float = 1.0,  # Token sampling temperature (match generation)
    ):
        """
        Args:
            model: The gpt-oss model
            tokenizer: The tokenizer
            accelerator: The accelerator instance
            max_length: Maximum sequence length
            reward_scale: Scale factor for KL reward (default 1.0)
            temperature: Token sampling temperature. Log probs are computed under
                         the temperature-scaled distribution to match the sampling
                         distribution, ensuring the sampled KL is an unbiased estimator.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.max_length = max_length
        self.reward_scale = reward_scale
        self.temperature = temperature
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Save original router weights for frozen teacher evaluation.
        # disable_adapter() only disables LoRA, not router weight updates.
        self._original_router_weights = {}
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'policy'):
            unwrapped = unwrapped.policy
        for name, param in unwrapped.named_parameters():
            if 'router' in name:
                self._original_router_weights[name] = param.data.clone()
        if self.accelerator.is_main_process:
            print(f"[KL-REWARD] Saved {len(self._original_router_weights)} original router weight tensors for frozen teacher")
        
        # Store last batch metrics for logging
        self.last_batch_kl_mean = 0.0
        self.last_batch_kl_std = 0.0
        self.last_batch_teacher_ppl_mean = 0.0
        self.last_batch_student_ppl_mean = 0.0
        
        # Accumulators for proper averaging across sub-batches
        self._kl_sum = 0.0
        self._kl_sq_sum = 0.0
        self._teacher_ppl_sum = 0.0
        self._student_ppl_sum = 0.0
        self._sample_count = 0
    
    def _swap_router_weights(self, model, use_original: bool):
        """Swap router weights between original (frozen) and current (trained) values."""
        model_params = dict(model.named_parameters())
        for name, original_data in self._original_router_weights.items():
            if name in model_params:
                param = model_params[name]
                if use_original:
                    if not hasattr(self, '_current_router_weights'):
                        self._current_router_weights = {}
                    self._current_router_weights[name] = param.data.clone()
                    param.data.copy_(original_data)
                else:
                    if hasattr(self, '_current_router_weights') and name in self._current_router_weights:
                        param.data.copy_(self._current_router_weights[name])

    def reset_batch_stats(self):
        """Reset accumulated stats at the start of each training step."""
        self._kl_sum = 0.0
        self._kl_sq_sum = 0.0
        self._teacher_ppl_sum = 0.0
        self._student_ppl_sum = 0.0
        self._sample_count = 0
    
    def finalize_batch_stats(self):
        """Compute final means/stds from accumulated stats."""
        if self._sample_count > 0:
            self.last_batch_kl_mean = self._kl_sum / self._sample_count
            variance = (self._kl_sq_sum / self._sample_count) - (self.last_batch_kl_mean ** 2)
            self.last_batch_kl_std = max(0, variance) ** 0.5
            self.last_batch_teacher_ppl_mean = self._teacher_ppl_sum / self._sample_count
            self.last_batch_student_ppl_mean = self._student_ppl_sum / self._sample_count
    
    def _get_unwrapped_model(self):
        """Get the unwrapped model."""
        return self.accelerator.unwrap_model(self.model)
    
    def _set_controller_enabled(self, model, enabled: bool):
        """Set controller enabled state for all MoE blocks."""
        count = 0
        for module in model.modules():
            if hasattr(module, 'controller_enabled'):
                module.controller_enabled = enabled
                count += 1
        return count
    
    @torch.no_grad()
    def compute_kl_divergence(
        self, 
        prompt: str, 
        response: str, 
        debug_idx: int = -1,
        recorded_actions: dict = None,  # Expert decisions from generation for replay
        sample_input_ids: torch.Tensor = None,  # Original token IDs [1, seq_len]
        sample_left_padding: int = None,  # Left padding length
        sample_response_len: int = None,  # Response length
        query_len: int = None,  # Padded query length
        sample_student_logits: torch.Tensor = None,  # Pre-computed student logits [1, response_len, vocab]
    ) -> tuple:
        """Compute per-token KL divergence between student (controller) and teacher (full model).
        
        We compute the FULL reverse KL divergence KL(student || teacher) at each position:
        1. Use original token IDs from generation (or re-tokenize if not provided)
        2. Get logits from BOTH student (controller enabled) and teacher (controller disabled)
           - If sample_student_logits provided, skip student forward pass (faster)
        3. Compute FULL KL over the vocabulary at each position:
           KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
        4. Sum across response token positions
        
        This is always >= 0, with 0 meaning perfect match.
        
        Args:
            prompt: The prompt text (used for empty response check)
            response: The response text (used for empty response check)
            debug_idx: Debug index for logging
            recorded_actions: Expert decisions from generation phase for replay (fixes D_gen != D_reward issue)
            sample_input_ids: Original token IDs [1, seq_len] - if provided, avoids re-tokenization
            sample_left_padding: Left padding length in the query
            sample_response_len: Actual response length (excluding padding)
            query_len: Padded query length
            sample_student_logits: Pre-computed student logits from generation [1, response_len, vocab_size].
                                   If provided, skips the student forward pass entirely (2x speedup).
        
        Returns: (kl_sum, per_token_kl_tensor, debug_info_dict)
            - kl_sum: scalar sum of KL across all tokens
            - per_token_kl_tensor: [response_len] tensor of per-position KL values
            - debug_info_dict: debugging information
        """
        # Large negative reward for empty responses to prevent reward hacking
        EMPTY_RESPONSE_PENALTY = -10.0
        
        debug_info = {
            "prompt_len_tokens": 0,
            "response_len_tokens": 0,
            "kl_sum": 0.0,
            "kl_per_token": 0.0,
            "teacher_ppl": float('inf'),
            "student_ppl": float('inf'),
        }
        
        unwrapped_model = self._get_unwrapped_model()
        device = next(unwrapped_model.parameters()).device
        
        # Helper to return empty response penalty with proper tensor for Option-Critic
        def _return_empty_penalty():
            debug_info["kl_per_token"] = -EMPTY_RESPONSE_PENALTY
            debug_info["response_len_tokens"] = 0
            # Return a single-element tensor with the penalty value (UNscaled)
            # Scaling happens in score_batch() for all cases (empty and non-empty)
            # This ensures consistency: penalty * scale, just like normal KL * scale
            penalty_tensor = torch.tensor([-EMPTY_RESPONSE_PENALTY], dtype=torch.float32)
            return -EMPTY_RESPONSE_PENALTY, penalty_tensor, debug_info
        
        # Track left padding for attention mask (0 in fallback case)
        left_padding_len = 0
        
        # Use original token IDs if provided (avoids re-tokenization mismatch with recorded_actions)
        if sample_input_ids is not None and sample_left_padding is not None and sample_response_len is not None and query_len is not None:
            # Check empty response using actual token count (not string which misses EOS-only)
            if sample_response_len == 0:
                return _return_empty_penalty()
            
            # Use original tokens directly - exact alignment with recorded_actions
            input_ids = sample_input_ids.to(device)  # [1, query_len + max_response_len]
            left_padding_len = sample_left_padding  # For attention mask
            
            # IMPORTANT: input_ids contains [LEFT_PAD | QUERY | RESPONSE]
            # - query_len = total query positions (including left padding)
            # - response starts at position query_len
            # - prompt_len is just for debug info (actual query token count, excluding padding)
            response_start_pos = query_len  # Position where response tokens start
            prompt_len = query_len - sample_left_padding  # For debug info only
            response_len = sample_response_len
            
            # Truncate to actual sequence length (remove right padding in response)
            actual_seq_len = query_len + response_len
            if input_ids.shape[1] > actual_seq_len:
                input_ids = input_ids[:, :actual_seq_len]
            
            debug_info["prompt_len_tokens"] = prompt_len  # Actual query tokens (for logging)
            debug_info["response_len_tokens"] = response_len
        else:
            # Fallback: re-tokenize from strings (may cause mismatch with recorded_actions)
            # Tokenize prompt WITH special tokens (includes BOS)
            prompt_ids = self.tokenizer(
                prompt, 
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
            prompt_len = prompt_ids.shape[1]
            response_start_pos = prompt_len  # In fallback case, no padding, so response starts at prompt_len
            debug_info["prompt_len_tokens"] = prompt_len
            
            # Tokenize response WITHOUT special tokens
            response_ids = self.tokenizer(
                response,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            response_len = response_ids.shape[1]
            debug_info["response_len_tokens"] = response_len
            
            if response_len == 0:
                return _return_empty_penalty()
            
            # Concatenate token IDs
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            
            # Truncate if too long
            if input_ids.shape[1] > self.max_length:
                input_ids = input_ids[:, :self.max_length]
                response_len = min(response_len, self.max_length - prompt_len)
                response_start_pos = prompt_len  # Still same position
                if response_len <= 0:
                    return _return_empty_penalty()

            input_ids = input_ids.to(device)
        
        # Create attention mask - must match generation (left padding has mask=0)
        attention_mask = torch.ones_like(input_ids)
        if left_padding_len > 0:
            attention_mask[:, :left_padding_len] = 0
        
        # Create labels for loss computation (mask query tokens including padding)
        # response_start_pos is the position where response tokens start
        labels = input_ids.clone()
        labels[:, :response_start_pos] = -100
        
        # Check if model is a PeftModel (has LoRA adapters)
        try:
            from peft import PeftModel
            is_peft = isinstance(unwrapped_model, PeftModel)
        except ImportError:
            is_peft = False
        
        # Forward pass with TEACHER (controller disabled + LoRA disabled = ORIGINAL frozen model)
        # CRITICAL: We disable LoRA adapters to get the original pre-trained model's predictions,
        # not the fine-tuned model. This ensures the teacher is truly the "reference" distribution.
        # Use torch.no_grad() since we don't need gradients for teacher (saves memory)
        num_disabled = self._set_controller_enabled(unwrapped_model, False)
        if debug_idx == 0:
            # Verify controller is disabled: check GptOssMLP.controller_enabled
            controller_values = []
            for name, module in unwrapped_model.named_modules():
                if module.__class__.__name__ == 'GptOssMLP' and hasattr(module, 'controller_enabled'):
                    controller_values.append(module.controller_enabled)
            all_controller_disabled = all(v == False for v in controller_values)
            print(f"[KL-TEACHER] Controller: {len(controller_values)} GptOssMLP modules, all disabled={all_controller_disabled}", flush=True)
        
        with torch.no_grad():
            # Swap in original router weights for true frozen teacher
            self._swap_router_weights(unwrapped_model, use_original=True)
            if is_peft:
                with unwrapped_model.disable_adapter():
                    if debug_idx == 0:
                        lora_values = []
                        for name, module in unwrapped_model.named_modules():
                            if module.__class__.__name__ == 'ParamWrapper' and hasattr(module, '_disable_adapters'):
                                lora_values.append(module._disable_adapters)
                        all_lora_disabled = all(v == True for v in lora_values)
                        print(f"[KL-TEACHER] LoRA: {len(lora_values)} ParamWrapper modules, all _disable_adapters=True: {all_lora_disabled}", flush=True)
                    
                    teacher_outputs = unwrapped_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                if debug_idx == 0:
                    print(f"[KL-TEACHER] No PEFT model, using base model directly", flush=True)
                teacher_outputs = unwrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            # Restore trained router weights
            self._swap_router_weights(unwrapped_model, use_original=False)
            teacher_logits = teacher_outputs.logits  # [1, seq_len, vocab]
            teacher_loss = teacher_outputs.loss
        
        # CRITICAL: Re-enable controller after teacher forward pass
        # This ensures subsequent generations have controller_enabled=True
        self._set_controller_enabled(unwrapped_model, True)
        
        # Forward pass with STUDENT (controller enabled + LoRA enabled)
        # If sample_student_logits is provided, skip this forward pass entirely (2x speedup)
        if sample_student_logits is not None:
            # Use pre-computed logits from generation
            # These are raw logits (before temperature scaling) for the response tokens only
            student_logits_for_response = sample_student_logits[0, :response_len, :].to(device)  # [response_len, vocab]
            student_loss = None
            if debug_idx >= 0 and debug_idx < 2:
                print(f"[KL-REWARD] Using pre-computed student logits, shape={sample_student_logits.shape}", flush=True)
                # DEBUG: Check actual values
                print(f"[KL-DEBUG] student_logits_for_response: shape={student_logits_for_response.shape}, "
                      f"min={student_logits_for_response.min().item():.4f}, max={student_logits_for_response.max().item():.4f}, "
                      f"has_nan={torch.isnan(student_logits_for_response).any().item()}, "
                      f"has_inf={torch.isinf(student_logits_for_response).any().item()}", flush=True)
                print(f"[KL-DEBUG] teacher_logits: shape={teacher_logits.shape}, "
                      f"min={teacher_logits.min().item():.4f}, max={teacher_logits.max().item():.4f}, "
                      f"has_nan={torch.isnan(teacher_logits).any().item()}, "
                      f"has_inf={torch.isinf(teacher_logits).any().item()}", flush=True)
                print(f"[KL-DEBUG] response_start_pos={response_start_pos}, response_len={response_len}", flush=True)
                
                # ALIGNMENT CHECK: Verify student logits match generated tokens
                # If tokens were sampled from student, argmax(logits[t]) should often match token[t]
                # and log_prob(token[t]) should be relatively high
                actual_tokens = input_ids[0, response_start_pos:response_start_pos + response_len]
                student_probs = torch.nn.functional.softmax(student_logits_for_response.float(), dim=-1)
                argmax_tokens = student_logits_for_response.argmax(dim=-1)
                
                # Check first 10 tokens
                num_check = min(10, response_len)
                matches = (argmax_tokens[:num_check] == actual_tokens[:num_check]).sum().item()
                
                # Get log probs for actual tokens
                student_log_probs_check = torch.nn.functional.log_softmax(student_logits_for_response[:num_check].float(), dim=-1)
                actual_log_probs = student_log_probs_check.gather(dim=-1, index=actual_tokens[:num_check].unsqueeze(-1)).squeeze(-1)
                
                print(f"[KL-ALIGNMENT] First {num_check} tokens: argmax matches {matches}/{num_check}", flush=True)
                print(f"[KL-ALIGNMENT] Actual tokens (first 10): {actual_tokens[:num_check].tolist()}", flush=True)
                print(f"[KL-ALIGNMENT] Argmax tokens (first 10): {argmax_tokens[:num_check].tolist()}", flush=True)
                print(f"[KL-ALIGNMENT] Student log_prob of actual tokens (first 10): {actual_log_probs.tolist()}", flush=True)
                print(f"[KL-ALIGNMENT] Student log_prob mean: {actual_log_probs.mean().item():.4f} (expected: ~-2 to -4 for temp=1 sampling)", flush=True)
        else:
            # CRITICAL: Use recorded_actions as replay_actions to ensure we evaluate
            # the SAME expert decisions that were used during generation (fixes D_gen != D_reward bug)
            # Note: controller is already enabled from above
            controller_runtime = None
            if recorded_actions is not None:
                first_layer_data = next(iter(recorded_actions.values()), {})
                recorded_switches = first_layer_data.get('switches')
                if recorded_switches is not None:
                    recorded_seq_len = recorded_switches.shape[1]
                    input_seq_len = input_ids.shape[1]
                    
                    # Handle length mismatch between recorded_actions and input_ids
                    # 
                    # Common case: recorded_seq_len < input_seq_len by exactly 1
                    # This happens because response_lengths includes EOS token, but no routing
                    # decision is recorded for EOS (generation stopped before EOS was used as input).
                    # In this case, we truncate input_ids to match recorded_actions.
                    #
                    # Other case: recorded_seq_len >= input_seq_len
                    # recorded_actions may have extra positions for padding tokens.
                    # Truncate recorded_actions to match input_ids.
                    
                    if recorded_seq_len >= input_seq_len:
                        # Truncate recorded_actions to match input_ids
                        truncated_recorded_actions = {}
                        for layer_idx, layer_data in recorded_actions.items():
                            truncated_recorded_actions[layer_idx] = {}
                            for key, tensor in layer_data.items():
                                if tensor is not None and hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
                                    truncated_recorded_actions[layer_idx][key] = tensor[:, :input_seq_len]
                                else:
                                    truncated_recorded_actions[layer_idx][key] = tensor
                        controller_runtime = {"replay_actions": truncated_recorded_actions}
                        if debug_idx >= 0 and debug_idx < 2:
                            print(f"[KL-REWARD] Replay: recorded={recorded_seq_len}, input={input_seq_len}, "
                                  f"truncated to input_seq_len={input_seq_len}", flush=True)
                    else:
                        # recorded_actions is shorter - likely because EOS token has no recorded routing
                        # Truncate input_ids and adjust response_len to match recorded_actions
                        len_diff = input_seq_len - recorded_seq_len
                        if len_diff <= 2:  # Allow up to 2 tokens difference (EOS + possible padding edge case)
                            input_ids = input_ids[:, :recorded_seq_len]
                            response_len = max(0, response_len - len_diff)
                            input_seq_len = recorded_seq_len
                            # Update attention_mask to match new length
                            attention_mask = attention_mask[:, :recorded_seq_len]
                            # Update labels to match new length  
                            labels = labels[:, :recorded_seq_len]
                            controller_runtime = {"replay_actions": recorded_actions}
                            # Update debug_info to reflect truncated response length
                            debug_info["response_len_tokens"] = response_len
                            if debug_idx >= 0 and debug_idx < 2:
                                print(f"[KL-REWARD] Replay: recorded={recorded_seq_len} < input={input_seq_len+len_diff}, "
                                      f"truncated input_ids to {recorded_seq_len} (EOS token excluded from KL)", flush=True)
                        else:
                            # Large mismatch - this is a real bug
                            raise ValueError(
                                f"[KL-REWARD] FATAL: recorded_actions is much shorter than input_ids! "
                                f"recorded_seq_len={recorded_seq_len} < input_seq_len={input_seq_len} (diff={len_diff}). "
                                f"This indicates a bug in token handling."
                            )
                else:
                    # No switches found, just pass recorded_actions as-is
                    controller_runtime = {"replay_actions": recorded_actions}
            student_outputs = unwrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                controller_runtime=controller_runtime,
            )
            student_logits = student_outputs.logits  # [1, seq_len, vocab]
            student_loss = student_outputs.loss
            # Slice to get logits for response tokens only
            # logits at position (response_start_pos - 1) predicts first response token
            student_logits_for_response = student_logits[0, response_start_pos-1:response_start_pos+response_len-1, :]  # [response_len, vocab]
        
        # Compute log probabilities for response tokens
        # Shift: logits at position t predict token at position t+1
        # So for response tokens at positions [response_start_pos : response_start_pos + response_len],
        # we use logits at positions [response_start_pos - 1 : response_start_pos + response_len - 1]
        
        # Get the target token IDs (response tokens)
        # IMPORTANT: Explicit slice to response_len to avoid any ambiguity
        target_ids = input_ids[0, response_start_pos:response_start_pos + response_len]  # [response_len]
        
        # Get teacher logits for predicting response tokens
        # logits at position (response_start_pos - 1) predicts first response token
        teacher_logits_for_response = teacher_logits[0, response_start_pos-1:response_start_pos+response_len-1, :]  # [response_len, vocab]
        # student_logits_for_response is already set above (either from pre-computed or from forward pass)
        
        # DEBUG: Check sliced teacher logits
        if debug_idx >= 0 and debug_idx < 2:
            print(f"[KL-DEBUG] teacher_logits_for_response: shape={teacher_logits_for_response.shape}, "
                  f"min={teacher_logits_for_response.min().item():.4f}, max={teacher_logits_for_response.max().item():.4f}", flush=True)
        
        # Compute log softmax (over vocabulary) under temperature-scaled distribution
        # This matches the sampling distribution (tokens are sampled with temperature τ),
        # ensuring the sampled KL is an unbiased estimator of KL(p_student^τ || p_teacher^τ)
        temp = self.temperature
        # teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits_for_response.float() / temp, dim=-1)  # [response_len, vocab]
        # student_log_probs = torch.nn.functional.log_softmax(student_logits_for_response.float() / temp, dim=-1)  # [response_len, vocab]
        # keep the older scales so that we don't need to tune parameters
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits_for_response.float(), dim=-1)  # [response_len, vocab]
        student_log_probs = torch.nn.functional.log_softmax(student_logits_for_response.float(), dim=-1)  # [response_len, vocab]

        # DEBUG: Check log_probs
        if debug_idx >= 0 and debug_idx < 2:
            print(f"[KL-DEBUG] teacher_log_probs: min={teacher_log_probs.min().item():.4f}, "
                  f"has_neginf={(teacher_log_probs == float('-inf')).any().item()}", flush=True)
            print(f"[KL-DEBUG] student_log_probs: min={student_log_probs.min().item():.4f}, "
                  f"has_neginf={(student_log_probs == float('-inf')).any().item()}", flush=True)
        
        # Compute SAMPLED KL divergence (on-policy distillation approach)
        # As described in https://thinkingmachines.ai/blog/on-policy-distillation/
        # 
        # Instead of computing full KL over the entire vocabulary:
        #   KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
        # 
        # We compute the sampled KL, which only looks at the generated token:
        #   sampled_kl_t = log p_student(a_t) - log p_teacher(a_t)
        # 
        # where a_t is the token that was actually generated at position t.
        # 
        # This is an unbiased estimator of the full KL (in expectation over samples),
        # and is the standard approach used in PPO/RLHF for KL penalties.
        # 
        # The reward is then: r_t = -sampled_kl_t = log p_teacher(a_t) - log p_student(a_t)
        # Positive reward when teacher assigns higher probability than student (teacher approves)
        
        # Get the generated token IDs
        # target_ids is already exactly response_len due to explicit slice above
        generated_token_ids = target_ids.unsqueeze(-1)  # [response_len, 1]
        
        # Gather log probs for the generated tokens only
        student_log_probs_sampled = student_log_probs.gather(dim=-1, index=generated_token_ids).squeeze(-1)  # [response_len]
        teacher_log_probs_sampled = teacher_log_probs.gather(dim=-1, index=generated_token_ids).squeeze(-1)  # [response_len]
        
        # Sampled KL: log p_student(a_t) - log p_teacher(a_t)
        # Note: This can be positive or negative (unlike full KL which is always >= 0)
        # Negative sampled_kl means teacher assigns higher probability (good)
        per_token_kl = student_log_probs_sampled - teacher_log_probs_sampled  # [response_len]
        
        # DEBUG: Check per_token_kl
        if debug_idx >= 0 and debug_idx < 2:
            print(f"[KL-DEBUG] per_token_kl (sampled): shape={per_token_kl.shape}, "
                  f"min={per_token_kl.min().item():.4f}, max={per_token_kl.max().item():.4f}, "
                  f"mean={per_token_kl.mean().item():.4f}, "
                  f"has_nan={torch.isnan(per_token_kl).any().item()}", flush=True)
            print(f"[KL-DEBUG] student_log_probs_sampled: min={student_log_probs_sampled.min().item():.4f}, "
                  f"max={student_log_probs_sampled.max().item():.4f}", flush=True)
            print(f"[KL-DEBUG] teacher_log_probs_sampled: min={teacher_log_probs_sampled.min().item():.4f}, "
                  f"max={teacher_log_probs_sampled.max().item():.4f}", flush=True)
        
        # Note: Unlike full KL which is always >= 0, sampled KL can be negative
        # (when teacher likes the token more than student does)
        # Sum across token positions
        kl_sum = per_token_kl.sum().item()
        kl_per_token = kl_sum / response_len if response_len > 0 else 0.0
        
        # Compute perplexities for logging
        teacher_ppl = torch.exp(teacher_loss).item() if teacher_loss is not None else float('inf')
        student_ppl = torch.exp(student_loss).item() if student_loss is not None else float('inf')
        
        debug_info["kl_sum"] = kl_sum
        debug_info["kl_per_token"] = kl_per_token
        debug_info["teacher_ppl"] = teacher_ppl
        debug_info["student_ppl"] = student_ppl
        
        # Return per_token_kl tensor for Option-Critic (detached, moved to CPU)
        return kl_sum, per_token_kl.detach().cpu(), debug_info
    
    def score_batch(
        self, 
        prompts: List[str], 
        responses: List[str],
        recorded_actions: dict = None,  # Expert decisions from generation for replay
        input_ids: torch.Tensor = None,  # Original token IDs [batch, seq_len] - avoids re-tokenization
        left_padding_lengths: torch.Tensor = None,  # [batch] - left padding in queries
        response_lengths: torch.Tensor = None,  # [batch] - actual response lengths
        query_len: int = None,  # Padded query length
        student_logits: torch.Tensor = None,  # Pre-computed logits [batch, response_len, vocab] - skips student forward
        ground_truth_answers: Optional[List[str]] = None,  # Ground truth answers for correctness checking
    ) -> List[float]:
        """Score a batch of responses based on sampled KL divergence (on-policy distillation).
        
        Uses SAMPLED KL as described in https://thinkingmachines.ai/blog/on-policy-distillation/
        
        Per-token sampled KL: log p_student(a_t) - log p_teacher(a_t)
        - Positive: student more confident than teacher (bad - student overconfident)
        - Negative: teacher more confident than student (good - teacher approves)
        
        Reward = -kl_per_token * scale
        - Higher reward when teacher approves of the generated tokens
        
        Also stores per_token_kl tensors in self.last_batch_per_token_kl for Option-Critic.
        
        Args:
            prompts: List of prompt strings (used for empty response check)
            responses: List of response strings (used for empty response check)
            recorded_actions: Dict of expert decisions from generation phase for replay.
                Structure: {layer_idx: {'switches': tensor[batch, seq], 'selected_indices': tensor[batch, seq, k]}}
            input_ids: Original token IDs from generation [batch, query_len + response_len].
                If provided, avoids re-tokenization and ensures exact alignment with recorded_actions.
            left_padding_lengths: [batch] tensor of left padding lengths in queries
            response_lengths: [batch] tensor of actual response lengths (excluding padding)
            query_len: Padded query length (same for all samples in batch)
            student_logits: Pre-computed student logits from generation [batch, response_len, vocab_size].
                           If provided, skips the student forward pass entirely (2x speedup).
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        
        unwrapped_model = self._get_unwrapped_model()
        
        # Log whether we're using pre-computed student logits (2x speedup)
        using_precomputed = student_logits is not None
        print(f"[KL-REWARD] === Starting batch scoring (n={len(prompts)}, precomputed_logits={using_precomputed}) ===", flush=True)
        
        try:
            # Check if model is using LoRA on experts (PeftModel)
            # If so, keep in train mode because GptOssExperts.forward uses torch.bmm
            # in eval mode which breaks with LoRA-wrapped 3D parameters
            try:
                from peft import PeftModel
                is_peft = isinstance(unwrapped_model, PeftModel)
            except ImportError:
                is_peft = False
            
            was_training = unwrapped_model.training
            if is_peft:
                # Keep in train mode for LoRA compatibility
                unwrapped_model.train()
            else:
                unwrapped_model.eval()
            
            kl_values = []
            rewards = []
            debug_infos = []
            per_token_kls = []  # Store per-token KL for Option-Critic
            
            for idx, (prompt, response) in enumerate(zip(prompts, responses)):
                # Slice recorded_actions for this sample if provided
                # recorded_actions has structure: {layer_idx: {'switches': [batch, seq], 'selected_indices': [batch, seq, k]}}
                sample_recorded_actions = None
                if recorded_actions is not None:
                    sample_recorded_actions = {}
                    for layer_idx, layer_data in recorded_actions.items():
                        sample_recorded_actions[layer_idx] = {}
                        for key, tensor in layer_data.items():
                            if tensor is not None and hasattr(tensor, 'shape') and len(tensor.shape) >= 1:
                                # Slice to get single sample (keep batch dim for compatibility)
                                sample_recorded_actions[layer_idx][key] = tensor[idx:idx+1]
                            else:
                                sample_recorded_actions[layer_idx][key] = tensor
                
                # Extract sample's token IDs if provided (avoids re-tokenization)
                sample_input_ids = None
                sample_left_padding = None
                sample_response_len = None
                if input_ids is not None and left_padding_lengths is not None and response_lengths is not None:
                    sample_input_ids = input_ids[idx:idx+1]  # [1, seq_len]
                    sample_left_padding = left_padding_lengths[idx].item()
                    sample_response_len = response_lengths[idx].item()
                
                # Extract sample's pre-computed student logits if provided (skips student forward pass)
                sample_student_logits = None
                if student_logits is not None:
                    sample_student_logits = student_logits[idx:idx+1]  # [1, response_len, vocab]
                
                kl_sum, per_token_kl, debug_info = self.compute_kl_divergence(
                    prompt, response, debug_idx=idx, recorded_actions=sample_recorded_actions,
                    sample_input_ids=sample_input_ids,
                    sample_left_padding=sample_left_padding,
                    sample_response_len=sample_response_len,
                    query_len=query_len,
                    sample_student_logits=sample_student_logits,
                )
                kl_values.append(kl_sum)
                debug_infos.append(debug_info)
                
                # Scale per_token_kl for consistency with scalar reward
                # Option-Critic uses per_token_kl directly, so it must have the same scale
                if per_token_kl is not None:
                    scaled_per_token_kl = per_token_kl * self.reward_scale
                else:
                    scaled_per_token_kl = None
                per_token_kls.append(scaled_per_token_kl)
                
                # Reward = -KL_per_token * scale
                # Use per-token KL to normalize by response length
                # This ensures fair comparison between short and long responses
                kl_per_token = debug_info['kl_per_token']
                reward = -kl_per_token * self.reward_scale
                
                rewards.append(reward)
                
                # Debug print for first 3 samples in batch
                if idx < 3:
                    print(f"[KL-REWARD] Sample {idx}: "
                          f"prompt_tokens={debug_info['prompt_len_tokens']}, "
                          f"response_tokens={debug_info['response_len_tokens']}, "
                          f"kl_sum={kl_sum:.4f}, "
                          f"kl_per_token={kl_per_token:.4f}, "
                          f"teacher_ppl={debug_info['teacher_ppl']:.2f}, "
                          f"student_ppl={debug_info['student_ppl']:.2f}, "
                          f"reward={reward:.4f} (using per-token KL)", flush=True)
                    # Show full response
                    print(f"[KL-REWARD]   Full response: {response}", flush=True)
            
            # Check correctness for each sample if ground truth answers are available
            batch_correctness = None
            if ground_truth_answers is not None:
                batch_correctness = []
                for idx, (response, gt_answer) in enumerate(zip(responses, ground_truth_answers)):
                    correct = check_answer_correctness(response, gt_answer)
                    batch_correctness.append(correct if correct is not None else False)
                
                num_correct = sum(1 for c in batch_correctness if c)
                print(f"[KL-REWARD] Correctness: {num_correct}/{len(batch_correctness)} correct", flush=True)
            
            self.last_batch_correctness = batch_correctness
            
            # Summary statistics
            valid_kls = [k for k in kl_values if not math.isnan(k) and not math.isinf(k)]
            valid_kl_per_tokens = [d['kl_per_token'] for d in debug_infos 
                                   if not math.isnan(d['kl_per_token']) and not math.isinf(d['kl_per_token'])]
            
            if valid_kls:
                avg_kl_sum = sum(valid_kls) / len(valid_kls)
                avg_kl_per_token = sum(valid_kl_per_tokens) / len(valid_kl_per_tokens) if valid_kl_per_tokens else 0.0
                std_kl_per_token = (sum((k - avg_kl_per_token) ** 2 for k in valid_kl_per_tokens) / len(valid_kl_per_tokens)) ** 0.5 if valid_kl_per_tokens else 0.0
                avg_reward = sum(rewards) / len(rewards)
                
                teacher_ppls = [d['teacher_ppl'] for d in debug_infos if d['teacher_ppl'] != float('inf')]
                student_ppls = [d['student_ppl'] for d in debug_infos if d['student_ppl'] != float('inf')]
                avg_teacher_ppl = sum(teacher_ppls) / len(teacher_ppls) if teacher_ppls else float('inf')
                avg_student_ppl = sum(student_ppls) / len(student_ppls) if student_ppls else float('inf')
                
                # ACCUMULATE stats across sub-batches (for gradient accumulation)
                # Caller should call reset_batch_stats() at start of training step
                # and finalize_batch_stats() after all sub-batches
                n_valid = len(valid_kl_per_tokens)
                self._kl_sum += sum(valid_kl_per_tokens)
                self._kl_sq_sum += sum(k**2 for k in valid_kl_per_tokens)
                if teacher_ppls:
                    self._teacher_ppl_sum += sum(teacher_ppls)
                if student_ppls:
                    self._student_ppl_sum += sum(student_ppls)
                self._sample_count += n_valid
                
                # Also update the "last batch" values for this sub-batch (for debug prints)
                self.last_batch_kl_mean = avg_kl_per_token
                self.last_batch_kl_std = std_kl_per_token
                self.last_batch_teacher_ppl_mean = avg_teacher_ppl
                self.last_batch_student_ppl_mean = avg_student_ppl
                
                print(f"[KL-REWARD] Batch summary:", flush=True)
                print(f"[KL-REWARD]   KL per token: mean={avg_kl_per_token:.4f}, std={std_kl_per_token:.4f}", flush=True)
                print(f"[KL-REWARD]   KL sum (for reference): mean={avg_kl_sum:.4f}", flush=True)
                print(f"[KL-REWARD]   PPL: teacher={avg_teacher_ppl:.2f}, student={avg_student_ppl:.2f}", flush=True)
                print(f"[KL-REWARD]   Reward: mean={avg_reward:.4f} (based on per-token KL)", flush=True)
                print(f"[KL-REWARD]   Valid samples: {len(valid_kls)}/{len(kl_values)}", flush=True)
                # Print per_token_kl shapes for alignment verification
                valid_kl_shapes = [kl.shape[0] if kl is not None else 0 for kl in per_token_kls]
                print(f"[KL-REWARD]   per_token_kl lengths: {valid_kl_shapes}", flush=True)
            else:
                print(f"[KL-REWARD] WARNING: No valid KL values in batch!", flush=True)
                self.last_batch_kl_mean = 0.0
                self.last_batch_kl_std = 0.0
                self.last_batch_teacher_ppl_mean = 0.0
                self.last_batch_student_ppl_mean = 0.0
            
            # Restore training mode
            if was_training:
                unwrapped_model.train()
            
            # Store per-token KL for Option-Critic (will be attached to rollout)
            self.last_batch_per_token_kl = per_token_kls
            
            print(f"[KL-REWARD] === Batch scoring complete ===", flush=True)
            
            return rewards
            
        except Exception as e:
            import traceback
            print(f"[KL-REWARD] ERROR in score_batch: {e}", flush=True)
            traceback.print_exc()
            # Re-enable controller even on error
            self._set_controller_enabled(unwrapped_model, True)
            raise


# =============================================================================
# Data Loading
# =============================================================================

def collect_prompts_by_category(
    data_dir: Path,
    per_category: int = 100,
    max_length: int = 1024,
    tokenizer=None,
    seed: int = 42,
    categories_filter: List[str] = None,
) -> List[str]:
    """Load prompts from Nemotron dataset.
    
    The dataset can be organized in two ways:
    1. Flat: parquet files directly in data_dir (e.g., chat-00000.parquet)
    2. Nested: subdirectories per category with parquet files inside
    
    Args:
        categories_filter: If provided, only load from these categories (e.g., ['math', 'code']).
                          If None, load from all categories.
    """
    import pandas as pd
    
    rng = random.Random(seed)
    all_prompts: List[str] = []
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for flat structure (parquet files directly in data_dir)
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if parquet_files:
        # Flat structure - group by prefix (e.g., "chat", "code", "math")
        categories = {}
        for pf in parquet_files:
            # Extract category from filename like "chat-00000-of-00012.parquet"
            name = pf.stem
            parts = name.split("-")
            if parts:
                category = parts[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(pf)
        
        # Filter categories if specified
        if categories_filter is not None:
            filtered_categories = {k: v for k, v in categories.items() if k in categories_filter}
            skipped = set(categories.keys()) - set(filtered_categories.keys())
            if skipped:
                print(f"[DATA] Skipping categories (not in filter): {sorted(skipped)}")
            categories = filtered_categories
            print(f"[DATA] Loading from categories: {sorted(categories.keys())}")
        
        for category, files in sorted(categories.items()):
            category_prompts = []
            
            # Load from first file of each category
            try:
                df = pd.read_parquet(files[0])
            except Exception as e:
                print(f"[DATA] Error loading {files[0]}: {e}")
                continue
            
            # Try different column names for the prompt
            prompt_col = None
            for col in ["input", "prompt", "question", "text"]:
                if col in df.columns:
                    prompt_col = col
                    break
            
            # For Nemotron-style data with "messages" column
            if prompt_col is None and "messages" in df.columns:
                for _, row in df.iterrows():
                    msgs = row.get("messages", [])
                    # Handle both list and numpy array
                    if msgs is not None and hasattr(msgs, '__iter__'):
                        # Find first user message
                        for msg in msgs:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                content = msg.get("content", "")
                                if content and isinstance(content, str) and len(content) > 10:
                                    category_prompts.append(content)
                                    break
                    if len(category_prompts) >= per_category * 2:
                        break
            # For chat data with "conversations" column
            elif prompt_col is None and "conversations" in df.columns:
                # Extract first user message from conversations
                for _, row in df.iterrows():
                    convs = row.get("conversations", [])
                    if isinstance(convs, list) and len(convs) > 0:
                        first_msg = convs[0]
                        if isinstance(first_msg, dict):
                            content = first_msg.get("value", first_msg.get("content", ""))
                            if content:
                                category_prompts.append(content)
                    if len(category_prompts) >= per_category * 2:
                        break
            elif prompt_col is not None:
                category_prompts = df[prompt_col].dropna().tolist()
            
            if not category_prompts:
                print(f"[DATA] No prompts found in {files[0]}, columns: {list(df.columns)}")
                continue
            
            rng.shuffle(category_prompts)
            
            if len(category_prompts) > per_category * 2:
                category_prompts = category_prompts[:per_category * 2]
            
            # Filter by length
            filtered = []
            for prompt in category_prompts:
                if not isinstance(prompt, str):
                    continue
                if tokenizer is not None:
                    try:
                        length = len(tokenizer(prompt, truncation=False)["input_ids"])
                        if length < max_length:
                            filtered.append(prompt)
                    except Exception:
                        continue
                else:
                    filtered.append(prompt)
                
                if len(filtered) >= per_category:
                    break
            
            if filtered:
                print(f"[DATA] Loaded {len(filtered)} prompts from category '{category}'")
                all_prompts.extend(filtered[:per_category])
    else:
        # Nested structure - look for subdirectories
        for category_dir in sorted(data_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_parquets = list(category_dir.glob("*.parquet"))
            if not category_parquets:
                continue
            
            try:
                df = pd.read_parquet(category_parquets[0])
            except Exception as e:
                print(f"[DATA] Error loading {category_parquets[0]}: {e}")
                continue
            
            if "input" not in df.columns:
                continue
            
            category_prompts = df["input"].dropna().tolist()
            rng.shuffle(category_prompts)
            
            if len(category_prompts) > per_category * 2:
                category_prompts = category_prompts[:per_category * 2]
            
            # Filter by length
            filtered = []
            for prompt in category_prompts:
                if tokenizer is not None:
                    length = len(tokenizer(prompt, truncation=False)["input_ids"])
                    if length < max_length:
                        filtered.append(prompt)
                else:
                    filtered.append(prompt)
                
                if len(filtered) >= per_category:
                    break
            
            if filtered:
                all_prompts.extend(filtered[:per_category])
    
    if not all_prompts:
        raise RuntimeError(f"No prompts could be loaded from {data_dir}")
    
    rng.shuffle(all_prompts)
    print(f"[DATA] Total prompts loaded: {len(all_prompts)}")
    return all_prompts


def collect_mmlu_prompts(
    data_dir: Path,
    per_category: int = 50,
    max_length: int = 1024,
    tokenizer=None,
    seed: int = 42,
) -> tuple[List[str], dict]:
    """Load MMLU multiple-choice questions.
    
    Returns:
        prompts: List of formatted question prompts
        ground_truth: Dict mapping prompt -> correct answer letter (A, B, C, D)
    """
    import pandas as pd
    
    rng = random.Random(seed)
    all_prompts: List[str] = []
    ground_truth: dict = {}
    
    if not data_dir.exists():
        raise FileNotFoundError(f"MMLU data directory not found: {data_dir}")
    
    # Answer index to letter mapping
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    # Iterate over category directories
    category_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name not in ("all", "auxiliary_train")]
    
    for category_dir in category_dirs:
        # Look for test split (most questions)
        test_files = list(category_dir.glob("test-*.parquet"))
        if not test_files:
            continue
        
        try:
            df = pd.read_parquet(test_files[0])
        except Exception as e:
            print(f"[DATA] Error loading {test_files[0]}: {e}")
            continue
        
        if not all(col in df.columns for col in ["question", "choices", "answer"]):
            print(f"[DATA] Missing columns in {test_files[0]}, columns: {list(df.columns)}")
            continue
        
        # Sample up to per_category questions
        indices = list(range(len(df)))
        rng.shuffle(indices)
        
        category_count = 0
        for idx in indices:
            if category_count >= per_category:
                break
            
            row = df.iloc[idx]
            question = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            
            # Format the prompt as multiple-choice question
            # Instruct model to wrap answer in <answer></answer> tags
            prompt = (
                f"Answer the following multiple-choice question. "
                f"Think step by step, then provide your final answer wrapped in <answer></answer> tags, "
                f"e.g., <answer>A</answer>.\n\n"
                f"Question: {question}\n\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"C) {choices[2]}\n"
                f"D) {choices[3]}\n\n"
                f"Answer:"
            )
            
            # Filter by length if tokenizer provided
            if tokenizer is not None:
                try:
                    length = len(tokenizer(prompt, truncation=False)["input_ids"])
                    if length >= max_length:
                        continue
                except Exception:
                    continue
            
            correct_letter = idx_to_letter[answer_idx]
            all_prompts.append(prompt)
            ground_truth[prompt] = correct_letter
            category_count += 1
        
        if category_count > 0:
            print(f"[DATA] Loaded {category_count} questions from category '{category_dir.name}'")
    
    if not all_prompts:
        raise RuntimeError(f"No MMLU questions could be loaded from {data_dir}")
    
    # Shuffle all prompts
    combined = list(zip(all_prompts, [ground_truth[p] for p in all_prompts]))
    rng.shuffle(combined)
    all_prompts = [p for p, _ in combined]
    ground_truth = {p: gt for p, gt in combined}
    
    print(f"[DATA] Total MMLU questions loaded: {len(all_prompts)}")
    return all_prompts, ground_truth


def collect_math_prompts(
    data_dir: Path,
    prompts_per_category: int,
    max_prompt_length: int,
    tokenizer,
    seed: int = 42,
    split: str = "train",
) -> Tuple[List[str], List[str]]:
    """Collect prompts from Hendrycks MATH dataset with ground truth answers.
    
    Args:
        data_dir: Path to hendrycks_math directory
        prompts_per_category: Number of prompts per category
        max_prompt_length: Maximum prompt length in tokens
        tokenizer: Tokenizer for length filtering
        seed: Random seed
        split: Which split to use ("train" or "test")
        
    Returns:
        (prompts, answers): Lists of prompt strings and their ground truth answers
    """
    import re as _re
    import pandas as pd
    rng = random.Random(seed)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"MATH data directory not found: {data_dir}")
    
    # Get all category directories
    category_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    
    all_prompts = []
    all_answers = []
    
    for cat_dir in category_dirs:
        cat = cat_dir.name
        
        # Look for split files
        split_files = list(cat_dir.glob(f"{split}-*.parquet"))
        if not split_files:
            print(f"[DATA] Warning: No {split} file for {cat}")
            continue
        
        try:
            df = pd.read_parquet(split_files[0])
        except Exception as e:
            print(f"[DATA] Warning: Could not load {split_files[0]}: {e}")
            continue
        
        if not all(col in df.columns for col in ["problem", "solution"]):
            print(f"[DATA] Warning: Missing columns in {cat}, columns: {list(df.columns)}")
            continue
        
        # Sample random indices
        indices = list(range(len(df)))
        rng.shuffle(indices)
        
        count = 0
        for idx in indices:
            if count >= prompts_per_category:
                break
            
            row = df.iloc[idx]
            problem = row["problem"]
            solution = row["solution"]
            
            # Extract answer from \boxed{...} in solution
            boxed_match = _re.search(r'\\boxed\{', solution)
            if not boxed_match:
                continue
            start = boxed_match.end()
            brace_count = 1
            end = start
            while end < len(solution) and brace_count > 0:
                if solution[end] == '{':
                    brace_count += 1
                elif solution[end] == '}':
                    brace_count -= 1
                end += 1
            if brace_count != 0:
                continue
            answer = solution[start:end-1].strip()
            
            # Format prompt (same format as eval)
            prompt = (
                f"Solve the following math problem. Show your reasoning step by step.\n\n"
                f"CRITICAL: You MUST wrap your final answer in <answer></answer> tags. "
                f"Do NOT just write \"The answer is X\" - you must write \"The answer is <answer>X</answer>\".\n\n"
                f"Example of correct format: \"Therefore, the answer is <answer>42</answer>.\"\n\n"
                f"Problem: {problem}\n\n"
                f"Solution (remember to use <answer></answer> tags for your final answer):"
            )
            
            # Filter by token length
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > max_prompt_length:
                continue
            
            all_prompts.append(prompt)
            all_answers.append(answer)
            count += 1
        
        print(f"[DATA] Loaded {count} prompts from category '{cat}'")
    
    # Shuffle together
    combined = list(zip(all_prompts, all_answers))
    rng.shuffle(combined)
    all_prompts = [p for p, _ in combined]
    all_answers = [a for _, a in combined]
    
    print(f"[DATA] Total MATH prompts loaded: {len(all_prompts)}")
    return all_prompts, all_answers


def tokenize_prompts(tokenizer, prompts: List[str], ground_truth_answers: Optional[List[str]] = None) -> Dataset:
    """Tokenize prompts into a dataset (no padding - pad per batch later).
    
    Args:
        tokenizer: The tokenizer
        prompts: List of prompt strings
        ground_truth_answers: Optional list of ground truth answer strings (for correctness checking)
    """
    # Apply chat template if available (important for instruction-tuned models)
    # This wraps prompts in proper format like <|start|>user<|message|>...<|end|><|start|>assistant
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            # apply_chat_template returns token ids or string depending on tokenize arg
            formatted = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False,  # Return string, we'll tokenize below
            )
            formatted_prompts.append(formatted)
        prompts = formatted_prompts
    
    encoded = tokenizer(
        prompts,
        padding=False,  # Don't pad globally - pad per batch in collate_fn
        truncation=False,
        return_tensors=None,
    )
    data_dict = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    # Store ground truth answers as strings (will be carried through the dataloader)
    if ground_truth_answers is not None:
        data_dict["ground_truth_answer"] = ground_truth_answers
    return Dataset.from_dict(data_dict)


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, pad_token_id: int = 0, num_rollouts_per_prompt: int = 1) -> DataLoader:
    """Create a DataLoader from a dataset with per-batch padding.
    
    Args:
        dataset: HuggingFace Dataset with input_ids, attention_mask, and optionally ground_truth_answer
        batch_size: Total batch size (number of rollouts per batch)
        shuffle: Whether to shuffle the dataset
        pad_token_id: Token ID for left padding
        num_rollouts_per_prompt: Number of times to repeat each prompt in a batch.
            E.g., batch_size=4, num_rollouts_per_prompt=4 -> 1 unique prompt repeated 4 times.
            Each rollout produces a different response due to stochastic sampling.
    """
    # DataLoader samples batch_size // num_rollouts_per_prompt unique items per batch.
    # The collate_fn then repeats each item num_rollouts_per_prompt times to get
    # the full batch_size. This way the dataset iterator only advances by the number
    # of unique items, not the total rollout count.
    dl_batch_size = batch_size // num_rollouts_per_prompt
    
    def collate_fn(batch):
        # Repeat each unique item num_rollouts_per_prompt times
        if num_rollouts_per_prompt > 1:
            expanded = []
            for item in batch:
                for _ in range(num_rollouts_per_prompt):
                    expanded.append(item)
            batch = expanded
        
        # Get the max length in this batch
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Pad each item to max_len (left padding for causal LM)
        padded_input_ids = []
        padded_attention_mask = []
        for item in batch:
            input_ids = item["input_ids"]
            attn_mask = item["attention_mask"]
            pad_len = max_len - len(input_ids)
            
            # Left padding
            padded_input_ids.append([pad_token_id] * pad_len + input_ids)
            padded_attention_mask.append([0] * pad_len + attn_mask)
        
        result = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }
        
        # Carry ground truth answers through if present
        if "ground_truth_answer" in batch[0]:
            result["ground_truth_answers"] = [item["ground_truth_answer"] for item in batch]
        
        return result
    
    return DataLoader(
        dataset,
        batch_size=dl_batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# =============================================================================
# Model Setup
# =============================================================================

def freeze_non_controller(module: nn.Module, keep_lora: bool = False, keep_router: bool = False) -> None:
    """Freeze all parameters except controller (and optionally LoRA/router).
    
    Args:
        module: The model to freeze
        keep_lora: If True, keep LoRA adapter parameters trainable
        keep_router: If True, keep router parameters trainable
    """
    lora_count = 0
    router_count = 0
    controller_count = 0
    frozen_count = 0
    
    for name, param in module.named_parameters():
        if "controller" in name:
            param.requires_grad = True
            controller_count += 1
        elif keep_lora and "lora" in name.lower():
            # Match any LoRA parameter (lora_A, lora_B, lora_embedding, etc.)
            param.requires_grad = True
            lora_count += 1
        elif keep_router and "router" in name:
            param.requires_grad = True
            router_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    # Debug output
    print(f"[FREEZE] Controller: {controller_count}, LoRA: {lora_count}, Router: {router_count}, Frozen: {frozen_count}")


def print_trainable_params(model: nn.Module, name: str = "Model") -> None:
    """Print trainable parameter statistics."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Controller RL Training")
    
    # Model
    parser.add_argument("--model-path", type=Path, default=Path("/scratch/gpfs/KOROLOVA/gpt-oss-20b"))
    parser.add_argument("--controller-allowed-experts", type=int, default=16)
    parser.add_argument("--controller-hidden-dim", type=int, default=None,
                        help="Hidden dimension for controller RNN. Default: 4 * num_experts")
    parser.add_argument("--controller-layer-norm", type=int, default=1, choices=[0, 1],
                        help="Use LayerNorm on GRU hidden state. 1=on (default), 0=off")
    parser.add_argument("--controller-input-type", type=str, default="router_softmax",
                        choices=["router_softmax", "hidden_states"],
                        help="Controller input type: 'router_softmax' (default, 2*num_experts dims) or "
                             "'hidden_states' (hidden_dim + num_experts dims, richer but more compute)")
    parser.add_argument("--controller-expert-embed-dim", type=int, default=32,
                        help="DeepSets embedding dimension for expert set encoding in Q_U head")
    parser.add_argument("--controller-type", type=str, default="rnn", choices=["rnn", "activation"],
                        help="Controller type: 'rnn' (default, GRU-based) or 'activation' (uses LLM hidden states directly)")
    parser.add_argument("--activation-controller-mlp-hidden", type=int, default=512,
                        help="Hidden dimension for activation controller MLPs (termination and Q heads)")
    parser.add_argument("--joint-option", type=int, default=0,
                        help="Use joint (shared) option across all layers: 0=per-layer (default), 1=joint option")
    parser.add_argument("--joint-set-embed-dim", type=int, default=3072,
                        help="DeepSets output dimension for joint expert set embedding (joint option mode)")
    parser.add_argument("--joint-controller-mlp-hidden", type=int, default=4096,
                        help="MLP hidden dim for joint controller termination and Q heads")
    
    # Data
    parser.add_argument("--data-dir", type=Path, 
                        default=Path("/scratch/gpfs/HENDERSON/zs7353/mmlu"))
    parser.add_argument("--dataset-type", type=str, default="mmlu", choices=["mmlu", "nemotron", "math"],
                        help="Dataset type: 'mmlu' for MMLU, 'nemotron' for Nemotron, 'math' for Hendrycks MATH")
    parser.add_argument("--data-categories", type=str, default=None,
                        help="Comma-separated list of categories to load (e.g., 'math,code'). "
                             "If None, load all categories. Only applies to nemotron dataset.")
    parser.add_argument("--math-data-dir", type=Path,
                        default=Path("/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math"),
                        help="Path to Hendrycks MATH dataset (only used when dataset-type=math)")
    parser.add_argument("--math-split", type=str, default="train", choices=["train", "test"],
                        help="Which split to use for MATH dataset (default: train)")
    parser.add_argument("--correctness-reward-alpha", type=float, default=0.0,
                        help="Per-token reward bonus for correct trajectories (0=disabled). "
                             "Applied uniformly across all response tokens when the answer is correct.")
    parser.add_argument("--prompts-per-category", type=int, default=50,
                        help="Number of prompts per category (default 50 for MMLU, 100 for Nemotron)")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--num-rollouts-per-prompt", type=int, default=1,
                        help="Number of independent rollouts per prompt in each batch. "
                             "Reduces unique prompts per batch by this factor. "
                             "E.g., 4 with batch_size=4 means 1 unique prompt x 4 rollouts.")
    
    # Training
    parser.add_argument("--output-dir", type=Path, default=Path("./controller_rl_standalone"))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--num-update-epochs", type=int, default=1)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--token-temperature", type=float, default=0.7,
                        help="Temperature for token sampling (default 0.7). "
                             "Lower = more deterministic, higher = more random. "
                             "This is separate from controller_sampling_temperature for expert selection.")
    
    # Loss
    parser.add_argument("--value-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="Entropy bonus coefficient (0 = disabled, try 0.01-0.1 to encourage exploration)")
    parser.add_argument("--latency-cost", type=float, default=10.0)
    
    # Initialization
    parser.add_argument("--switch-init-bias", type=float, default=0.0,
                        help="Initial bias for switch head (negative = less switching, e.g. -3 for ~5%%, -4 for ~2%%)")
    
    # Reward Function (KL divergence)
    parser.add_argument("--kl-reward-scale", type=float, default=1.0,
                        help="KL reward scale: reward = -kl_per_token * scale (default 1.0)")
    
    # Option-Critic parameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for TD targets (default 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for bias-variance tradeoff (0=TD(0), 1=MC, default 0.95)")
    parser.add_argument("--deliberation-cost", type=float, default=0.1,
                        help="Deliberation cost η per switch (default 0.1)")
    
    # Intra-option policy update (Harb et al. 2017, Algorithm 1)
    # Enables updating router + experts via policy gradient
    parser.add_argument("--intra-option-update", type=int, default=1, choices=[0, 1],
                        help="Enable intra-option policy gradient on LLM (1=on, 0=off, default 1)")
    parser.add_argument("--intra-option-lr", type=float, default=1e-6,
                        help="Learning rate for LLM (LoRA + router) parameters (default 1e-6)")
    parser.add_argument("--intra-option-warmup-steps", type=int, default=0,
                        help="Skip intra-option LLM updates for first N steps (let value function warm up). default 0")
    parser.add_argument("--intra-option-q-baseline", type=int, default=0, choices=[0, 1],
                        help="Use Q(s,o) as baseline for intra-option policy gradient (A2OC: G-Q). 0=off (raw G), 1=on. default 0")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank for expert adapters (default 8)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha scaling factor (default 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout (default 0.0)")
    
    # Exploration: ε-greedy mixture for expert selection (with optional annealing)
    parser.add_argument("--selection-epsilon", type=float, default=0.0,
                        help="Fixed ε for mixed policy (if annealing not used). "
                             "Ignored if --selection-epsilon-start is set. (default 0.0)")
    parser.add_argument("--selection-epsilon-start", type=float, default=None,
                        help="Starting ε for annealing schedule. If set, enables annealing. (default: None)")
    parser.add_argument("--selection-epsilon-end", type=float, default=0.05,
                        help="Final ε after annealing. (default: 0.05)")
    parser.add_argument("--selection-epsilon-anneal-steps", type=int, default=200,
                        help="Number of steps to anneal ε from start to end. (default: 200)")
    
    # Termination advantage normalization
    parser.add_argument("--term-adv-rms-norm", type=int, default=0,
                        help="Apply RMS normalization to termination advantages. "
                             "Helps when advantage variance collapses during training. (0=off, 1=on)")
    
    # Repetition penalty (distance-based)
    parser.add_argument("--repetition-penalty-c", type=float, default=0.0,
                        help="c for repetition penalty: penalty = c * λ^d where d is distance to previous occurrence. "
                             "Use negative value for penalty (e.g., -1.0). 0 = disabled.")
    parser.add_argument("--repetition-penalty-decay", type=float, default=0.9,
                        help="λ decay factor for repetition penalty (0 < λ < 1). "
                             "Smaller λ = faster decay = less penalty for distant repeats. (default: 0.9)")
    
    # Termination TopK regularization
    parser.add_argument("--term-topk-lambda", type=float, default=0.0,
                        help="λ for TopK termination regularization: loss = λ * (1 - mean(TopK β))². "
                             "Forces some termination probs to be high. 0 = disabled.")
    parser.add_argument("--term-topk-k", type=int, default=1000,
                        help="K for TopK termination regularization. Number of top termination probs to average. (default: 1000)")
    
    # Q-based expert selection (alternative to Plackett-Luce)
    parser.add_argument("--q-based-selection", type=int, default=0,
                        help="Use Q-based selection instead of Plackett-Luce. "
                             "Selects options via argmax Q, no selection policy gradient. (0=PL, 1=Q-based)")
    parser.add_argument("--q-selection-steps", type=int, default=10,
                        help="Gradient ascent steps for Q-based selection (default: 10)")
    parser.add_argument("--q-selection-lr", type=float, default=1.0,
                        help="Learning rate for Q-based selection optimization (default: 1.0)")
    parser.add_argument("--q-selection-epsilon", type=float, default=0.1,
                        help="Fixed ε for Q-based selection (if annealing not used). "
                             "Ignored if --q-selection-epsilon-start is set. (default: 0.1)")
    parser.add_argument("--q-selection-epsilon-start", type=float, default=None,
                        help="Starting ε for Q-based selection annealing. If set, enables annealing. (default: None)")
    parser.add_argument("--q-selection-epsilon-end", type=float, default=0.05,
                        help="Final ε for Q-based selection after annealing. (default: 0.05)")
    parser.add_argument("--q-selection-epsilon-anneal-steps", type=int, default=200,
                        help="Steps to anneal Q-based selection ε from start to end. (default: 200)")
    parser.add_argument("--q-selection-debug", type=int, default=0,
                        help="Print debug info for Q-based selection (0=off, 1=on)")
    parser.add_argument("--q-selection-init-w", type=float, default=2.0,
                        help="Initial weight for current experts in Q-based selection (default: 2.0). Higher = more concentrated on current option initially.")
    
    # Teacher-mixed sampling (MiniLLM-style)
    parser.add_argument("--teacher-mix-alpha", type=float, default=0.0,
                        help="α for teacher-mixed sampling: p_mixed = α*p_teacher + (1-α)*p_student. "
                             "Set to 0.2 for MiniLLM-style teacher mixing. "
                             "Helps prevent reward hacking by mixing in teacher distribution during rollouts. "
                             "Reference: https://arxiv.org/pdf/2306.08543 (default: 0.0 = disabled)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    
    # Resume from checkpoint
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint file (.pt) to resume training from. Can be 'latest' to load controller_latest.pt from output_dir")
    
    return parser.parse_args()


def main() -> None:
    configure_offline_env()
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    
    if accelerator.is_main_process:
        print("=" * 60)
        print("Standalone Controller Training")
        print("=" * 60)
    
    # Using single GPU with device_map="auto" following the OpenAI cookbook pattern
    # https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts (and optionally ground truth answers for correctness checking)
    ground_truth_answers = None  # List[Optional[str]], set for datasets with verifiable answers
    
    if accelerator.is_main_process:
        print(f"[DATA] Loading prompts from {args.data_dir} (dataset_type={args.dataset_type})")
    
    if args.dataset_type == "mmlu":
        prompts, mmlu_ground_truth = collect_mmlu_prompts(
            args.data_dir,
            args.prompts_per_category,
            args.max_prompt_length,
            tokenizer,
            seed=args.seed,
        )
        # Convert dict {prompt: answer_letter} to list aligned with prompts
        ground_truth_answers = [mmlu_ground_truth.get(p) for p in prompts]
    elif args.dataset_type == "math":
        if accelerator.is_main_process:
            print(f"[DATA] Loading MATH prompts from {args.math_data_dir} (split={args.math_split})")
        prompts, ground_truth_answers = collect_math_prompts(
            args.math_data_dir,
            args.prompts_per_category,
            args.max_prompt_length,
            tokenizer,
            seed=args.seed,
            split=args.math_split,
        )
        if accelerator.is_main_process:
            print(f"[DATA] MATH: {len(prompts)} prompts with ground truth answers")
    else:
        # Parse categories filter if specified
        categories_filter = None
        if args.data_categories:
            categories_filter = [c.strip() for c in args.data_categories.split(",")]
            if accelerator.is_main_process:
                print(f"[DATA] Filtering to categories: {categories_filter}")
        
        prompts = collect_prompts_by_category(
            args.data_dir,
            args.prompts_per_category,
            args.max_prompt_length,
            tokenizer,
            seed=args.seed,
            categories_filter=categories_filter,
        )
    
    if accelerator.is_main_process:
        print(f"[DATA] Loaded {len(prompts)} prompts")
    
    # Create dataset and dataloader
    effective_batch_size = args.per_device_train_batch
    num_rollouts = args.num_rollouts_per_prompt
    
    # Validate num_rollouts_per_prompt
    if effective_batch_size % num_rollouts != 0:
        raise ValueError(
            f"per_device_train_batch ({effective_batch_size}) must be divisible by "
            f"num_rollouts_per_prompt ({num_rollouts})"
        )
    if num_rollouts > 1 and accelerator.is_main_process:
        unique_per_batch = effective_batch_size // num_rollouts
        print(f"[DATA] Multi-rollout: {unique_per_batch} unique prompts x {num_rollouts} rollouts = {effective_batch_size} per batch")
    
    train_dataset = tokenize_prompts(tokenizer, prompts, ground_truth_answers=ground_truth_answers)
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        pad_token_id=tokenizer.pad_token_id,
        num_rollouts_per_prompt=num_rollouts,
    )
    
    # Load model config
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config.controller_enabled = True
    config.controller_allowed_experts = args.controller_allowed_experts
    
    # Handle both gpt-oss (num_local_experts) and qwen3_moe (num_experts)
    num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", 32)
    
    # Controller hidden dim: use argument if provided, else default to 4 * num_experts
    if args.controller_hidden_dim is not None:
        config.controller_hidden_dim = args.controller_hidden_dim
    else:
        config.controller_hidden_dim = 4 * num_experts
    # Controller LayerNorm: prevents gradient explosion during BPTT
    config.controller_layer_norm = bool(args.controller_layer_norm)
    # Controller input type: router_softmax (default) or hidden_states (richer but more compute)
    config.controller_input_type = args.controller_input_type
    # Controller expert embedding dim: DeepSets embedding for Q_U head
    config.controller_expert_embed_dim = args.controller_expert_embed_dim
    # Activation controller MLP hidden dim (for termination and Q heads)
    config.activation_controller_mlp_hidden = args.activation_controller_mlp_hidden
    # Controller type: "rnn" (GRU-based) or "activation" (uses LLM hidden states directly)
    config.controller_type = args.controller_type
    # Joint option: single shared option across all layers (only for activation controller)
    config.joint_option = bool(args.joint_option)
    config.joint_set_embed_dim = args.joint_set_embed_dim
    config.joint_controller_mlp_hidden = args.joint_controller_mlp_hidden
    
    if accelerator.is_main_process:
        print(f"[CONFIG] model_type = {config.model_type}")
        print(f"[CONFIG] num_experts = {num_experts}")
        print(f"[CONFIG] controller_hidden_dim = {config.controller_hidden_dim}")
        print(f"[CONFIG] controller_layer_norm = {config.controller_layer_norm}")
        print(f"[CONFIG] controller_input_type = {config.controller_input_type}")
        print(f"[CONFIG] controller_expert_embed_dim = {config.controller_expert_embed_dim}")
        print(f"[CONFIG] activation_controller_mlp_hidden = {config.activation_controller_mlp_hidden}")
        print(f"[CONFIG] controller_type = {config.controller_type}")
        print(f"[CONFIG] joint_option = {config.joint_option}")
        print(f"[CONFIG] joint_set_embed_dim = {config.joint_set_embed_dim}")
        print(f"[CONFIG] joint_controller_mlp_hidden = {config.joint_controller_mlp_hidden}")
    
    # Load model
    if accelerator.is_main_process:
        print(f"[MODEL] Loading from {args.model_path}")
    
    # Load model in bfloat16 following OpenAI cookbook pattern
    # https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
    # Controller computations will be done in float32 separately in controller_trainer.py
    config.use_cache = False  # Disable KV cache for training
    model_kwargs = dict(
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    
    # device_map="auto" is incompatible with DDP - only use for single GPU
    if accelerator.num_processes == 1:
        model_kwargs["device_map"] = "auto"
    else:
        # Multi-GPU: Let accelerator handle device placement
        # Model will be moved to correct device after prepare()
        if accelerator.is_main_process:
            print(f"[MODEL] Multi-GPU mode ({accelerator.num_processes} GPUs): skipping device_map='auto'")
    
    # For intra-option policy gradient, we need to dequantize expert weights to enable LoRA
    # Following OpenAI cookbook: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
    if args.intra_option_update:
        try:
            from transformers import Mxfp4Config
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            if accelerator.is_main_process:
                print(f"[MODEL] Using Mxfp4Config(dequantize=True) for intra-option LoRA training")
        except ImportError as e:
            if accelerator.is_main_process:
                print(f"[MODEL] WARNING: Could not import Mxfp4Config: {e}")
                print(f"[MODEL] Trying alternative import from transformers.integrations.mxfp4...")
            try:
                from transformers.integrations.mxfp4 import Mxfp4Config
                model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                if accelerator.is_main_process:
                    print(f"[MODEL] Using Mxfp4Config(dequantize=True) for intra-option LoRA training")
            except ImportError as e2:
                if accelerator.is_main_process:
                    print(f"[MODEL] ERROR: Could not import Mxfp4Config from any location: {e2}")
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # Apply LoRA for intra-option policy gradient
    # Following OpenAI cookbook: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
    if args.intra_option_update:
        from peft import LoraConfig, get_peft_model
        
        # Determine number of layers
        num_layers = 24  # GPT-OSS-20B default
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        
        # Build target_parameters for ALL layers (as user requires)
        target_parameters = []
        for layer_idx in range(num_layers):
            target_parameters.append(f"{layer_idx}.mlp.experts.gate_up_proj")
            target_parameters.append(f"{layer_idx}.mlp.experts.down_proj")
        
        if accelerator.is_main_process:
            print(f"[LORA] Targeting {num_layers} layers for expert LoRA")
            print(f"[LORA] target_parameters (first 4): {target_parameters[:4]}...")
        
        # Apply LoRA only to expert parameters (NOT attention layers per user request)
        # target_modules=[] means no nn.Linear targets, only target_parameters for experts
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=target_parameters,  # Expert parameters for all 24 layers
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        try:
            peft_model = get_peft_model(model, lora_config)
            if accelerator.is_main_process:
                print(f"[LORA] Successfully applied LoRA to model")
                peft_model.print_trainable_parameters()
            
            # Manually make router parameters trainable (GptOssTopKRouter is custom, not nn.Linear)
            # Also convert to float32 - bfloat16 has insufficient precision for small gradient updates
            # (with lr=1e-6 and grad~0.06, update is ~6e-8 which rounds to 0 in bfloat16)
            router_count = 0
            for name, param in peft_model.named_parameters():
                if '.router.' in name or 'router.weight' in name or 'router.bias' in name:
                    param.requires_grad = True
                    # Convert to float32 for precision
                    param.data = param.data.float()
                    router_count += 1
            if accelerator.is_main_process:
                print(f"[LORA] Made {router_count} router parameters trainable (converted to float32)")
            
            # Use the PEFT-wrapped model for training
            model = peft_model
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[LORA] ERROR: Could not apply LoRA: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"LoRA setup failed: {e}")
    
    # Enable gradient checkpointing to reduce memory usage during intra-option backward pass
    # This trades compute for memory by recomputing activations during backward
    # Following TRL's pattern in enable_gradient_checkpointing()
    if args.intra_option_update:
        from peft import PeftModel
        
        # For PEFT models, enable gradient checkpointing on the BASE model (TRL pattern)
        if isinstance(model, PeftModel):
            gc_target = model.base_model
            if accelerator.is_main_process:
                print(f"[MODEL] Enabling gradient checkpointing on PEFT base_model")
        else:
            gc_target = model
        
        if hasattr(gc_target, 'gradient_checkpointing_enable'):
            # use_reentrant=False ensures proper gradient flow with PEFT
            gc_target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if accelerator.is_main_process:
                print(f"[MODEL] Gradient checkpointing enabled (use_reentrant=False) for intra-option memory efficiency")
            
            # CRITICAL: Even with use_reentrant=False, we need embedding outputs to have requires_grad=True
            # so that the computation graph properly connects to the trainable LoRA/router parameters.
            # This adds a hook to make embedding outputs require gradients.
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
                if accelerator.is_main_process:
                    print(f"[MODEL] Enabled input_require_grads for proper gradient flow")
            else:
                # Fallback: manually add the hook
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                if accelerator.is_main_process:
                    print(f"[MODEL] Added manual hook to make embedding outputs require gradients")
        else:
            if accelerator.is_main_process:
                print(f"[MODEL] WARNING: Model does not support gradient checkpointing")
    
    # CRITICAL: Reinitialize activation controller's selection_head from pretrained router weights
    # The controller is created during __init__ BEFORE state dict is loaded, so selection_head
    # has random weights. We must copy the now-pretrained router weights to selection_head.
    if args.controller_type == "activation":
        reinit_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'controller') and module.controller is not None:
                if hasattr(module.controller, 'selection_head'):
                    with torch.no_grad():
                        # Copy pretrained router weights to selection_head
                        module.controller.selection_head.weight.copy_(module.router.weight.data)
                        if module.router.bias is not None and module.controller.selection_head.bias is not None:
                            module.controller.selection_head.bias.copy_(module.router.bias.data)
                        reinit_count += 1
        if accelerator.is_main_process:
            print(f"[INIT] Reinitialized selection_head from pretrained router weights for {reinit_count} layers")
    
    # Freeze non-controller parameters (keep LoRA/router if intra-option enabled)
    use_intra_option = bool(args.intra_option_update)
    freeze_non_controller(model, keep_lora=use_intra_option, keep_router=use_intra_option)
    
    if accelerator.is_main_process:
        print_trainable_params(model, "Policy Model")
    
    # Create KL reward function
    if accelerator.is_main_process:
        print(f"[REWARD] Using KL divergence reward (scale={args.kl_reward_scale})")
    
    kl_scorer = KLReward(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        max_length=args.response_length + args.max_prompt_length,
        reward_scale=args.kl_reward_scale,
        temperature=args.token_temperature,
    )
    reward_fn = lambda p, r, recorded_actions=None, input_ids=None, left_padding_lengths=None, response_lengths=None, query_len=None, student_logits=None, ground_truth_answers=None: torch.tensor(
        kl_scorer.score_batch(
            p, r, 
            recorded_actions=recorded_actions,
            input_ids=input_ids,
            left_padding_lengths=left_padding_lengths,
            response_lengths=response_lengths,
            query_len=query_len,
            student_logits=student_logits,
            ground_truth_answers=ground_truth_answers,
        ), dtype=torch.float32
    )
    reward_scorer = kl_scorer
    
    # Create trainer config
    trainer_config = ControllerTrainerConfig(
        output_dir=str(args.output_dir),
        run_name=f"controller_rl_{int(time.time())}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_train_epochs,
        num_update_epochs=args.num_update_epochs,
        response_length=args.response_length,
        temperature=args.token_temperature,  # Token sampling temperature (separate from expert selection)
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        latency_cost_per_switch=args.latency_cost,
        option_critic_deliberation_cost=args.deliberation_cost,
        switch_init_bias=args.switch_init_bias,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        # Intra-option policy update (Harb et al. 2017)
        intra_option_update=bool(args.intra_option_update),
        intra_option_lr=args.intra_option_lr,
        intra_option_warmup_steps=args.intra_option_warmup_steps,
        intra_option_q_baseline=bool(args.intra_option_q_baseline),
        correctness_reward_alpha=args.correctness_reward_alpha,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Exploration: ε-greedy mixture for expert selection (with optional annealing)
        selection_epsilon=args.selection_epsilon,
        selection_epsilon_start=args.selection_epsilon_start,
        selection_epsilon_end=args.selection_epsilon_end,
        selection_epsilon_anneal_steps=args.selection_epsilon_anneal_steps,
        # Termination advantage normalization
        term_adv_rms_norm=bool(args.term_adv_rms_norm),
        # Repetition penalty
        repetition_penalty_c=args.repetition_penalty_c,
        repetition_penalty_decay=args.repetition_penalty_decay,
        # Termination TopK regularization
        term_topk_lambda=args.term_topk_lambda,
        term_topk_k=args.term_topk_k,
        # Q-based selection
        q_based_selection=bool(args.q_based_selection),
        q_selection_steps=args.q_selection_steps,
        q_selection_lr=args.q_selection_lr,
        q_selection_epsilon=args.q_selection_epsilon,
        q_selection_epsilon_start=args.q_selection_epsilon_start,
        q_selection_epsilon_end=args.q_selection_epsilon_end,
        q_selection_epsilon_anneal_steps=args.q_selection_epsilon_anneal_steps,
        q_selection_debug=bool(args.q_selection_debug),
        q_selection_init_w=args.q_selection_init_w,
        # Teacher-mixed sampling
        teacher_mix_alpha=args.teacher_mix_alpha,
        # Joint option
        joint_option=bool(args.joint_option),
        joint_set_embed_dim=args.joint_set_embed_dim,
        joint_controller_mlp_hidden=args.joint_controller_mlp_hidden,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    
    # Create trainer
    if args.controller_type == "activation":
        if accelerator.is_main_process:
            print("\n[TRAINER] Using ActivationControllerTrainer (LLM hidden states)")
        trainer = ActivationControllerTrainer(
            config=trainer_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            reward_fn=reward_fn,
            accelerator=accelerator,
            ppl_scorer=reward_scorer,
        )
    else:
        if accelerator.is_main_process:
            print("\n[TRAINER] Using ControllerTrainer (RNN-based)")
        trainer = ControllerTrainer(
            config=trainer_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            reward_fn=reward_fn,
            accelerator=accelerator,
            ppl_scorer=reward_scorer,  # Can be PerplexityReward or KLReward
        )
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        # Handle 'latest' as a special value
        if checkpoint_path.lower() == "latest":
            checkpoint_path = args.output_dir / "controller_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True)
        else:
            if accelerator.is_main_process:
                print(f"[WARN] Checkpoint not found: {checkpoint_path}, starting from scratch")
    
    # Train
    trainer.train()
    
    if accelerator.is_main_process:
        print("Training completed!")


if __name__ == "__main__":
    main()

