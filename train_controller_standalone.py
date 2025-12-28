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
from typing import List

import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from controller_trainer import ControllerTrainer, ControllerTrainerConfig
from activation_controller_trainer import ActivationControllerTrainer


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
    
    os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
    
    # Create wandb directories
    wandb_dir = Path(os.environ["WANDB_DIR"])
    wandb_dir.mkdir(parents=True, exist_ok=True)
    for extra_key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        extra_path = os.environ.get(extra_key)
        if extra_path:
            Path(extra_path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reward Model
# =============================================================================

class PerplexityReward:
    """Compute perplexity-based reward using gpt-oss with full expert selection.
    
    The idea: we want the controller to select experts such that the generated text
    matches what the FULL model (all experts) would produce. Lower perplexity under
    the full model = better expert selection = higher reward.
    
    This class:
    1. Temporarily disables the controller to use all experts
    2. Computes conditional perplexity: P(response | prompt)
    3. Converts perplexity to a reward in [0, 2] range
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        accelerator,
        max_length: int = 2048,
        reward_scale: float = 3.0,  # Controls sensitivity: higher = more lenient
        reward_max: float = 2.0,    # Maximum reward value
        repetition_penalty: float = 1.0,  # Additive penalty for repetition
    ):
        """
        Args:
            model: The gpt-oss model (may be wrapped by accelerator/deepspeed)
            tokenizer: The tokenizer
            accelerator: The accelerator instance
            max_length: Maximum sequence length for perplexity computation
            reward_scale: Divisor for log-perplexity in reward computation
                         reward = reward_max - log(ppl) / reward_scale
                         Default 3.0 gives: ppl=20 -> reward≈1.0, ppl=400 -> reward≈0.0
            reward_max: Maximum reward value (default 2.0 for compatibility with MMLU scale)
            repetition_penalty: Additive penalty for repetition
                         repetition_rate = 1 - unique_tokens / total_tokens
                         final_reward = ppl_reward - repetition_penalty * repetition_rate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.max_length = max_length
        self.reward_scale = reward_scale
        self.reward_max = reward_max
        self.repetition_penalty = repetition_penalty
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Store accumulated metrics for logging (log-perplexity, not raw perplexity)
        # These accumulate across gradient accumulation sub-batches
        self.last_batch_log_ppl_mean = 0.0
        self.last_batch_log_ppl_std = 0.0
        self.last_batch_repetition_rate_mean = 0.0
        self.last_batch_repetition_rate_std = 0.0
        
        # Accumulators for proper averaging across sub-batches
        self._log_ppl_sum = 0.0
        self._log_ppl_sq_sum = 0.0
        self._rep_rate_sum = 0.0
        self._rep_rate_sq_sum = 0.0
        self._sample_count = 0
    
    def reset_batch_stats(self):
        """Reset accumulated stats at the start of each training step."""
        self._log_ppl_sum = 0.0
        self._log_ppl_sq_sum = 0.0
        self._rep_rate_sum = 0.0
        self._rep_rate_sq_sum = 0.0
        self._sample_count = 0
    
    def finalize_batch_stats(self):
        """Compute final means/stds from accumulated stats."""
        if self._sample_count > 0:
            self.last_batch_log_ppl_mean = self._log_ppl_sum / self._sample_count
            variance = (self._log_ppl_sq_sum / self._sample_count) - (self.last_batch_log_ppl_mean ** 2)
            self.last_batch_log_ppl_std = max(0, variance) ** 0.5
            
            self.last_batch_repetition_rate_mean = self._rep_rate_sum / self._sample_count
            rep_variance = (self._rep_rate_sq_sum / self._sample_count) - (self.last_batch_repetition_rate_mean ** 2)
            self.last_batch_repetition_rate_std = max(0, rep_variance) ** 0.5
    
    def _get_unwrapped_model(self):
        """Get the unwrapped model (handles DeepSpeed/FSDP wrapping)."""
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
    def compute_conditional_perplexity(self, prompt: str, response: str, debug_idx: int = -1) -> tuple:
        """Compute perplexity of response given prompt using full model.
        
        We compute P(response | prompt) by:
        1. Tokenizing prompt and response SEPARATELY to avoid BPE boundary issues
        2. Concatenating token IDs directly (not strings)
        3. Computing loss only on response tokens (prompt tokens masked with -100)
        4. Returning exp(loss)
        
        CRITICAL: BPE tokenization is NOT additive - tokens(prompt + response) may differ
        from tokens(prompt) + tokens(response) due to boundary merging. By tokenizing
        separately and concatenating IDs, we ensure the prompt/response boundary is exact.
        
        Returns: (perplexity, debug_info_dict)
        """
        debug_info = {
            "prompt_len_chars": len(prompt),
            "response_len_chars": len(response),
            "prompt_len_tokens": 0,
            "response_len_tokens": 0,
            "total_len_tokens": 0,
            "loss": float('inf'),
            "perplexity": float('inf'),
        }
        
        if not response.strip():
            return float('inf'), debug_info
        
        unwrapped_model = self._get_unwrapped_model()
        device = next(unwrapped_model.parameters()).device
        
        # Tokenize prompt WITH special tokens (includes BOS)
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
        debug_info["prompt_len_tokens"] = prompt_len
        
        # Tokenize response WITHOUT special tokens (no extra BOS)
        response_ids = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False,  # Don't add BOS to response
        )["input_ids"]
        response_len = response_ids.shape[1]
        debug_info["response_len_tokens"] = response_len
        
        if response_len == 0:
            return float('inf'), debug_info
        
        # Concatenate token IDs directly (avoids BPE boundary issues)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Truncate if too long
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            # Recalculate response_len after truncation
            response_len = min(response_len, self.max_length - prompt_len)
            if response_len <= 0:
                return float('inf'), debug_info
        
        total_len = input_ids.shape[1]
        debug_info["total_len_tokens"] = total_len
        
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Create labels: -100 for prompt tokens (ignored in loss), actual IDs for response
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # Mask prompt tokens - now exact boundary
        
        # Forward pass with full model (controller disabled)
        outputs = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            return float('inf'), debug_info
        
        loss_val = loss.item()
        ppl = torch.exp(loss).item()
        debug_info["loss"] = loss_val
        debug_info["perplexity"] = ppl
        
        return ppl, debug_info
    
    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Score a batch of responses based on perplexity under the full model.
        
        Returns rewards in [0, reward_max] range.
        Lower perplexity = higher reward.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        
        unwrapped_model = self._get_unwrapped_model()
        
        # Disable controller to use full experts for perplexity computation
        num_disabled = self._set_controller_enabled(unwrapped_model, False)
        print(f"[PPL-REWARD] === Starting batch scoring (n={len(prompts)}) ===", flush=True)
        print(f"[PPL-REWARD] Disabled controller on {num_disabled} MoE blocks", flush=True)
        
        try:
            # Set model to eval mode for perplexity computation
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            
            perplexities = []
            rewards = []
            debug_infos = []
            
            for idx, (prompt, response) in enumerate(zip(prompts, responses)):
                ppl, debug_info = self.compute_conditional_perplexity(prompt, response, debug_idx=idx)
                perplexities.append(ppl)
                debug_infos.append(debug_info)
                
                # Convert perplexity to reward
                # reward = reward_max - log(ppl) / reward_scale (no clamping)
                if ppl == float('inf') or ppl > 1e6:
                    # For invalid perplexity, use a large negative reward
                    log_ppl = 10.0  # Corresponds to ppl ~22000
                    ppl_reward = self.reward_max - log_ppl / self.reward_scale
                else:
                    log_ppl = math.log(max(ppl, 1.0))
                    ppl_reward = self.reward_max - log_ppl / self.reward_scale
                    # No clamping - allow negative rewards for high perplexity
                
                # Compute repetition penalty
                # repetition_rate = 1 - unique_tokens / total_tokens
                # Higher repetition_rate = more repetitive = lower reward
                response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                if len(response_tokens) > 0:
                    unique_tokens = len(set(response_tokens))
                    total_tokens = len(response_tokens)
                    repetition_rate = 1.0 - unique_tokens / total_tokens
                else:
                    repetition_rate = 0.0
                
                # Apply additive penalty
                reward = ppl_reward - self.repetition_penalty * repetition_rate
                
                rewards.append(reward)
                debug_info['repetition_rate'] = repetition_rate
                debug_info['ppl_reward'] = ppl_reward
                
                # Debug print for first 3 samples in batch
                if idx < 3:
                    print(f"[PPL-REWARD] Sample {idx}: "
                          f"prompt_tokens={debug_info['prompt_len_tokens']}, "
                          f"response_tokens={debug_info['response_len_tokens']}, "
                          f"loss={debug_info['loss']:.4f}, "
                          f"ppl={ppl:.2f}, "
                          f"ppl_reward={ppl_reward:.3f}, "
                          f"rep_rate={repetition_rate:.3f}, "
                          f"final_reward={reward:.3f}", flush=True)
                    # Show snippet of response (first 100 chars)
                    response_snippet = response[:100].replace('\n', ' ')
                    print(f"[PPL-REWARD]   Response snippet: {response_snippet}...", flush=True)
            
            # Summary statistics
            valid_ppls = [p for p in perplexities if p != float('inf') and p < 1e6]
            rep_rates = [d['repetition_rate'] for d in debug_infos]
            
            if valid_ppls:
                # Compute log-perplexity stats (this is what we use for reward)
                log_ppls = [math.log(p) for p in valid_ppls]
                avg_log_ppl = sum(log_ppls) / len(log_ppls)
                std_log_ppl = (sum((lp - avg_log_ppl) ** 2 for lp in log_ppls) / len(log_ppls)) ** 0.5
                min_log_ppl = min(log_ppls)
                max_log_ppl = max(log_ppls)
                
                # Also keep raw perplexity stats for reference in prints
                avg_ppl = sum(valid_ppls) / len(valid_ppls)
                min_ppl = min(valid_ppls)
                max_ppl = max(valid_ppls)
                avg_reward = sum(rewards) / len(rewards)
                
                # Repetition rate stats
                avg_rep_rate = sum(rep_rates) / len(rep_rates)
                std_rep_rate = (sum((r - avg_rep_rate) ** 2 for r in rep_rates) / len(rep_rates)) ** 0.5
                
                # ACCUMULATE stats across sub-batches (for gradient accumulation)
                # Caller should call reset_batch_stats() at start of training step
                # and finalize_batch_stats() after all sub-batches
                n_valid = len(valid_ppls)
                self._log_ppl_sum += sum(log_ppls)
                self._log_ppl_sq_sum += sum(lp**2 for lp in log_ppls)
                self._rep_rate_sum += sum(rep_rates)
                self._rep_rate_sq_sum += sum(r**2 for r in rep_rates)
                self._sample_count += n_valid
                
                # Also update the "last batch" values for this sub-batch (for debug prints)
                self.last_batch_log_ppl_mean = avg_log_ppl
                self.last_batch_log_ppl_std = std_log_ppl
                self.last_batch_repetition_rate_mean = avg_rep_rate
                self.last_batch_repetition_rate_std = std_rep_rate
                
                # Also compute average token lengths
                avg_prompt_tokens = sum(d['prompt_len_tokens'] for d in debug_infos) / len(debug_infos)
                avg_response_tokens = sum(d['response_len_tokens'] for d in debug_infos) / len(debug_infos)
                
                print(f"[PPL-REWARD] Batch summary:", flush=True)
                print(f"[PPL-REWARD]   Raw PPL: mean={avg_ppl:.2f}, min={min_ppl:.2f}, max={max_ppl:.2f}", flush=True)
                print(f"[PPL-REWARD]   Log PPL: mean={avg_log_ppl:.3f}, std={std_log_ppl:.3f}, min={min_log_ppl:.3f}, max={max_log_ppl:.3f}", flush=True)
                print(f"[PPL-REWARD]   Repetition rate: mean={avg_rep_rate:.3f}, std={std_rep_rate:.3f}", flush=True)
                print(f"[PPL-REWARD]   Reward: mean={avg_reward:.3f}", flush=True)
                print(f"[PPL-REWARD]   Tokens: prompt_mean={avg_prompt_tokens:.1f}, response_mean={avg_response_tokens:.1f}", flush=True)
                print(f"[PPL-REWARD]   Valid samples: {len(valid_ppls)}/{len(perplexities)}", flush=True)
            else:
                print(f"[PPL-REWARD] WARNING: No valid perplexity values in batch!", flush=True)
                # Set default values
                self.last_batch_log_ppl_mean = 0.0
                self.last_batch_log_ppl_std = 0.0
                self.last_batch_repetition_rate_mean = 0.0
                self.last_batch_repetition_rate_std = 0.0
            
            # Restore training mode
            if was_training:
                unwrapped_model.train()
            
            print(f"[PPL-REWARD] Re-enabled controller on {num_disabled} MoE blocks", flush=True)
            print(f"[PPL-REWARD] === Batch scoring complete ===", flush=True)
            
            return rewards
            
        except Exception as e:
            import traceback
            print(f"[PPL-REWARD] ERROR in score_batch: {e}", flush=True)
            traceback.print_exc()
            # Re-enable controller even on error
            self._set_controller_enabled(unwrapped_model, True)
            raise
        finally:
            # Re-enable controller
            self._set_controller_enabled(unwrapped_model, True)


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
    ):
        """
        Args:
            model: The gpt-oss model (may be wrapped by accelerator/deepspeed)
            tokenizer: The tokenizer
            accelerator: The accelerator instance
            max_length: Maximum sequence length
            reward_scale: Scale factor for KL reward (default 1.0)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.max_length = max_length
        self.reward_scale = reward_scale
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
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
        """Get the unwrapped model (handles DeepSpeed/FSDP wrapping)."""
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
    def compute_kl_divergence(self, prompt: str, response: str, debug_idx: int = -1) -> tuple:
        """Compute per-token KL divergence between student (controller) and teacher (full model).
        
        We compute the FULL reverse KL divergence KL(student || teacher) at each position:
        1. Tokenize prompt and response separately
        2. Get logits from BOTH student (controller enabled) and teacher (controller disabled)
        3. Compute FULL KL over the vocabulary at each position:
           KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
        4. Sum across response token positions
        
        This is always >= 0, with 0 meaning perfect match.
        
        Returns: (kl_sum, per_token_kl_tensor, debug_info_dict)
            - kl_sum: scalar sum of KL across all tokens
            - per_token_kl_tensor: [response_len] tensor of per-position KL values
            - debug_info_dict: debugging information
        """
        debug_info = {
            "prompt_len_tokens": 0,
            "response_len_tokens": 0,
            "kl_sum": 0.0,
            "kl_per_token": 0.0,
            "teacher_ppl": float('inf'),
            "student_ppl": float('inf'),
        }
        
        if not response.strip():
            return 0.0, None, debug_info  # None for empty per_token_kl
        
        unwrapped_model = self._get_unwrapped_model()
        device = next(unwrapped_model.parameters()).device
        
        # Tokenize prompt WITH special tokens (includes BOS)
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
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
            return 0.0, None, debug_info
        
        # Concatenate token IDs
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Truncate if too long
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            response_len = min(response_len, self.max_length - prompt_len)
            if response_len <= 0:
                return 0.0, None, debug_info

        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Create labels for loss computation (mask prompt tokens)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        
        # Forward pass with TEACHER (controller disabled = full model)
        self._set_controller_enabled(unwrapped_model, False)
        teacher_outputs = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        teacher_logits = teacher_outputs.logits  # [1, seq_len, vocab]
        teacher_loss = teacher_outputs.loss
        
        # Forward pass with STUDENT (controller enabled)
        self._set_controller_enabled(unwrapped_model, True)
        student_outputs = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        student_logits = student_outputs.logits  # [1, seq_len, vocab]
        student_loss = student_outputs.loss
        
        # Compute log probabilities for response tokens
        # Shift: logits at position t predict token at position t+1
        # So for response tokens [prompt_len : prompt_len + response_len],
        # we use logits at [prompt_len - 1 : prompt_len + response_len - 1]
        
        # Get the target token IDs (response tokens)
        target_ids = input_ids[0, prompt_len:]  # [response_len]
        
        # Get logits for predicting response tokens
        # logits at position (prompt_len - 1) predicts first response token
        teacher_logits_for_response = teacher_logits[0, prompt_len-1:prompt_len+response_len-1, :]  # [response_len, vocab]
        student_logits_for_response = student_logits[0, prompt_len-1:prompt_len+response_len-1, :]  # [response_len, vocab]
        
        # Compute log softmax (over vocabulary)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits_for_response.float(), dim=-1)  # [response_len, vocab]
        student_log_probs = torch.nn.functional.log_softmax(student_logits_for_response.float(), dim=-1)  # [response_len, vocab]
        
        # Compute FULL KL divergence over vocabulary at each position
        # KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
        # This is the reverse KL as described in https://thinkingmachines.ai/blog/on-policy-distillation/
        #
        # PyTorch's F.kl_div(input, target, log_target=True) computes:
        #   target * (log target - input) = P * (log P - log Q) = KL(P || Q)
        # So for KL(student || teacher), we use:
        #   F.kl_div(input=log_teacher, target=log_student, log_target=True)
        per_token_kl = torch.nn.functional.kl_div(
            teacher_log_probs,           # input: log Q (teacher)
            student_log_probs,           # target: log P (student)
            reduction='none',            # [response_len, vocab]
            log_target=True,             # target is log probs
        ).sum(dim=-1)                    # [response_len] - sum over vocabulary
        
        # Note: This is always >= 0 (KL divergence is non-negative)
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
    
    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Score a batch of responses based on KL divergence.
        
        Reward = -KL_sum * scale (we want to minimize KL, so negative)
        
        KL(student || teacher) >= 0 always (this is a true KL over the full vocab)
        - KL = 0: student perfectly matches teacher
        - KL > 0: student diverges from teacher
        
        Higher reward (less negative) = smaller KL = better match to teacher.
        
        Also stores per_token_kl tensors in self.last_batch_per_token_kl for Option-Critic.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        
        unwrapped_model = self._get_unwrapped_model()
        
        print(f"[KL-REWARD] === Starting batch scoring (n={len(prompts)}) ===", flush=True)
        
        try:
            # Set model to eval mode
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            
            kl_values = []
            rewards = []
            debug_infos = []
            per_token_kls = []  # Store per-token KL for Option-Critic
            
            for idx, (prompt, response) in enumerate(zip(prompts, responses)):
                kl_sum, per_token_kl, debug_info = self.compute_kl_divergence(prompt, response, debug_idx=idx)
                kl_values.append(kl_sum)
                debug_infos.append(debug_info)
                per_token_kls.append(per_token_kl)  # May be None if empty response
                
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
                    # Show snippet of response
                    response_snippet = response[:100].replace('\n', ' ')
                    print(f"[KL-REWARD]   Response snippet: {response_snippet}...", flush=True)
            
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
) -> List[str]:
    """Load prompts from Nemotron dataset.
    
    The dataset can be organized in two ways:
    1. Flat: parquet files directly in data_dir (e.g., chat-00000.parquet)
    2. Nested: subdirectories per category with parquet files inside
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


def tokenize_prompts(tokenizer, prompts: List[str]) -> Dataset:
    """Tokenize prompts into a dataset (no padding - pad per batch later)."""
    encoded = tokenizer(
        prompts,
        padding=False,  # Don't pad globally - pad per batch in collate_fn
        truncation=False,
        return_tensors=None,
    )
    return Dataset.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    })


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, pad_token_id: int = 0) -> DataLoader:
    """Create a DataLoader from a dataset with per-batch padding."""
    def collate_fn(batch):
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
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# =============================================================================
# Model Setup
# =============================================================================

def freeze_non_controller(module: nn.Module) -> None:
    """Freeze all parameters except controller."""
    for name, param in module.named_parameters():
        if "controller" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


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
    
    # Data
    parser.add_argument("--data-dir", type=Path, 
                        default=Path("/scratch/gpfs/HENDERSON/zs7353/mmlu"))
    parser.add_argument("--dataset-type", type=str, default="mmlu", choices=["mmlu", "nemotron"],
                        help="Dataset type: 'mmlu' for MMLU multiple-choice, 'nemotron' for Nemotron open-ended")
    parser.add_argument("--prompts-per-category", type=int, default=50,
                        help="Number of prompts per category (default 50 for MMLU, 100 for Nemotron)")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    
    # Training
    parser.add_argument("--output-dir", type=Path, default=Path("./controller_rl_standalone"))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--num-update-epochs", type=int, default=1)
    parser.add_argument("--response-length", type=int, default=1024)
    
    # Loss
    parser.add_argument("--value-coef", type=float, default=0.1)
    parser.add_argument("--latency-cost", type=float, default=10.0)
    
    # Initialization
    parser.add_argument("--switch-init-bias", type=float, default=0.0,
                        help="Initial bias for switch head (negative = less switching, e.g. -3 for ~5%%, -4 for ~2%%)")
    
    # Reward Function
    parser.add_argument("--reward-type", type=str, default="ppl", 
                        choices=["ppl", "ppl_only", "kl"],
                        help="Reward type: 'ppl' (perplexity + repetition penalty), "
                             "'ppl_only' (perplexity only, no repetition penalty), "
                             "'kl' (KL divergence between controller and full model)")
    parser.add_argument("--ppl-reward-scale", type=float, default=3.0,
                        help="Perplexity reward scale: reward = max - log(ppl) / scale (default 3.0)")
    parser.add_argument("--ppl-reward-max", type=float, default=2.0,
                        help="Maximum perplexity reward value (default 2.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Additive penalty for repetition: reward -= penalty * (1 - unique_tokens/total_tokens)")
    parser.add_argument("--kl-reward-scale", type=float, default=0.1,
                        help="KL reward scale: reward = -kl_sum * scale (default 0.1)")
    
    # Advantage computation method
    parser.add_argument("--advantage-method", type=str, default="option_critic", choices=["option_critic", "grpo"],
                        help="Advantage computation: 'option_critic' (Harb et al. 2017, per-token TD) or 'grpo' (group-level baseline)")
    parser.add_argument("--num-generations-per-prompt", type=int, default=4,
                        help="Number of rollouts per prompt (only used for GRPO, default 4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for Option-Critic TD targets (default 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for bias-variance tradeoff (0=TD(0), 1=MC, default 0.95)")
    parser.add_argument("--deliberation-cost", type=float, default=0.1,
                        help="Option-Critic deliberation cost η per switch (default 0.1). "
                             "Should be on scale of per-token rewards (~0.01-1.0)")
    
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
    
    # DeepSpeed config
    ds_config_path = Path(__file__).parent / "deepspeed_config.json"
    if ds_config_path.exists():
        with ds_config_path.open("r", encoding="utf-8") as fh:
            ds_config_data = json.load(fh)
        HfDeepSpeedConfig(copy.deepcopy(ds_config_data))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    if accelerator.is_main_process:
        print(f"[DATA] Loading prompts from {args.data_dir} (dataset_type={args.dataset_type})")
    
    if args.dataset_type == "mmlu":
        prompts, _ = collect_mmlu_prompts(
            args.data_dir,
            args.prompts_per_category,
            args.max_prompt_length,
            tokenizer,
            seed=args.seed,
        )
    else:
        prompts = collect_prompts_by_category(
            args.data_dir,
            args.prompts_per_category,
            args.max_prompt_length,
            tokenizer,
            seed=args.seed,
        )
    
    if accelerator.is_main_process:
        print(f"[DATA] Loaded {len(prompts)} prompts")
    
    # Create dataset and dataloader
    # For GRPO: adjust batch size so total generations per step stays constant
    # Each prompt generates num_generations completions, so we need fewer unique prompts
    if args.advantage_method == "grpo":
        if args.per_device_train_batch % args.num_generations_per_prompt != 0:
            raise ValueError(
                f"For GRPO, per_device_train_batch ({args.per_device_train_batch}) must be divisible by "
                f"num_generations_per_prompt ({args.num_generations_per_prompt})"
            )
        effective_batch_size = args.per_device_train_batch // args.num_generations_per_prompt
        if accelerator.is_main_process:
            print(f"[GRPO] Adjusting dataloader batch: {args.per_device_train_batch} -> {effective_batch_size} unique prompts")
            print(f"[GRPO] Each prompt gets {args.num_generations_per_prompt} generations")
            print(f"[GRPO] Total generations per GPU per accumulation step: {effective_batch_size * args.num_generations_per_prompt}")
    else:
        effective_batch_size = args.per_device_train_batch
    
    train_dataset = tokenize_prompts(tokenizer, prompts)
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        pad_token_id=tokenizer.pad_token_id,
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
    
    if accelerator.is_main_process:
        print(f"[CONFIG] model_type = {config.model_type}")
        print(f"[CONFIG] num_experts = {num_experts}")
        print(f"[CONFIG] controller_hidden_dim = {config.controller_hidden_dim}")
        print(f"[CONFIG] controller_layer_norm = {config.controller_layer_norm}")
        print(f"[CONFIG] controller_input_type = {config.controller_input_type}")
        print(f"[CONFIG] controller_expert_embed_dim = {config.controller_expert_embed_dim}")
        print(f"[CONFIG] activation_controller_mlp_hidden = {config.activation_controller_mlp_hidden}")
        print(f"[CONFIG] controller_type = {config.controller_type}")
    
    # Load model
    if accelerator.is_main_process:
        print(f"[MODEL] Loading from {args.model_path}")
    
    # Load model in bfloat16 (required by Triton MoE kernels)
    # Controller computations will be done in float32 separately in controller_trainer.py
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    
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
    
    # Freeze non-controller parameters
    freeze_non_controller(model)
    
    if accelerator.is_main_process:
        print_trainable_params(model, "Policy Model")
    
    # Create reward function based on reward type
    reward_scorer = None  # Will hold the scorer object for logging
    
    if args.reward_type == "kl":
        # KL divergence reward: -KL(student || teacher)
        if accelerator.is_main_process:
            print(f"[REWARD] Using KL divergence reward (scale={args.kl_reward_scale})")
        
        kl_scorer = KLReward(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_length=args.response_length + args.max_prompt_length,
            reward_scale=args.kl_reward_scale,
        )
        reward_fn = lambda p, r: torch.tensor(kl_scorer.score_batch(p, r), dtype=torch.float32)
        reward_scorer = kl_scorer
        
    elif args.reward_type == "ppl_only":
        # Perplexity-only reward (no repetition penalty)
        if accelerator.is_main_process:
            print(f"[REWARD] Using perplexity-only reward (scale={args.ppl_reward_scale}, max={args.ppl_reward_max}, NO repetition penalty)")
        
        ppl_scorer = PerplexityReward(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_length=args.response_length + args.max_prompt_length,
            reward_scale=args.ppl_reward_scale,
            reward_max=args.ppl_reward_max,
            repetition_penalty=0.0,  # No repetition penalty
        )
        reward_fn = lambda p, r: torch.tensor(ppl_scorer.score_batch(p, r), dtype=torch.float32)
        reward_scorer = ppl_scorer
        
    else:  # args.reward_type == "ppl" (default)
        # Perplexity + repetition penalty
        if accelerator.is_main_process:
            print(f"[REWARD] Using perplexity reward (scale={args.ppl_reward_scale}, max={args.ppl_reward_max}, rep_penalty={args.repetition_penalty})")
        
        ppl_scorer = PerplexityReward(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_length=args.response_length + args.max_prompt_length,
            reward_scale=args.ppl_reward_scale,
            reward_max=args.ppl_reward_max,
            repetition_penalty=args.repetition_penalty,
        )
        reward_fn = lambda p, r: torch.tensor(ppl_scorer.score_batch(p, r), dtype=torch.float32)
        reward_scorer = ppl_scorer
    
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
        value_coef=args.value_coef,
        latency_cost_per_switch=args.latency_cost,
        option_critic_deliberation_cost=args.deliberation_cost,  # η for Option-Critic
        switch_init_bias=args.switch_init_bias,
        advantage_method=args.advantage_method,
        num_generations_per_prompt=args.num_generations_per_prompt,
        gamma=args.gamma,  # Discount factor for Option-Critic
        gae_lambda=args.gae_lambda,  # GAE lambda for bias-variance tradeoff
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

