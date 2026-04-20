#!/usr/bin/env python
"""
MATH (Hendrycks) Evaluation Script for Controller Checkpoints

Evaluates models on the Hendrycks MATH dataset with open-ended math problems.
Model loading follows exactly the same pattern as eval_mmlu.py and debug_kl_mmlu.py.

Usage:
    python eval_math.py --checkpoint-dir <path> --steps 100 200 300
"""

import os
import sys
import json
import random
import re
import argparse
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple
from functools import partial
from contextlib import contextmanager

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add eval directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import answer equivalence checker
from is_equiv import is_equiv


# =============================================================================
# Expert tracking and patching utilities (from eval_mmlu.py)
# =============================================================================

@contextmanager
def patch_router_for_fixed_experts(model, expert_indices_per_layer: Dict[int, List[int]], top_k: int):
    """
    Context manager that patches routers to only select from fixed experts PER LAYER.
    """
    # Find all MLP modules with router, extract layer index
    # Use same pattern as FrequencyExpertSelector: find modules with both 'router' and 'experts'
    routers = []
    for name, module in model.named_modules():
        if hasattr(module, 'router') and hasattr(module, 'experts'):
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                router = module.router
                routers.append((layer_idx, name, router))
    
    if not routers:
        print("Warning: No routers found in model!")
        yield
        return
    
    routers.sort(key=lambda x: x[0])
    num_experts = routers[0][2].weight.shape[0]
    num_layers = len(routers)
    
    print(f"[PATCH] Found {num_layers} routers with {num_experts} experts")
    print(f"[PATCH] Using PER-LAYER expert selection (top_k={top_k})")
    
    # Save original biases and modify to mask out non-selected experts
    original_biases = []
    large_negative = -1e9
    
    for layer_idx, name, router in routers:
        if layer_idx in expert_indices_per_layer:
            fixed_set = set(expert_indices_per_layer[layer_idx])
        else:
            print(f"  WARNING: Layer {layer_idx} not in expert_indices_per_layer, using all experts")
            fixed_set = set(range(num_experts))
        
        if len(fixed_set) == 0:
            print(f"  ERROR: Layer {layer_idx} has EMPTY expert set! Using all experts instead.")
            fixed_set = set(range(num_experts))
        
        original_biases.append((router, router.bias.data.clone()))
        
        with torch.no_grad():
            for expert_idx in range(num_experts):
                if expert_idx not in fixed_set:
                    router.bias.data[expert_idx] = large_negative
    
    try:
        yield
    finally:
        # Restore original biases
        for router, original_bias in original_biases:
            router.bias.data.copy_(original_bias)
        print(f"[PATCH] Restored {len(original_biases)} router biases")


class ExpertUsageTracker:
    """Tracks which experts are used during generation PER LAYER."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.expert_counts_per_layer = {}
        self.total_tokens_per_layer = {}
    
    def install_hooks(self):
        """Install forward pre-hooks on MLPs to track expert usage."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    self.expert_counts_per_layer[layer_idx] = Counter()
                    self.total_tokens_per_layer[layer_idx] = 0
                    hook = module.register_forward_pre_hook(
                        self._make_mlp_pre_hook(name, module, layer_idx)
                    )
                    self.hooks.append(hook)
        print(f"[ExpertTracker] Installed {len(self.hooks)} pre-hooks on layers: {sorted(self.expert_counts_per_layer.keys())}")
    
    def _make_mlp_pre_hook(self, name, mlp, layer_idx):
        def hook(module, args):
            if len(args) == 0:
                return
            hidden_states = args[0]
            router = mlp.router
            
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, hidden_size = hidden_states.shape
                hidden_flat = hidden_states.view(-1, hidden_size)
            else:
                hidden_flat = hidden_states
            
            with torch.no_grad():
                router_logits = F.linear(hidden_flat, router.weight, router.bias)
                top_k = router.top_k if hasattr(router, 'top_k') else 8
                _, indices = torch.topk(router_logits, top_k, dim=-1)
            
                for idx in indices.view(-1).tolist():
                    self.expert_counts_per_layer[layer_idx][idx] += 1
                self.total_tokens_per_layer[layer_idx] += indices.shape[0]
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_top_experts_per_layer(self, k: int) -> Dict[int, List[int]]:
        """Get top-k most frequent experts for each layer."""
        result = {}
        for layer_idx, counter in self.expert_counts_per_layer.items():
            top_experts = [exp for exp, _ in counter.most_common(k)]
            result[layer_idx] = top_experts
        return result
    
    def reset(self):
        """Reset all counts - use between samples for per-sample tracking."""
        for layer_idx in self.expert_counts_per_layer:
            self.expert_counts_per_layer[layer_idx] = Counter()
            self.total_tokens_per_layer[layer_idx] = 0
    
    def __enter__(self):
        self.install_hooks()
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()


class RouterSwitchTracker:
    """Track router expert selections to compute switch rate for the base model.
    
    For each layer, records per-token: the top-4 activated experts (router's
    actual routing) and a larger top-N set (for cache-based switch simulation).
    
    Cache-based switch rate (compute_cache_switch_rate):
      At the first generated token, "load" the top-cache_k experts from router
      logits.  For each subsequent token, if any of the top-4 activated experts
      falls outside the loaded set, trigger a switch and reload the top-cache_k.
    """

    def __init__(self, model, store_top_n=16):
        self.model = model
        self.hooks = []
        self.layer_selections = {}
        self.store_top_n = store_top_n

    def install_hooks(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    self.layer_selections[layer_idx] = []
                    router = module.router
                    route_k = router.top_k if hasattr(router, 'top_k') else 4
                    hook = router.register_forward_hook(
                        self._make_router_hook(layer_idx, route_k)
                    )
                    self.hooks.append(hook)

    def _make_router_hook(self, layer_idx, route_k):
        store_n = self.store_top_n
        n = max(store_n, route_k)
        def hook(module, input, output):
            with torch.no_grad():
                router_logits = module._last_router_logits
                if len(router_logits.shape) == 3:
                    router_logits = router_logits.view(-1, router_logits.shape[-1])
                _, indices = torch.topk(router_logits, n, dim=-1)
                for t in range(indices.shape[0]):
                    all_idx = indices[t].tolist()
                    self.layer_selections[layer_idx].append({
                        'top_route': frozenset(all_idx[:route_k]),
                        'top_n': all_idx,
                    })
        return hook

    def compute_switch_rate(self, prompt_len):
        """Legacy: switch whenever top-4 set differs between consecutive tokens."""
        layer_rates = []
        for layer_idx in sorted(self.layer_selections.keys()):
            sels = self.layer_selections[layer_idx]
            gen_sels = sels[prompt_len:]
            if len(gen_sels) < 2:
                continue
            switches = sum(
                1 for i in range(1, len(gen_sels))
                if gen_sels[i]['top_route'] != gen_sels[i - 1]['top_route']
            )
            layer_rates.append(switches / (len(gen_sels) - 1))
        if layer_rates:
            return sum(layer_rates) / len(layer_rates)
        return None

    def compute_cache_switch_rate(self, cache_k, prompt_len):
        """Simulate a loaded-expert cache of size cache_k.
        
        At the first generated token, load the top-cache_k experts from router
        logits.  For each subsequent token, if any of the top-4 activated
        experts is not in the loaded set, trigger a switch and reload the
        top-cache_k from the current token's router logits.
        
        Returns: average switch rate across layers (fraction of gen tokens
                 where a reload occurs), or None.
        """
        layer_rates = []
        for layer_idx in sorted(self.layer_selections.keys()):
            sels = self.layer_selections[layer_idx]
            gen_sels = sels[prompt_len:]
            if len(gen_sels) < 2:
                continue
            loaded_set = set(gen_sels[0]['top_n'][:cache_k])
            switches = 0
            for i in range(1, len(gen_sels)):
                activated = gen_sels[i]['top_route']
                if not activated.issubset(loaded_set):
                    switches += 1
                    loaded_set = set(gen_sels[i]['top_n'][:cache_k])
            layer_rates.append(switches / (len(gen_sels) - 1))
        if layer_rates:
            return sum(layer_rates) / len(layer_rates)
        return None

    def reset(self):
        for layer_idx in self.layer_selections:
            self.layer_selections[layer_idx] = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_math_samples(
    data_dir: Path,
    num_per_category: int = 10, 
    seed: int = 42, 
    num_categories: Optional[int] = None,
    levels: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    """
    Load random samples from local MATH parquet files.
    
    Args:
        data_dir: Path to hendrycks_math directory
        num_per_category: Number of samples per category
        seed: Random seed
        num_categories: Optionally limit number of categories
        levels: Optionally filter by difficulty levels (e.g., ["Level 1", "Level 2"])
    """
    random.seed(seed)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"MATH data directory not found: {data_dir}")
    
    # Get all category directories
    all_category_dirs = [
        d for d in sorted(data_dir.iterdir()) 
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    # Optionally limit number of categories for quick testing
    if num_categories is not None and num_categories < len(all_category_dirs):
        category_dirs = random.sample(all_category_dirs, num_categories)
        print(f"Quick test mode: using {num_categories} categories")
    else:
        category_dirs = all_category_dirs
    
    samples_by_category = {}
    
    print(f"Loading MATH samples ({num_per_category} per category, {len(category_dirs)} categories)...")
    for cat_dir in tqdm(category_dirs, desc="Loading categories"):
        cat = cat_dir.name
        
        # Look for test split
        test_files = list(cat_dir.glob("test-*.parquet"))
        if not test_files:
            print(f"  Warning: No test file for {cat}")
            continue
        
        try:
            df = pd.read_parquet(test_files[0])
        except Exception as e:
            print(f"  Warning: Could not load {test_files[0]}: {e}")
            continue
        
        if not all(col in df.columns for col in ["problem", "solution"]):
            print(f"  Warning: Missing columns in {cat}, columns: {list(df.columns)}")
            continue
        
        # Filter by difficulty level if specified
        if levels is not None:
            df = df[df["level"].isin(levels)]
            if len(df) == 0:
                print(f"  Warning: No samples in {cat} for levels {levels}")
                continue
        
        # Sample random indices
        indices = random.sample(range(len(df)), min(num_per_category, len(df)))
        
        samples = []
        for idx in indices:
            row = df.iloc[idx]
            # Extract answer from solution (inside \boxed{...})
            answer = extract_boxed_answer(row["solution"])
            samples.append({
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
                "level": row.get("level", "Unknown"),
                "type": row.get("type", cat),
            })
        
        samples_by_category[cat] = samples
    
    total = sum(len(v) for v in samples_by_category.values())
    print(f"Loaded {total} samples from {len(samples_by_category)} categories")
    return samples_by_category


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{...} in solution text.
    Handles nested braces.
    """
    # Find \boxed{ and then match braces
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end()
    brace_count = 1
    end = start
    
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1
    
    if brace_count == 0:
        return text[start:end-1].strip()
    return None


def extract_answer_tags(text: str) -> Optional[str]:
    """
    Extract answer from <answer>...</answer> tags in model response.
    Uses the LAST complete <answer>...</answer> pair where content doesn't contain <answer>.
    """
    # Match <answer>...</answer> where content doesn't contain nested <answer>
    # This prevents matching from "inside <answer> tags" to a later </answer>
    pattern = r'<answer>((?:(?!<answer>).)*?)</answer>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return None
    
    # Use the last match
    answer = matches[-1].group(1).strip()
    
    # Remove LaTeX display mode wrappers if present
    answer = re.sub(r'^\\[\(\[]?\s*\\displaystyle\s*', '', answer)
    answer = re.sub(r'\s*\\[\)\]]?$', '', answer)
    
    return answer if answer else None


def format_math_prompt(sample: dict) -> Tuple[str, str]:
    """Format MATH sample as prompt and return (prompt, correct_answer)."""
    problem = sample["problem"]
    correct_answer = sample["answer"]
    
    prompt = f"""Solve the following math problem. Show your reasoning step by step.

CRITICAL: You MUST wrap your final answer in <answer></answer> tags. Do NOT just write "The answer is X" - you must write "The answer is <answer>X</answer>".

Example of correct format: "Therefore, the answer is <answer>42</answer>."

Problem: {problem}

Solution (remember to use <answer></answer> tags for your final answer):"""
    
    return prompt, correct_answer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.5,
    controller_sampling: bool = False,
    q_based_selection: bool = False,
    q_selection_steps: int = 10,
    q_selection_lr: float = 1.0,
    q_selection_init_w: float = 2.0,
    termination_mode: str = "sampling",
    termination_threshold: float = 0.5,
    router_switch_tracker: Optional['RouterSwitchTracker'] = None,
    joint_controller=None,
    joint_option_k: int = 8,
) -> Tuple[str, Optional[float]]:
    """
    Generate response from model.
    Follows the same pattern as eval_mmlu.py generate_response.
    """
    # Apply chat template if available (important for instruction-tuned models)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False,
        )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # CRITICAL: Only set controller_enabled=True if model has controllers
    # Check if any MLP has a controller that's not None
    has_controller = False
    for module in model.modules():
        if hasattr(module, 'controller') and module.controller is not None:
            has_controller = True
            break
    
    if has_controller:
        ctrl_count = 0
        for module in model.modules():
            if hasattr(module, 'controller_enabled'):
                module.controller_enabled = True
                ctrl_count += 1
        
        # Debug: Print once to verify
        if not hasattr(generate_response, '_debug_printed'):
            generate_response._debug_printed = True
            print(f"  [DEBUG] Set controller_enabled=True on {ctrl_count} modules")
            # Check layer_idx like debug_kl does
            for idx, layer in enumerate(model.model.layers):
                if idx < 3:
                    mlp = layer.mlp
                    layer_idx = getattr(mlp, "layer_idx", "NOT_SET")
                    ctrl_enabled = getattr(mlp, "controller_enabled", "NOT_SET")
                    experts_cls = mlp.experts.__class__.__name__ if hasattr(mlp, "experts") else "N/A"
                    print(f"  [DEBUG] Layer {idx}: mlp.layer_idx={layer_idx}, controller_enabled={ctrl_enabled}, experts={experts_cls}")
    
    # Set up controller runtime
    controller_runtime = None
    if controller_sampling:
        controller_runtime = {
            "sampling": True,
            "generator": torch.Generator(device=model.device).manual_seed(42),
            "selection_epsilon": 0.0,  # Match training default: pure Plackett-Luce
            "record_actions": {},
        }
    
    # Add Q-based selection parameters if enabled
    if q_based_selection:
        if controller_runtime is None:
            controller_runtime = {}
        controller_runtime["sampling"] = True
        controller_runtime["record_actions"] = {}
        controller_runtime["selection_epsilon"] = 0.0
        controller_runtime["q_based_selection"] = True
        controller_runtime["q_selection_steps"] = q_selection_steps
        controller_runtime["q_selection_lr"] = q_selection_lr
        controller_runtime["q_selection_epsilon"] = 0.0
        controller_runtime["q_selection_debug"] = False
        controller_runtime["q_selection_init_w"] = q_selection_init_w
        
        # Add termination mode parameters
        # termination_mode: "sampling" uses Bernoulli sampling (as in training)
        # termination_mode: "threshold" uses a fixed threshold to decide switch
        controller_runtime["termination_mode"] = termination_mode
        controller_runtime["termination_threshold"] = termination_threshold
    
    # Debug: Print controller_runtime once
    if not hasattr(generate_response, '_runtime_printed'):
        generate_response._runtime_printed = True
        print(f"  [DEBUG] controller_runtime = {controller_runtime}")
        print(f"  [DEBUG] joint_controller = {joint_controller is not None}")
    
    if joint_controller is not None:
        # =========================================================================
        # Joint option mode: manual token-by-token generation
        # =========================================================================
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssJointOptionState, _mixed_policy_sample, _runtime_get
        )
        
        # Disable per-layer controllers
        for module in model.modules():
            if hasattr(module, 'controller_enabled'):
                module.controller_enabled = False
        
        device = inputs["input_ids"].device
        current_ids = inputs["input_ids"].clone()
        batch_size = current_ids.shape[0]
        past_kv = None
        joint_state = GptOssJointOptionState()
        moe_layer_indices = joint_controller.moe_layer_indices
        num_experts = joint_controller.num_experts
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        total_switches = 0
        total_steps = 0
        
        for step in range(max_new_tokens):
            if past_kv is None:
                input_ids = current_ids
                position_ids = None
            else:
                input_ids = current_ids[:, -1:]
                position_ids = torch.tensor([[current_ids.shape[1] - 1]], device=device).expand(batch_size, 1)
            
            attention_mask = (current_ids != pad_token_id).long()
            
            step_runtime = {"record_actions": {}, "joint_option_mode": True}
            
            # Set per-layer masks from joint state
            if joint_state.current_expert_indices_all is not None:
                joint_masks = {}
                for pos, layer_idx in enumerate(moe_layer_indices):
                    layer_indices = joint_state.current_expert_indices_all[:, pos, :]
                    mask = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
                    mask.scatter_(1, layer_indices, True)
                    joint_masks[layer_idx] = mask
                step_runtime["joint_option_masks"] = joint_masks
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    controller_runtime=step_runtime,
                )
            
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            past_kv = outputs.past_key_values
            
            # Joint option decision
            step_actions = step_runtime.get("record_actions", {})
            last_layer_hidden = step_actions.get("_last_layer_post_mlp_hidden", None)
            if last_layer_hidden is not None:
                last_layer_hidden_t = last_layer_hidden[:, -1, :]
            else:
                last_layer_hidden_t = None
            
            if joint_state.current_expert_indices_all is None:
                # First token: init from router top-k
                all_indices = []
                for pos, layer_idx in enumerate(moe_layer_indices):
                    layer_data = step_actions.get(layer_idx, {})
                    layer_router_logits = layer_data.get("router_logits", None)
                    if layer_router_logits is not None:
                        top_k_indices = torch.topk(layer_router_logits[:, -1, :], joint_option_k, dim=-1).indices
                    else:
                        top_k_indices = torch.zeros(batch_size, joint_option_k, dtype=torch.long, device=device)
                    all_indices.append(top_k_indices)
                joint_state.current_expert_indices_all = torch.stack(all_indices, dim=1)
                joint_state.last_layer_hidden = last_layer_hidden_t
            else:
                # Termination decision
                h_for_decision = joint_state.last_layer_hidden
                current_all = joint_state.current_expert_indices_all
                
                with torch.no_grad():
                    switch_logits_jt, _, _ = joint_controller(h_for_decision, current_all)
                
                switch_logits_jt = switch_logits_jt.clamp(-20, 20)
                switch_probs = torch.sigmoid(switch_logits_jt)
                
                if termination_mode == "sampling":
                    rand = torch.rand_like(switch_probs)
                    switch_decision = rand < switch_probs
                else:
                    switch_decision = switch_probs > termination_threshold
                
                if switch_decision.any():
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
                        
                        with torch.no_grad():
                            candidate_logits = joint_controller.compute_selection_logits(layer_idx, h_layer)
                        candidate_logits = candidate_logits.clamp(-20, 20)
                        selected = _mixed_policy_sample(candidate_logits, joint_option_k, epsilon=0.0, generator=None)
                        selected_indices_all.append(selected)
                    selected_indices_all = torch.stack(selected_indices_all, dim=1)
                    
                    new_all = torch.where(
                        switch_decision.unsqueeze(-1).unsqueeze(-1).expand_as(selected_indices_all),
                        selected_indices_all,
                        current_all,
                    )
                    joint_state.current_expert_indices_all = new_all
                    total_switches += switch_decision.sum().item()
                
                joint_state.last_layer_hidden = last_layer_hidden_t
                total_steps += batch_size
            
            next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)
            finished = finished | (next_token == eos_token_id)
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            
            if finished.all():
                break
        
        prompt_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(current_ids[0][prompt_len:], skip_special_tokens=True)
        switch_rate = total_switches / max(total_steps, 1)
        
        # Re-enable per-layer controllers
        for module in model.modules():
            if hasattr(module, 'controller_enabled'):
                module.controller_enabled = True
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                controller_runtime=controller_runtime,
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        switch_rate = None
        if controller_runtime is not None and "record_actions" in controller_runtime:
            record = controller_runtime["record_actions"]
            if record:
                layer_rates = []
                for layer_idx, data in record.items():
                    switches = data["switches"]  # [1, prompt_len + generated_len]
                    gen_switches = switches[:, prompt_len:]
                    if gen_switches.numel() > 0:
                        layer_rates.append(gen_switches.float().mean().item())
                if layer_rates:
                    switch_rate = sum(layer_rates) / len(layer_rates)
        elif router_switch_tracker is not None:
            switch_rate = router_switch_tracker.compute_switch_rate(prompt_len)
            router_switch_tracker.reset()

    return response, switch_rate


def load_model(
    model_path: str, 
    controller_enabled: bool = False, 
    lora_path: Optional[str] = None,
    controller_type: str = "rnn",
    controller_allowed_experts: int = 16,
    device: str = "cuda"
):
    """
    Load model with or without controller and LoRA adapter.
    Follows exactly the same pattern as eval_mmlu.py load_model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
            
    config.controller_enabled = controller_enabled
    
    if controller_enabled:
        config.controller_allowed_experts = controller_allowed_experts
        config.controller_hidden_dim = 512
        config.controller_layer_norm = True
        config.controller_input_type = "router_softmax"
        config.controller_expert_embed_dim = 128
        config.controller_type = controller_type
        config.activation_controller_mlp_hidden = 1024
        config.controller_sampling_temperature = 1.0
        config.controller_switch_threshold = 0.5
    
    model_kwargs = dict(
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if lora_path is not None:
        try:
            from transformers import Mxfp4Config
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print(f"  Using Mxfp4Config(dequantize=True) for LoRA loading")
        except ImportError:
            try:
                from transformers.integrations.mxfp4 import Mxfp4Config
                model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                print(f"  Using Mxfp4Config(dequantize=True) for LoRA loading")
            except ImportError:
                print(f"  WARNING: Could not import Mxfp4Config, LoRA loading may fail")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    if controller_enabled:
        has_controller = hasattr(model.model.layers[0].mlp, 'controller') and model.model.layers[0].mlp.controller is not None
        print(f"  Controller created: {has_controller}")
        
        if not has_controller:
            print(f"  Creating controllers manually...")
            if controller_type == "rnn":
                from transformers.models.gpt_oss.modeling_gpt_oss import GptOssController
                controller_class = GptOssController
            else:
                from transformers.models.gpt_oss.modeling_gpt_oss import GptOssActivationController
                controller_class = GptOssActivationController
            
            for layer_idx, layer in enumerate(model.model.layers):
                mlp = layer.mlp
                mlp.controller_enabled = True
                mlp.controller_allowed_experts = config.controller_allowed_experts
                mlp.controller_switch_threshold = config.controller_switch_threshold
                mlp.controller_sampling_temperature = config.controller_sampling_temperature
                mlp.controller_type = controller_type
                mlp.layer_idx = layer_idx
                mlp.controller = controller_class(config)
                mlp.controller = mlp.controller.to(next(mlp.parameters()).device)
            print(f"  Created {len(model.model.layers)} controllers")
            has_controller = True
        
        if has_controller:
            ctrl = model.model.layers[0].mlp.controller
            print(f"  Controller type: {type(ctrl).__name__}")
    
    if lora_path is not None:
        print(f"Loading LoRA adapter from {lora_path}")
        from peft import PeftModel
        
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        
        target_modules = adapter_config.get("target_modules", [])
        target_parameters = adapter_config.get("target_parameters", [])
        print(f"  [LORA] Config: r={adapter_config.get('r')}, "
              f"alpha={adapter_config.get('lora_alpha')}, "
              f"target_modules={target_modules}, "
              f"{len(target_parameters)} target_parameters")
        
        # Snapshot expert weights BEFORE LoRA to verify merge changes them
        layer0_experts = model.model.layers[0].mlp.experts
        pre_lora_expert_norm = layer0_experts.gate_up_proj.data.float().norm().item()
        pre_lora_expert_hash = layer0_experts.gate_up_proj.data.float().flatten()[:8].tolist()
        # Snapshot attention weights BEFORE LoRA
        pre_lora_attn_norm = model.model.layers[0].self_attn.q_proj.weight.data.float().norm().item()
        print(f"  [LORA] BEFORE: layer0 expert gate_up_proj norm={pre_lora_expert_norm:.6f}, "
              f"attn q_proj norm={pre_lora_attn_norm:.6f}")
        print(f"  [LORA] BEFORE: layer0 expert first 8 values={[f'{v:.6f}' for v in pre_lora_expert_hash]}")
        
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  [LORA] PeftModel loaded successfully")
        
        lora_params = sum(1 for n, _ in model.named_parameters() if 'lora_' in n)
        print(f"  [LORA] LoRA parameters found in model: {lora_params}")
        if lora_params == 0:
            print(f"  [LORA] WARNING: Zero LoRA parameters found! Loading may have failed.")
        
        # Check LoRA weight norms (non-zero = trained, not just initialized)
        lora_norms = []
        for n, p in model.named_parameters():
            if 'lora_' in n and p.numel() > 0 and not p.is_meta:
                lora_norms.append((n, p.float().norm().item()))
        if lora_norms:
            norms_only = [v for _, v in lora_norms]
            n_zero = sum(1 for v in norms_only if v < 1e-8)
            print(f"  [LORA] Param norms: avg={sum(norms_only)/len(norms_only):.6f}, "
                  f"max={max(norms_only):.6f}, min={min(norms_only):.6f}, "
                  f"count={len(norms_only)}, n_zero={n_zero}")
            if n_zero > 0:
                zero_names = [n for n, v in lora_norms if v < 1e-8]
                print(f"  [LORA] WARNING: {n_zero} LoRA params are zero: {zero_names[:5]}")
        
        print(f"  [LORA] Merging LoRA weights into base parameters...")
        model = model.merge_and_unload()
        print(f"  [LORA] Merged and unloaded successfully")
        
        # Verify expert/attn weights CHANGED after merge
        layer0_experts = model.model.layers[0].mlp.experts
        post_lora_expert_norm = layer0_experts.gate_up_proj.data.float().norm().item()
        post_lora_expert_hash = layer0_experts.gate_up_proj.data.float().flatten()[:8].tolist()
        post_lora_attn_norm = model.model.layers[0].self_attn.q_proj.weight.data.float().norm().item()
        expert_changed = abs(post_lora_expert_norm - pre_lora_expert_norm) > 1e-6
        attn_changed = abs(post_lora_attn_norm - pre_lora_attn_norm) > 1e-6
        print(f"  [LORA] AFTER:  layer0 expert gate_up_proj norm={post_lora_expert_norm:.6f}, "
              f"attn q_proj norm={post_lora_attn_norm:.6f}")
        print(f"  [LORA] AFTER:  layer0 expert first 8 values={[f'{v:.6f}' for v in post_lora_expert_hash]}")
        print(f"  [LORA] Expert weights changed: {expert_changed} (diff={abs(post_lora_expert_norm - pre_lora_expert_norm):.6f})")
        print(f"  [LORA] Attn weights changed:   {attn_changed} (diff={abs(post_lora_attn_norm - pre_lora_attn_norm):.6f})")
        if not expert_changed:
            print(f"  [LORA] WARNING: Expert weights DID NOT change after merge! LoRA may not have been applied to experts.")
        if not attn_changed:
            print(f"  [LORA] WARNING: Attention weights DID NOT change after merge! LoRA may not have been applied to attention.")
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_controller_checkpoint(model, checkpoint_path: str, controller_type: str = "rnn", joint_option: bool = False, num_experts_k: int = 8):
    """
    Load controller weights from checkpoint into model.
    Handles both RNN controller (controller_state_dict) and activation controller (activation_controllers).
    
    Returns:
        joint_controller (or None): If joint_option=True and checkpoint has joint controller, returns it.
    """
    print(f"Loading controller weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  Checkpoint step: {checkpoint.get('step', 'N/A')}, epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Check if this is an activation controller checkpoint
    if "activation_controllers" in checkpoint and controller_type == "activation":
        print("  Detected activation controller checkpoint format")
        activation_controllers_state = checkpoint["activation_controllers"]
        
        # Navigate through potential PEFT wrapping
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.base_model.model
        else:
            base_model = model
        
        # Access model.model.layers for GptOssForCausalLM
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        else:
            print(f"  Error: Cannot find model layers. Model type: {type(base_model)}")
            return
        
        # --- Snapshot BEFORE loading (for before/after comparison) ---
        ctrl0_before = {k: v.clone() for k, v in layers[0].mlp.controller.state_dict().items()}
        router0_w_before = layers[0].mlp.router.weight.data.clone()
        router0_b_before = layers[0].mlp.router.bias.data.clone() if layers[0].mlp.router.bias is not None else None
        
        # --- Load activation controller weights ---
        loaded_count = 0
        for layer_idx, state_dict in activation_controllers_state.items():
            if layer_idx < len(layers):
                mlp = layers[layer_idx].mlp
                if hasattr(mlp, 'controller') and mlp.controller is not None:
                    mlp.controller.load_state_dict(state_dict)
                    loaded_count += 1
                else:
                    print(f"  Warning: Layer {layer_idx} has no controller")
            else:
                print(f"  Warning: Layer {layer_idx} out of range (model has {len(layers)} layers)")
        
        print(f"  [CONTROLLER] Loaded activation controller weights for {loaded_count}/{len(activation_controllers_state)} layers")
        
        # Verify controller weights CHANGED (before vs after)
        ctrl0_after = layers[0].mlp.controller.state_dict()
        ctrl_diffs = {}
        for k in ctrl0_before:
            diff = (ctrl0_after[k].float() - ctrl0_before[k].float()).abs().max().item()
            ctrl_diffs[k] = diff
        n_changed = sum(1 for v in ctrl_diffs.values() if v > 1e-8)
        print(f"  [CONTROLLER] Layer 0 before/after: {n_changed}/{len(ctrl_diffs)} params changed")
        for k, v in list(ctrl_diffs.items())[:5]:
            print(f"    {k}: max_diff={v:.6f}")
        if n_changed == 0:
            print(f"  [CONTROLLER] WARNING: Controller weights DID NOT change! Loading may have failed.")
        
        # --- Load router weights ---
        if "router_state_dict" in checkpoint:
            router_state = checkpoint["router_state_dict"]
            print(f"  [ROUTER] Checkpoint has {len(router_state)} router weight tensors")
            
            model_params = dict(base_model.named_parameters())
            
            sample_ckpt_key = next(iter(router_state.keys())) if router_state else "N/A"
            sample_model_router_keys = [k for k in model_params.keys() if 'router' in k][:2]
            print(f"  [ROUTER] Sample checkpoint key: {sample_ckpt_key}")
            print(f"  [ROUTER] Sample model router keys: {sample_model_router_keys}")
            
            loaded_router = 0
            unmatched_keys = []
            for name, saved_param in router_state.items():
                if name in model_params:
                    model_params[name].data.copy_(saved_param.to(model_params[name].device))
                    loaded_router += 1
                else:
                    unmatched_keys.append(name)
            
            if unmatched_keys:
                print(f"  [ROUTER] {len(unmatched_keys)} keys didn't match directly, trying prefix remapping...")
                for name in unmatched_keys:
                    if name.startswith("model.") and name[len("model."):] in model_params:
                        stripped = name[len("model."):]
                        model_params[stripped].data.copy_(router_state[name].to(model_params[stripped].device))
                        loaded_router += 1
                    elif ("model." + name) in model_params:
                        prefixed = "model." + name
                        model_params[prefixed].data.copy_(router_state[name].to(model_params[prefixed].device))
                        loaded_router += 1
                    else:
                        print(f"  [ROUTER] WARNING: Could not match key: {name}")
            
            print(f"  [ROUTER] Loaded {loaded_router}/{len(router_state)} router weight tensors")
            if loaded_router != len(router_state):
                print(f"  [ROUTER] WARNING: Not all router weights were loaded!")
            
            # Verify router weights CHANGED (before vs after)
            router0_w_after = layers[0].mlp.router.weight.data
            w_diff = (router0_w_after.float() - router0_w_before.float()).abs().max().item()
            b_diff = 0.0
            if router0_b_before is not None:
                b_diff = (layers[0].mlp.router.bias.data.float() - router0_b_before.float()).abs().max().item()
            print(f"  [ROUTER] Layer 0 before/after: weight max_diff={w_diff:.6f}, bias max_diff={b_diff:.6f}")
            if w_diff < 1e-8 and b_diff < 1e-8:
                print(f"  [ROUTER] WARNING: Router weights DID NOT change! Loading may have failed or router was not trained.")
            else:
                print(f"  [ROUTER] Router weights changed successfully.")
        else:
            print(f"  [ROUTER] WARNING: No router_state_dict in checkpoint! Router weights NOT loaded.")
        
        # Load joint controller if requested
        if joint_option and "joint_controller" in checkpoint:
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssJointOptionController
            
            # Get model config and MoE layer indices
            if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                model_config = base_model.config
                moe_layer_indices = [i for i, layer in enumerate(base_model.model.layers) 
                                     if hasattr(layer.mlp, 'controller') and layer.mlp.controller is not None]
            else:
                print(f"  [JOINT] ERROR: Cannot determine MoE layer indices")
                return None
            
            print(f"  [JOINT] Creating joint controller with {len(moe_layer_indices)} MoE layers")
            joint_ctrl = GptOssJointOptionController(
                config=model_config,
                moe_layer_indices=moe_layer_indices,
            )
            joint_ctrl.load_state_dict(checkpoint["joint_controller"])
            joint_ctrl = joint_ctrl.to(model.device if hasattr(model, 'device') else 'cuda')
            joint_ctrl.eval()
            for param in joint_ctrl.parameters():
                param.data = param.data.float()
            print(f"  [JOINT] Loaded joint controller ({sum(p.numel() for p in joint_ctrl.parameters()):,} params)")
            return joint_ctrl
        elif joint_option:
            print(f"  [JOINT] WARNING: joint_option=True but no joint_controller in checkpoint")
        
        return None
    
    # Fall back to RNN controller format (controller_state_dict)
    controller_state = checkpoint.get("controller_state_dict", {})
    
    if not controller_state:
        print("  Warning: No controller_state_dict found in checkpoint")
        return
    
    from peft import PeftModel
    is_peft = isinstance(model, PeftModel)
    print(f"  Model is PEFT-wrapped: {is_peft}")
    
    first_ckpt_key = next(iter(controller_state.keys()))
    ckpt_has_peft_prefix = first_ckpt_key.startswith("base_model.model.")
    print(f"  Checkpoint has PEFT prefix: {ckpt_has_peft_prefix}")
    
    model_controller_keys = [k for k in model.state_dict().keys() if 'controller' in k]
    if model_controller_keys:
        first_model_key = model_controller_keys[0]
        model_has_peft_prefix = first_model_key.startswith("base_model.model.")
        print(f"  Model has PEFT prefix: {model_has_peft_prefix}")
    else:
        model_has_peft_prefix = False
    
    if ckpt_has_peft_prefix and not model_has_peft_prefix:
        print("  Stripping PEFT prefix from checkpoint keys...")
        new_state = {}
        prefix = "base_model.model."
        for k, v in controller_state.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_state[new_key] = v
            else:
                new_state[k] = v
        controller_state = new_state
    elif not ckpt_has_peft_prefix and model_has_peft_prefix:
        print("  Adding PEFT prefix to checkpoint keys...")
        new_state = {}
        prefix = "base_model.model."
        for k, v in controller_state.items():
            new_state[prefix + k] = v
        controller_state = new_state
    
    missing, unexpected = model.load_state_dict(controller_state, strict=False)
    loaded = len(controller_state) - len(unexpected)
    print(f"  Loaded {loaded} controller parameters")


def evaluate_samples(
    model,
    tokenizer,
    samples_by_category: Dict[str, List[dict]],
    desc: str = "Evaluating",
    print_rollouts: bool = True,
    controller_sampling: bool = False,
    q_based_selection: bool = False,
    q_selection_steps: int = 10,
    q_selection_lr: float = 1.0,
    q_selection_init_w: float = 2.0,
    max_new_tokens: int = 1024,
    temperature: float = 0.5,
    track_experts: bool = False,
    termination_mode: str = "sampling",
    termination_threshold: float = 0.5,
    joint_controller=None,
    joint_option_k: int = 8,
) -> Tuple[float, Dict[str, float], Optional[ExpertUsageTracker]]:
    """
    Evaluate model on MATH samples.
    
    Returns:
        (overall_accuracy, accuracy_by_category, expert_tracker_or_None)
    """
    correct = 0
    total = 0
    correct_by_category = {}
    total_by_category = {}
    switch_rates = []
    
    # Set up expert tracker if needed
    expert_tracker = None
    if track_experts:
        expert_tracker = ExpertUsageTracker(model)
        expert_tracker.install_hooks()
    
    all_samples = [(cat, sample) for cat, samples in samples_by_category.items() for sample in samples]
    
    for idx, (cat, sample) in enumerate(tqdm(all_samples, desc=desc)):
        prompt, correct_answer = format_math_prompt(sample)
        
        response, switch_rate = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            controller_sampling=controller_sampling,
            q_based_selection=q_based_selection,
            q_selection_steps=q_selection_steps,
            q_selection_lr=q_selection_lr,
            q_selection_init_w=q_selection_init_w,
            termination_mode=termination_mode,
            termination_threshold=termination_threshold,
            joint_controller=joint_controller,
            joint_option_k=joint_option_k,
        )
        if switch_rate is not None:
            switch_rates.append(switch_rate)
        
        # Extract answer from model's response (using <answer> tags)
        predicted = extract_answer_tags(response)
        is_correct = is_equiv(predicted, correct_answer)
        
        if is_correct:
            correct += 1
            correct_by_category[cat] = correct_by_category.get(cat, 0) + 1
        total += 1
        total_by_category[cat] = total_by_category.get(cat, 0) + 1
        
        if print_rollouts:
            current_acc = correct / total if total > 0 else 0
            sr_str = f" | Switch rate: {switch_rate:.4f}" if switch_rate is not None else ""
            print(f"\n{'='*80}")
            print(f"[{desc}] Question {idx+1}/{len(all_samples)} | Category: {cat} | Level: {sample.get('level', 'N/A')}{sr_str}")
            print(f"{'='*80}")
            print(f"\n[PROBLEM]:\n{sample['problem']}")
            print(f"\n[MODEL RESPONSE]:\n{response}")
            print(f"\n[EXTRACTED ANSWER]: {predicted}")
            print(f"[CORRECT ANSWER]: {correct_answer}")
            print(f"[RESULT]: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print(f"[RUNNING ACCURACY]: {correct}/{total} = {current_acc:.2%}")
            if switch_rates:
                print(f"[RUNNING AVG SWITCH RATE]: {sum(switch_rates)/len(switch_rates):.4f}")
            print(f"{'='*80}\n", flush=True)
    
    overall_accuracy = correct / total if total > 0 else 0
    accuracy_by_category = {
        cat: correct_by_category.get(cat, 0) / total_by_category[cat] 
        for cat in total_by_category
    }
    avg_switch_rate = sum(switch_rates) / len(switch_rates) if switch_rates else None
    if avg_switch_rate is not None:
        print(f"\n[{desc}] Average switch rate: {avg_switch_rate:.4f}")
    
    # Clean up expert tracker
    if expert_tracker is not None:
        expert_tracker.remove_hooks()
    
    return overall_accuracy, accuracy_by_category, expert_tracker, avg_switch_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                        default="/scratch/gpfs/KOROLOVA/gpt-oss-20b")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math"),
                        help="Path to local MATH parquet data directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--samples-per-category", type=int, default=10)
    parser.add_argument("--num-categories", type=int, default=None,
                        help="Number of categories to use (for quick testing)")
    parser.add_argument("--levels", type=str, nargs="+", default=None,
                        help="Filter by difficulty levels (e.g., 'Level 1' 'Level 2')")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="baseline",
                        help="Which method to run: 'all', 'baselines', 'baseline', 'random', 'hindsight', 'controller', or 'controller_<step>'")
    parser.add_argument("--num-experts", type=int, default=16,
                        help="Number of experts to use (for random/hindsight baselines)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of experts activated per token")
    parser.add_argument("--hindsight-experts-file", type=str, default=None,
                        help="Path to JSON file with hindsight experts per layer (from baseline run)")
    parser.add_argument("--controller-type", type=str, default="rnn",
                        choices=["rnn", "activation"],
                        help="Controller type: 'rnn' or 'activation'")
    parser.add_argument("--controller-sampling", action="store_true",
                        help="Use sampling for controller switch decisions")
    parser.add_argument("--q-based-selection", action="store_true",
                        help="Use Q-based gradient optimization for expert selection")
    parser.add_argument("--q-selection-steps", type=int, default=10)
    parser.add_argument("--q-selection-lr", type=float, default=1.0)
    parser.add_argument("--q-selection-init-w", type=float, default=2.0)
    parser.add_argument("--termination-mode", type=str, default="sampling",
                        choices=["sampling", "threshold"],
                        help="Termination decision mode: 'sampling' uses Bernoulli sampling (as in training), 'threshold' uses fixed threshold")
    parser.add_argument("--termination-threshold", type=float, default=0.5,
                        help="Threshold for termination decisions when using threshold mode")
    parser.add_argument("--joint-option", type=int, default=0,
                        help="Use joint (shared) option across all layers: 0=per-layer (default), 1=joint option")
    parser.add_argument("--joint-option-k", type=int, default=8,
                        help="Number of experts per layer for joint option")
    parser.add_argument("--with-lora", action="store_true",
                        help="Also load LoRA adapter from checkpoint directory")
    parser.add_argument("--controller-allowed-experts", type=int, default=16,
                        help="Number of experts the controller selects per token (must match training config)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for generation")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="Shard index for parallel evaluation (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards for parallel evaluation")
    parser.add_argument("--base-model", action="store_true",
                        help="Evaluate base model only (no pruning, no controller, track router switch rate)")
    parser.add_argument("--cache-k", type=int, nargs="+", default=None,
                        help="Cache sizes for base model switch rate (e.g., --cache-k 8 16)")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    total_start_time = time.time()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MATH samples
    samples_by_category = load_math_samples(
        args.data_dir, 
        args.samples_per_category, 
        args.seed, 
        args.num_categories,
        args.levels,
    )
    
    # Apply sharding if num_shards > 1
    if args.num_shards > 1:
        all_samples_flat = [(cat, sample) for cat, samples in sorted(samples_by_category.items()) 
                           for sample in samples]
        total_count = len(all_samples_flat)
        shard_size = (total_count + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_idx * shard_size
        end_idx = min(start_idx + shard_size, total_count)
        
        shard_samples = all_samples_flat[start_idx:end_idx]
        
        samples_by_category = {}
        for cat, sample in shard_samples:
            if cat not in samples_by_category:
                samples_by_category[cat] = []
            samples_by_category[cat].append(sample)
        
        print(f"\nSharding: shard {args.shard_idx + 1}/{args.num_shards}, samples {start_idx}-{end_idx} of {total_count}")
    
    total_samples = sum(len(v) for v in samples_by_category.values())
    print(f"\nTotal samples to evaluate (this shard): {total_samples}")
    
    results = {}
    num_layers = 24  # GPT-OSS-20B has 24 layers
    num_total_experts = 32  # GPT-OSS-20B has 32 experts
    hindsight_experts_per_layer = {}
    
    # Determine which methods to run
    run_baseline = args.method in ["all", "baselines", "baseline"]
    run_random = args.method in ["all", "baselines", "random"]
    run_hindsight = args.method in ["all", "baselines", "hindsight"]
    run_controller = args.method == "controller" or args.method.startswith("controller_")
    
    # Parse controller steps
    if args.method.startswith("controller_"):
        try:
            step_from_method = int(args.method.split("_")[1])
            controller_steps = [step_from_method]
        except (IndexError, ValueError):
            controller_steps = args.steps
    elif args.method == "controller":
        controller_steps = args.steps
    else:
        controller_steps = []
    
    # Load hindsight experts from file if needed
    if args.hindsight_experts_file and os.path.exists(args.hindsight_experts_file):
        with open(args.hindsight_experts_file, "r") as f:
            data = json.load(f)
            hindsight_experts_per_layer = {int(k): v for k, v in data.get("hindsight_experts_per_layer", {}).items()}
            print(f"Loaded hindsight experts from {args.hindsight_experts_file}")
    
    # =========================================================================
    # Base model only (no pruning, no controller, just track router switch rate)
    # =========================================================================
    if args.base_model:
        cache_k_list = args.cache_k if args.cache_k else [8, 16]
        max_cache_k = max(cache_k_list)

        print("\n" + "="*70)
        print("BASE MODEL: Running with full experts (cache-based switch rate)")
        print(f"  Cache k values: {cache_k_list}")
        print("="*70)

        t0 = time.time()
        model, tokenizer = load_model(args.model_path, controller_enabled=False)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        switch_tracker = RouterSwitchTracker(model, store_top_n=max_cache_k)
        switch_tracker.install_hooks()
        print(f"Installed router switch tracker on {len(switch_tracker.layer_selections)} layers")

        all_samples = []
        for cat, samples in samples_by_category.items():
            for sample in samples:
                all_samples.append((cat, sample))

        correct, total = 0, 0
        by_cat = {}
        switch_rates_by_k = {ck: [] for ck in cache_k_list}

        for idx, (cat, sample) in enumerate(tqdm(all_samples, desc="Base model")):
            prompt, correct_answer = format_math_prompt(sample)

            print(f"\n{'='*80}")
            print(f"Problem {idx+1}/{len(all_samples)} | Category: {cat} | Level: {sample.get('level', 'N/A')}")
            print(f"{'='*80}")
            print(f"\n[PROBLEM]:\n{sample['problem']}")

            # Tokenize to get prompt_len (same logic as generate_response)
            fmt_prompt = prompt
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                fmt_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
            prompt_inputs = tokenizer(fmt_prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Don't pass tracker to generate_response — we compute cache switch rates ourselves
            response, _ = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # Compute cache-based switch rates for each k
            sr_parts = []
            for ck in cache_k_list:
                csr = switch_tracker.compute_cache_switch_rate(ck, prompt_len)
                if csr is not None:
                    switch_rates_by_k[ck].append(csr)
                    sr_parts.append(f"k={ck}: {csr:.4f}")
            switch_tracker.reset()

            pred = extract_answer_tags(response)
            is_correct_flag = is_equiv(pred, correct_answer)
            if is_correct_flag:
                correct += 1
                by_cat[cat] = by_cat.get(cat, 0) + 1
            total += 1

            sr_str = " | Switch: " + ", ".join(sr_parts) if sr_parts else ""
            print(f"\n[BASE MODEL] Response:\n{response}")
            print(f"[BASE MODEL] Extracted: {pred} | Correct: {correct_answer} | {'CORRECT' if is_correct_flag else 'WRONG'}{sr_str}")

            avg_parts = []
            for ck in cache_k_list:
                if switch_rates_by_k[ck]:
                    avg_parts.append(f"k={ck}: {sum(switch_rates_by_k[ck])/len(switch_rates_by_k[ck]):.4f}")
            avg_sr_str = f" | Avg switch: {', '.join(avg_parts)}" if avg_parts else ""
            print(f"\n[RUNNING] Base model: {correct}/{total} = {correct/total:.2%}{avg_sr_str}")
            print(f"{'='*80}\n", flush=True)

        switch_tracker.remove_hooks()

        accuracy = correct / total if total > 0 else 0
        results["base_model"] = accuracy
        for ck in cache_k_list:
            if switch_rates_by_k[ck]:
                avg_sr = sum(switch_rates_by_k[ck]) / len(switch_rates_by_k[ck])
                results[f"base_model_switch_rate_k{ck}"] = avg_sr
                results[f"base_model_switch_rates_k{ck}"] = switch_rates_by_k[ck]

        print("\n" + "="*70)
        print("FINAL BASE MODEL RESULTS")
        print("="*70)
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
        for ck in cache_k_list:
            if switch_rates_by_k[ck]:
                avg_sr = sum(switch_rates_by_k[ck]) / len(switch_rates_by_k[ck])
                print(f"Cache switch rate (k={ck}): {avg_sr:.4f}")
        print(f"Accuracy by category:")
        for cat_name in sorted(by_cat.keys()):
            cat_total = sum(1 for c, _ in all_samples if c == cat_name)
            print(f"  {cat_name}: {by_cat[cat_name]}/{cat_total} = {by_cat[cat_name]/cat_total:.2%}")

        os.makedirs(args.output_dir, exist_ok=True)
        cache_tag = "_".join(f"k{ck}" for ck in cache_k_list)
        out_file = os.path.join(args.output_dir, f"base_model_math_cache_sr_{cache_tag}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_file}")
        print(f"Time: {(time.time() - t0)/60:.1f} min")

        del model
        torch.cuda.empty_cache()
        return

    # =========================================================================
    # Run all baselines PER-PROBLEM (baseline, hindsight, random for each problem)
    # =========================================================================
    if run_baseline or run_random or run_hindsight:
        print("\n" + "="*70)
        print("BASELINES: Running all 3 baselines per-problem")
        print("="*70)
        
        t0 = time.time()
        model, tokenizer = load_model(args.model_path, controller_enabled=False)
        num_total_experts = model.config.num_local_experts if hasattr(model.config, 'num_local_experts') else 32
        print(f"Model loaded in {time.time() - t0:.1f}s")
        
        # Set up expert tracker for per-problem hindsight
        expert_tracker = ExpertUsageTracker(model)
        expert_tracker.install_hooks()
        
        # Set up router switch tracker for base model switch rate
        switch_tracker = RouterSwitchTracker(model)
        switch_tracker.install_hooks()
        
        # Flatten samples
        all_samples = []
        for cat, samples in samples_by_category.items():
            for sample in samples:
                all_samples.append((cat, sample))
        
        # Track results per method
        baseline_correct, baseline_total = 0, 0
        hindsight_correct, hindsight_total = 0, 0
        random_correct, random_total = 0, 0
        baseline_by_cat = {}
        hindsight_by_cat = {}
        random_by_cat = {}
        baseline_switch_rates = []
        
        random.seed(args.seed)
        
        for idx, (cat, sample) in enumerate(tqdm(all_samples, desc="Per-problem baselines")):
            prompt, correct_answer = format_math_prompt(sample)
            
            print(f"\n{'='*80}")
            print(f"Problem {idx+1}/{len(all_samples)} | Category: {cat} | Level: {sample.get('level', 'N/A')}")
            print(f"{'='*80}")
            print(f"\n[PROBLEM]:\n{sample['problem']}")
            
            # --- Baseline (no restriction) ---
            expert_tracker.reset()
            baseline_response, baseline_sr = generate_response(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, router_switch_tracker=switch_tracker)
            baseline_pred = extract_answer_tags(baseline_response)
            baseline_is_correct = is_equiv(baseline_pred, correct_answer)
            if baseline_is_correct:
                baseline_correct += 1
                baseline_by_cat[cat] = baseline_by_cat.get(cat, 0) + 1
            baseline_total += 1
            if baseline_sr is not None:
                baseline_switch_rates.append(baseline_sr)
            
            # Get hindsight experts for THIS problem
            hindsight_experts = expert_tracker.get_top_experts_per_layer(args.num_experts)
            
            sr_str = f" | Switch rate: {baseline_sr:.4f}" if baseline_sr is not None else ""
            print(f"\n[BASELINE] Response:\n{baseline_response}")
            print(f"[BASELINE] Extracted: {baseline_pred} | Correct: {correct_answer} | {'✓' if baseline_is_correct else '✗'}{sr_str}")
            
            # --- Hindsight (use experts from baseline run of THIS problem) ---
            with patch_router_for_fixed_experts(model, hindsight_experts, args.top_k):
                hindsight_response, _ = generate_response(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            hindsight_pred = extract_answer_tags(hindsight_response)
            hindsight_is_correct = is_equiv(hindsight_pred, correct_answer)
            if hindsight_is_correct:
                hindsight_correct += 1
                hindsight_by_cat[cat] = hindsight_by_cat.get(cat, 0) + 1
            hindsight_total += 1
            
            print(f"\n[HINDSIGHT] Response:\n{hindsight_response}")
            print(f"[HINDSIGHT] Extracted: {hindsight_pred} | Correct: {correct_answer} | {'✓' if hindsight_is_correct else '✗'}")
            
            # --- Random experts (different random set per problem) ---
            random_experts = {}
            for layer_idx in range(num_layers):
                random_experts[layer_idx] = random.sample(range(num_total_experts), args.num_experts)
            
            with patch_router_for_fixed_experts(model, random_experts, args.top_k):
                random_response, _ = generate_response(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            random_pred = extract_answer_tags(random_response)
            random_is_correct = is_equiv(random_pred, correct_answer)
            if random_is_correct:
                random_correct += 1
                random_by_cat[cat] = random_by_cat.get(cat, 0) + 1
            random_total += 1
            
            print(f"\n[RANDOM] Response:\n{random_response}")
            print(f"[RANDOM] Extracted: {random_pred} | Correct: {correct_answer} | {'✓' if random_is_correct else '✗'}")
            
            # Running summary
            avg_sr_str = ""
            if baseline_switch_rates:
                avg_sr_str = f" | Avg switch rate: {sum(baseline_switch_rates)/len(baseline_switch_rates):.4f}"
            print(f"\n[RUNNING] Baseline: {baseline_correct}/{baseline_total} = {baseline_correct/baseline_total:.2%}{avg_sr_str}")
            print(f"[RUNNING] Hindsight: {hindsight_correct}/{hindsight_total} = {hindsight_correct/hindsight_total:.2%}")
            print(f"[RUNNING] Random: {random_correct}/{random_total} = {random_correct/random_total:.2%}")
            print(f"{'='*80}\n", flush=True)
        
        expert_tracker.remove_hooks()
        switch_tracker.remove_hooks()
        
        # Final results
        results["baseline_no_controller"] = baseline_correct / baseline_total if baseline_total > 0 else 0
        results["hindsight_best"] = hindsight_correct / hindsight_total if hindsight_total > 0 else 0
        results["random_experts"] = random_correct / random_total if random_total > 0 else 0
        avg_baseline_sr = sum(baseline_switch_rates) / len(baseline_switch_rates) if baseline_switch_rates else None
        if avg_baseline_sr is not None:
            results["baseline_switch_rate"] = avg_baseline_sr
        
        print("\n" + "="*70)
        print("FINAL BASELINE RESULTS")
        print("="*70)
        print(f"Baseline (no restriction): {baseline_correct}/{baseline_total} = {results['baseline_no_controller']:.2%}")
        if avg_baseline_sr is not None:
            print(f"Baseline switch rate:      {avg_baseline_sr:.4f}")
        print(f"Hindsight (per-problem):   {hindsight_correct}/{hindsight_total} = {results['hindsight_best']:.2%}")
        print(f"Random experts:            {random_correct}/{random_total} = {results['random_experts']:.2%}")
        
        del model
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Controller methods
    # =========================================================================
    if run_controller:
        # Step 0 = untrained controller, no checkpoint needed
        if args.checkpoint_dir is None and 0 not in controller_steps:
            print("ERROR: --checkpoint-dir required for controller evaluation (unless step=0)")
            return
        
        print(f"\nWill evaluate controller at steps: {controller_steps}")
    
    for step in (controller_steps if run_controller else []):
        # Step 0 = untrained controller (fresh initialization)
        if step == 0:
            print("\n" + "="*70)
            print("CONTROLLER: Untrained Controller (Step 0 - fresh initialization)")
            print("="*70)
            
            t0 = time.time()
            model, tokenizer = load_model(
                args.model_path, 
                controller_enabled=True, 
                lora_path=None,
                controller_type=args.controller_type,
                controller_allowed_experts=args.controller_allowed_experts,
            )
            # Do NOT load any checkpoint - this is the untrained controller
            lora_path = None  # For consistency with later code
            lora_str = ""
            print(f"Model loaded with fresh controller in {time.time() - t0:.1f}s")
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"controller_step_{step}.pt")
            
            # Also check for activation controller checkpoint naming
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(args.checkpoint_dir, f"activation_controller_step_{step}.pt")
            
            if not os.path.exists(checkpoint_path):
                print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
                continue
            
            lora_path = None
            if args.with_lora:
                lora_dir = os.path.join(args.checkpoint_dir, f"lora_step_{step}")
                if os.path.exists(lora_dir):
                    lora_path = lora_dir
                    print(f"Found LoRA adapter at {lora_dir}")
                else:
                    print(f"Warning: --with-lora specified but no LoRA adapter found at {lora_dir}")
            
            print("\n" + "="*70)
            lora_str = " + LoRA" if lora_path else ""
            print(f"CONTROLLER: With Trained Controller{lora_str} (Step {step})")
            print("="*70)
            
            t0 = time.time()
            model, tokenizer = load_model(
                args.model_path, 
                controller_enabled=True, 
                lora_path=lora_path,
                controller_type=args.controller_type,
                controller_allowed_experts=args.controller_allowed_experts,
            )
            joint_ctrl = load_controller_checkpoint(
                model, checkpoint_path, controller_type=args.controller_type,
                joint_option=bool(args.joint_option), num_experts_k=args.joint_option_k,
            )
            print(f"Model loaded in {time.time() - t0:.1f}s")
            
            # Post-load verification: summarize what was loaded
            print(f"\n  [VERIFY] controller_allowed_experts={args.controller_allowed_experts}")
            print(f"  [VERIFY] controller_type={args.controller_type}")
            print(f"  [VERIFY] with_lora={args.with_lora}, lora_path={lora_path}")
            n_ctrl = sum(1 for layer in model.model.layers 
                         if hasattr(layer.mlp, 'controller') and layer.mlp.controller is not None)
            print(f"  [VERIFY] Controllers present in model: {n_ctrl}/{len(model.model.layers)} layers")
            # Check controller_enabled flags
            n_enabled = sum(1 for layer in model.model.layers 
                           if getattr(layer.mlp, 'controller_enabled', False))
            print(f"  [VERIFY] controller_enabled=True on {n_enabled}/{len(model.model.layers)} MLP layers")
            # Check controller_allowed_experts on MLP layers
            mlp0 = model.model.layers[0].mlp
            print(f"  [VERIFY] MLP layer 0: controller_allowed_experts={getattr(mlp0, 'controller_allowed_experts', 'NOT_SET')}")
        
        t0 = time.time()
        term_str = f"term={args.termination_mode}" if args.termination_mode == "sampling" else f"term=thresh{args.termination_threshold}"
        desc_str = f"Controller{lora_str} (step {step}, {term_str})"
        controller_acc, acc_by_cat, _, avg_switch_rate = evaluate_samples(
            model, tokenizer, samples_by_category,
            desc=desc_str,
            controller_sampling=args.controller_sampling,
            q_based_selection=args.q_based_selection,
            q_selection_steps=args.q_selection_steps,
            q_selection_lr=args.q_selection_lr,
            q_selection_init_w=args.q_selection_init_w,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            termination_mode=args.termination_mode,
            termination_threshold=args.termination_threshold,
            joint_controller=joint_ctrl if bool(args.joint_option) else None,
            joint_option_k=args.joint_option_k,
        )
        controller_time = time.time() - t0
        
        result_key = f"controller_lora_step_{step}" if lora_path else f"controller_step_{step}"
        results[result_key] = controller_acc
        results[f"{result_key}_by_category"] = acc_by_cat
        if avg_switch_rate is not None:
            results[f"{result_key}_switch_rate"] = avg_switch_rate
        
        sampling_str = "(sampling)" if args.controller_sampling else ("(Q-based)" if args.q_based_selection else "(greedy)")
        term_info = f"[term={args.termination_mode}" + (f",thresh={args.termination_threshold}]" if args.termination_mode == "threshold" else "]")
        sr_str = f" | Switch rate: {avg_switch_rate:.4f}" if avg_switch_rate is not None else ""
        print(f"\nController{lora_str} (Step {step}) {sampling_str} {term_info} Accuracy: {controller_acc:.2%}{sr_str}")
        print(f"Time: {controller_time:.1f}s ({controller_time/total_samples:.2f}s per sample)")
        
        print("\nAccuracy by category:")
        for cat, acc in sorted(acc_by_cat.items()):
            print(f"  {cat}: {acc:.2%}")
        
        del model
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, value in results.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.2%}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Save results
    if args.num_shards > 1:
        output_name = f"math_eval_{args.method}_shard{args.shard_idx}.json"
    else:
        output_name = f"math_eval_{args.method}.json"
    output_path = os.path.join(args.output_dir, output_name)
    
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "config": config_dict,
            "hindsight_experts_per_layer": hindsight_experts_per_layer,
            "total_time_seconds": total_time,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
