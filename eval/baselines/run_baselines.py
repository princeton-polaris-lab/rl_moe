#!/usr/bin/env python
"""
Run expert pruning baselines for evaluation.

Supports two modes:
1. "general" - Calibration on C4 sequences (task-agnostic)
2. "math" - Calibration on MATH problem rollouts (task-specific)

Methods implemented:
- frequency: Count expert activations, keep top-k most frequent
- reconstruction: Greedy addition minimizing reconstruction loss
- wanda: Canonical weight pruning (unstructured 50% or 2:4 structured)
- random: Random expert selection
- eep: Evolutionary search using accuracy as fitness (math mode only)
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import get_calibration_data, load_math_problems
from frequency_pruning import run_frequency_pruning
from reconstruction_pruning import run_reconstruction_pruning
from eep_pruning import run_eep_pruning
from wanda_pruning import run_wanda_pruning

# Import evaluation utilities from parent
from eval_math import (
    load_model, load_math_samples, evaluate_samples,
    patch_router_for_fixed_experts, format_math_prompt,
    extract_answer_tags, generate_response
)
from is_equiv import is_equiv


def save_expert_selections(
    selections: Dict[int, List[int]],
    output_path: str,
    method: str,
    mode: str,
    metadata: Optional[Dict] = None,
):
    """Save expert selections to JSON file."""
    data = {
        "method": method,
        "mode": mode,
        "expert_selections": {str(k): v for k, v in selections.items()},
        "metadata": metadata or {},
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved expert selections to {output_path}")


def load_expert_selections(path: str) -> Dict[int, List[int]]:
    """Load expert selections from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    selections = {int(k): v for k, v in data["expert_selections"].items()}
    return selections


def evaluate_with_experts(
    model,
    tokenizer,
    expert_selections: Dict[int, List[int]],
    samples_by_category: Dict[str, List[dict]],
    method_name: str,
    top_k: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 0.5,
) -> float:
    """
    Evaluate model with fixed expert selections.
    
    Returns:
        Accuracy
    """
    correct = 0
    total = 0
    
    all_samples = [(cat, sample) for cat, samples in samples_by_category.items() 
                   for sample in samples]
    
    # One-time summary of expert selections used for evaluation
    num_layers_with_sel = len(expert_selections)
    expert_counts = [len(v) for v in expert_selections.values()]
    print(f"[EVAL-CHECK] {method_name}: {num_layers_with_sel} layers, "
          f"experts per layer: min={min(expert_counts)}, max={max(expert_counts)}, "
          f"layer0={sorted(expert_selections.get(0, []))}", flush=True)
    
    with patch_router_for_fixed_experts(model, expert_selections, top_k):
        # Verify patching: check that excluded experts have large negative bias
        import torch.nn.functional as F
        for name, module in model.named_modules():
            if hasattr(module, 'compute_router_logits') and hasattr(module, 'bias'):
                import re as _re
                _m = _re.search(r'layers\.(\d+)', name)
                if _m and int(_m.group(1)) == 0:
                    n_masked = (module.bias.data < -1e6).sum().item()
                    n_total = module.bias.data.shape[0]
                    print(f"[EVAL-CHECK] Layer 0 router: {n_masked}/{n_total} experts masked out", flush=True)
                    break

        for idx, (cat, sample) in enumerate(tqdm(all_samples, desc=f"Evaluating {method_name}")):
            prompt, correct_answer = format_math_prompt(sample)
            
            response, _ = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            predicted = extract_answer_tags(response)
            
            is_correct = is_equiv(predicted, correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            # Print full response for every sample (no truncation)
            print(f"\n{'='*60}", flush=True)
            print(f"[{method_name}] Sample {idx+1}/{len(all_samples)} | Category: {cat}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"PROBLEM:\n{sample.get('problem', 'N/A')}", flush=True)
            print(f"\n--- MODEL RESPONSE ---", flush=True)
            print(response, flush=True)
            print(f"\n--- END RESPONSE ---", flush=True)
            print(f"Correct answer: {correct_answer}", flush=True)
            print(f"Extracted answer: {predicted}", flush=True)
            print(f"Result: {'CORRECT' if is_correct else 'WRONG'}", flush=True)
            print(f"Running accuracy: {correct}/{total} = {correct/total:.2%}", flush=True)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_model_directly(
    model,
    tokenizer,
    samples_by_category: Dict[str, List[dict]],
    method_name: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.5,
) -> float:
    """
    Evaluate model directly without expert selection patching.
    Used for Wanda (weight pruning) which keeps all experts but with sparse weights.
    
    Returns:
        Accuracy
    """
    correct = 0
    total = 0
    
    all_samples = [(cat, sample) for cat, samples in samples_by_category.items() 
                   for sample in samples]
    
    for idx, (cat, sample) in enumerate(tqdm(all_samples, desc=f"Evaluating {method_name}")):
        prompt, correct_answer = format_math_prompt(sample)
        
        response, _ = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        predicted = extract_answer_tags(response)
        
        is_correct = is_equiv(predicted, correct_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # Print full response for every sample (no truncation)
        print(f"\n{'='*60}", flush=True)
        print(f"[{method_name}] Sample {idx+1}/{len(all_samples)} | Category: {cat}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"PROBLEM:\n{sample.get('problem', 'N/A')}", flush=True)
        print(f"\n--- MODEL RESPONSE ---", flush=True)
        print(response, flush=True)
        print(f"\n--- END RESPONSE ---", flush=True)
        print(f"Correct answer: {correct_answer}", flush=True)
        print(f"Extracted answer: {predicted}", flush=True)
        print(f"Result: {'CORRECT' if is_correct else 'WRONG'}", flush=True)
        print(f"Running accuracy: {correct}/{total} = {correct/total:.2%}", flush=True)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run expert pruning baselines")
    
    # Model and data paths
    parser.add_argument("--model-path", type=str,
                        default="/scratch/gpfs/KOROLOVA/gpt-oss-20b")
    parser.add_argument("--c4-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/c4/en.noblocklist")
    parser.add_argument("--math-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math")
    
    # Mode and method
    parser.add_argument("--mode", type=str, choices=["general", "math", "nemotron"], default="math",
                        help="Calibration mode: 'general' (C4), 'math' (MATH rollouts), or 'nemotron' (Nemotron rollouts)")
    parser.add_argument("--method", type=str, 
                        choices=["frequency", "reconstruction", "wanda", "eep", "random", "all"],
                        default="all",
                        help="Which method to run")
    
    # Calibration parameters
    parser.add_argument("--num-calibration", type=int, default=128,
                        help="Number of calibration sequences/problems")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length for calibration")
    
    # Expert selection parameters
    parser.add_argument("--num-experts", type=int, default=16,
                        help="Number of experts to keep per layer")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-k experts activated per token")
    
    # EEP parameters
    parser.add_argument("--eep-iterations", type=int, default=200,
                        help="Total EEP iterations (split 40/160 pruning/merging)")
    parser.add_argument("--eep-population", type=int, default=30,
                        help="EEP population size")
    
    # Wanda parameters
    parser.add_argument("--wanda-sparsity-type", type=str, default="unstructured",
                        choices=["unstructured", "structured"],
                        help="Wanda sparsity type: 'unstructured' or 'structured' (N:M)")
    parser.add_argument("--wanda-sparsity-ratio", type=float, default=0.5,
                        help="Wanda sparsity ratio for unstructured (0.5 = 50%% pruned)")
    parser.add_argument("--wanda-structured-n", type=int, default=2,
                        help="For structured sparsity: keep N weights per block")
    parser.add_argument("--wanda-structured-m", type=int, default=4,
                        help="For structured sparsity: block size M (N:M pattern)")
    
    # Evaluation parameters
    parser.add_argument("--eval-samples-per-category", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for generation")
    
    # Output
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/eval/baselines/results")
    parser.add_argument("--seed", type=int, default=42)
    
    # Cache parameters
    parser.add_argument("--cache-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/eval/baselines/cache",
                        help="Directory to cache calibration data")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching (regenerate calibration data)")
    
    # Actions
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run calibration, don't evaluate")
    parser.add_argument("--load-selections", type=str, default=None,
                        help="Load expert selections from file instead of calibrating")
    parser.add_argument("--peft-checkpoint", type=str, default=None,
                        help="Path to PEFT/LoRA checkpoint to load on top of pruned model")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Load model
    print(f"\n{'='*70}")
    print("Loading model...")
    print(f"{'='*70}")
    model, tokenizer = load_model(args.model_path, controller_enabled=False)
    
    # Load PEFT adapter if specified (for fine-tuned pruned models)
    if args.peft_checkpoint:
        from peft import PeftModel
        print(f"\n[PEFT] Loading adapter from {args.peft_checkpoint}")
        # Fix base_model_name_or_path if it points to vast.ai path
        import json as _json
        adapter_cfg_path = os.path.join(args.peft_checkpoint, "adapter_config.json")
        with open(adapter_cfg_path) as _f:
            adapter_cfg = _json.load(_f)
        if adapter_cfg.get("base_model_name_or_path", "").startswith("/workspace"):
            adapter_cfg["base_model_name_or_path"] = args.model_path
            with open(adapter_cfg_path, "w") as _f:
                _json.dump(adapter_cfg, _f, indent=2)
            print(f"[PEFT] Fixed base_model_name_or_path to {args.model_path}")
        
        model = PeftModel.from_pretrained(model, args.peft_checkpoint)
        model.eval()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[PEFT] Loaded adapter: {trainable:,} trainable / {total:,} total params")
        print(f"[PEFT] Adapter has LoRA on: {adapter_cfg.get('target_modules', 'unknown')}")
    
    results = {}
    sequences = None  # Will be set by calibration if needed
    
    # Determine which methods to run
    if args.method == "all":
        if args.mode == "general":
            methods = ["frequency", "reconstruction", "wanda", "random"]  # No EEP for general mode
        else:
            methods = ["frequency", "reconstruction", "wanda", "eep", "random"]
    else:
        methods = [args.method]
    
    # Check if EEP is valid for mode
    if "eep" in methods and args.mode == "general":
        print("Warning: EEP requires 'math' mode (needs accuracy evaluation)")
        methods.remove("eep")
    
    # Get calibration data
    if args.load_selections is None:
        print(f"\n{'='*70}")
        print(f"Getting calibration data (mode={args.mode})")
        print(f"{'='*70}")
        
        sequences, problems = get_calibration_data(
            mode=args.mode,
            model=model,
            tokenizer=tokenizer,
            num_sequences=args.num_calibration,
            seq_length=args.seq_length,
            c4_dir=args.c4_dir,
            math_dir=args.math_dir,
            seed=args.seed,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
        )
        
        # Run calibration methods
        expert_selections = {}
        
        if "frequency" in methods:
            print(f"\n{'='*70}")
            print("Running FREQUENCY-based expert selection")
            print(f"{'='*70}")
            
            freq_selections = run_frequency_pruning(
                model=model,
                sequences=sequences,
                num_experts_to_keep=args.num_experts,
                batch_size=4,
            )
            expert_selections["frequency"] = freq_selections
            
            save_expert_selections(
                freq_selections,
                os.path.join(args.output_dir, f"frequency_{args.mode}_k{args.num_experts}_experts.json"),
                method="frequency",
                mode=args.mode,
                metadata={"num_calibration": args.num_calibration, "num_experts": args.num_experts},
            )
        
        if "reconstruction" in methods:
            print(f"\n{'='*70}")
            print("Running RECONSTRUCTION-based expert selection")
            print(f"{'='*70}")
            
            recon_selections = run_reconstruction_pruning(
                model=model,
                sequences=sequences,
                num_experts_to_keep=args.num_experts,
                batch_size=4,
            )
            expert_selections["reconstruction"] = recon_selections
            
            save_expert_selections(
                recon_selections,
                os.path.join(args.output_dir, f"reconstruction_{args.mode}_k{args.num_experts}_experts.json"),
                method="reconstruction",
                mode=args.mode,
                metadata={"num_calibration": args.num_calibration, "num_experts": args.num_experts},
            )
        
        # Note: Wanda is weight pruning, not expert selection
        # We handle it separately after this block
        
        if "eep" in methods:
            if problems is None:
                print("ERROR: EEP requires problems with answers (math mode)")
            else:
                print(f"\n{'='*70}")
                print("Running EEP expert selection")
                print(f"{'='*70}")
                
                eep_selections = run_eep_pruning(
                    model=model,
                    tokenizer=tokenizer,
                    problems=problems,
                    num_experts_to_keep=args.num_experts,
                    total_iterations=args.eep_iterations,
                    population_size=args.eep_population,
                )
                expert_selections["eep"] = eep_selections
                
                save_expert_selections(
                    eep_selections,
                    os.path.join(args.output_dir, f"eep_{args.mode}_k{args.num_experts}_experts.json"),
                    method="eep",
                    mode=args.mode,
                    metadata={
                        "num_calibration": args.num_calibration,
                        "iterations": args.eep_iterations,
                        "population": args.eep_population,
                        "num_experts": args.num_experts,
                    },
                )
        
        if "random" in methods:
            print(f"\n{'='*70}")
            print("Generating RANDOM expert selection")
            print(f"{'='*70}")
            
            import random as random_module
            random_module.seed(args.seed)
            
            # Get number of layers and experts from model
            num_layers = 24  # GPT-OSS-20B has 24 layers
            num_total_experts = 32  # GPT-OSS-20B has 32 experts
            
            # Generate random expert selections per layer
            random_selections = {}
            for layer_idx in range(num_layers):
                random_selections[layer_idx] = random_module.sample(
                    range(num_total_experts), args.num_experts
                )
            
            expert_selections["random"] = random_selections
            
            print(f"[RANDOM] Generated random selection: {args.num_experts} experts per layer")
            print(f"[RANDOM] Layer 0 example: {random_selections[0][:8]}...")
            
            save_expert_selections(
                random_selections,
                os.path.join(args.output_dir, f"random_{args.mode}_k{args.num_experts}_experts.json"),
                method="random",
                mode=args.mode,
                metadata={"seed": args.seed, "num_experts": args.num_experts},
            )
    
    else:
        # Load pre-computed selections
        print(f"\nLoading expert selections from {args.load_selections}")
        loaded = load_expert_selections(args.load_selections)
        expert_selections = {"loaded": loaded}
        methods = ["loaded"]
    
    if args.calibrate_only:
        print("\n--calibrate-only specified, skipping evaluation")
        return
    
    # Reload model fresh for evaluation to avoid any state corruption
    # from calibration forward passes (frequency/reconstruction run forward
    # passes with hooks that can leave residual state in the model).
    if args.method in ("frequency", "reconstruction"):
        print(f"\n{'='*70}")
        print("Reloading model fresh for evaluation...")
        print(f"{'='*70}")
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        model, tokenizer = load_model(args.model_path, controller_enabled=False)
    
    # Evaluation
    print(f"\n{'='*70}")
    print("Loading evaluation samples")
    print(f"{'='*70}")
    
    samples_by_category = load_math_samples(
        Path(args.math_dir),
        num_per_category=args.eval_samples_per_category,
        seed=args.seed,
    )
    
    total_samples = sum(len(v) for v in samples_by_category.values())
    print(f"Loaded {total_samples} evaluation samples")
    
    # Evaluate expert selection methods first
    for method in methods:
        if method == "wanda":
            continue  # Wanda is handled separately (weight pruning, not expert selection)
        if method not in expert_selections:
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating {method.upper()}")
        print(f"{'='*70}")
        
        selections = expert_selections[method]
        
        accuracy = evaluate_with_experts(
            model=model,
            tokenizer=tokenizer,
            expert_selections=selections,
            samples_by_category=samples_by_category,
            method_name=method,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        results[method] = accuracy
        print(f"\n{method.upper()} Accuracy: {accuracy:.2%}")
    
    # Handle Wanda weight pruning separately (modifies model in-place)
    if "wanda" in methods:
        print(f"\n{'='*70}")
        print("Running WANDA weight pruning")
        print(f"{'='*70}")
        
        # Ensure sequences are available for Wanda calibration
        if sequences is None:
            print("[WANDA] Loading/generating calibration sequences...")
            sequences, _ = get_calibration_data(
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
                num_sequences=args.num_calibration,
                seq_length=args.seq_length,
                c4_dir=args.c4_dir,
                math_dir=args.math_dir,
                seed=args.seed,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
        
        # Apply Wanda pruning to the model
        model = run_wanda_pruning(
            model=model,
            sequences=sequences,
            sparsity_type=args.wanda_sparsity_type,
            sparsity_ratio=args.wanda_sparsity_ratio,
            structured_n=args.wanda_structured_n,
            structured_m=args.wanda_structured_m,
            batch_size=4,
        )
        
        print(f"\n{'='*70}")
        print("Evaluating WANDA (all experts, sparse weights)")
        print(f"{'='*70}")
        
        # Evaluate on pruned model (all 32 experts, but with sparse weights)
        wanda_accuracy = evaluate_model_directly(
            model=model,
            tokenizer=tokenizer,
            samples_by_category=samples_by_category,
            method_name="wanda",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        results["wanda"] = wanda_accuracy
        print(f"\nWANDA Accuracy: {wanda_accuracy:.2%}")
        
        # Save Wanda pruning info
        wanda_info = {
            "sparsity_type": args.wanda_sparsity_type,
            "sparsity_ratio": args.wanda_sparsity_ratio,
            "structured_n": args.wanda_structured_n,
            "structured_m": args.wanda_structured_m,
            "accuracy": wanda_accuracy,
        }
        with open(os.path.join(args.output_dir, f"wanda_{args.mode}_k{args.num_experts}_results.json"), 'w') as f:
            json.dump(wanda_info, f, indent=2)
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for method, acc in results.items():
        print(f"  {method}: {acc:.2%}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"eval_results_{args.mode}_{args.method}_k{args.num_experts}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": results,
            "config": vars(args),
            "total_time_seconds": total_time,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
