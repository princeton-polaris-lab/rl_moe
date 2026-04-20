#!/usr/bin/env python
"""Evaluate expert pruning baselines (frequency/reconstruction/random/wanda) on MMLU and MMMLU.

For freq/recon/random: loads expert selections from JSON, patches router, evaluates.
For wanda: loads model, runs wanda calibration + pruning, then evaluates.

Reuses MMLU/MMMLU data loading and evaluation logic from eval_mmlu_controller.py.
"""
import os, sys, json, random, argparse, time
from pathlib import Path
from contextlib import nullcontext
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "baselines"))

from eval_math import load_model, patch_router_for_fixed_experts
from eval_mmlu_controller import (
    load_mmlu, load_mmmlu, format_mmlu_prompt, extract_answer_letter, ANSWER_LETTERS,
)


def evaluate_mmlu_baseline(
    model, tokenizer, questions, benchmark_name="MMLU",
    max_new_tokens=2048, temperature=0.5,
):
    """Evaluate model on MMLU/MMMLU (no controller, direct generation)."""
    correct = total = 0
    by_subject = {}

    for idx, q in enumerate(tqdm(questions, desc=benchmark_name)):
        prompt = format_mmlu_prompt(q)

        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted = extract_answer_letter(response)
        is_correct = (predicted == q["answer_letter"])
        if is_correct:
            correct += 1
        total += 1

        subj = q.get("subject", "unknown")
        if subj not in by_subject:
            by_subject[subj] = {"correct": 0, "total": 0}
        by_subject[subj]["total"] += 1
        if is_correct:
            by_subject[subj]["correct"] += 1

        lang_str = f" | Lang: {q['language']}" if "language" in q else ""
        print(f"\n{'='*60}", flush=True)
        print(f"[{benchmark_name}] Sample {idx+1}/{len(questions)} | Subject: {subj}{lang_str}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"PROMPT:\n{prompt}", flush=True)
        print(f"\n--- MODEL RESPONSE ---", flush=True)
        print(response, flush=True)
        print(f"\n--- END RESPONSE ---", flush=True)
        print(f"Predicted: {predicted} | Correct: {q['answer_letter']} | "
              f"{'PASS' if is_correct else 'FAIL'}", flush=True)
        print(f"Running: {correct}/{total}={correct/total:.2%}", flush=True)

    accuracy = correct / max(total, 1)
    subject_acc = {
        subj: data["correct"] / max(data["total"], 1)
        for subj, data in sorted(by_subject.items())
    }
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "accuracy_by_subject": subject_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/scratch/gpfs/HENDERSON/transformer_cache/gpt-oss-20b")
    parser.add_argument("--benchmark", type=str, required=True, choices=["mmlu", "mmmlu"])
    parser.add_argument("--mmlu-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/mmlu")
    parser.add_argument("--mmmlu-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/MMMLU")
    parser.add_argument("--method", type=str, required=True,
                        choices=["frequency", "reconstruction", "random", "wanda", "base"])
    parser.add_argument("--expert-selections", type=str, default=None,
                        help="Path to expert selection JSON (for freq/recon/random)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num-calibration", type=int, default=128,
                        help="Number of calibration sequences (for wanda)")
    parser.add_argument("--wanda-structured-n", type=int, default=1)
    parser.add_argument("--wanda-structured-m", type=int, default=4)
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/eval/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark == "mmlu":
        questions = load_mmlu(args.mmlu_dir, args.num_samples, args.seed)
    else:
        questions = load_mmmlu(args.mmmlu_dir, args.num_samples, args.seed)

    print(f"\nLoading model from {args.model_path}")
    model, tokenizer = load_model(args.model_path, controller_enabled=False)

    if args.method == "base":
        print("Running base model (no pruning, full experts)")
        ctx = nullcontext()
    elif args.method == "wanda":
        from baselines.data_utils import get_calibration_data
        from baselines.wanda_pruning import run_wanda_pruning

        print(f"Getting Nemotron calibration data ({args.num_calibration} sequences)...")
        sequences, _ = get_calibration_data(
            mode="nemotron", model=model, tokenizer=tokenizer,
            num_sequences=args.num_calibration, seq_length=2048,
            seed=args.seed,
        )

        print(f"Applying wanda structured {args.wanda_structured_n}:{args.wanda_structured_m} pruning...")
        model = run_wanda_pruning(
            model=model, sequences=sequences,
            sparsity_type="structured",
            sparsity_ratio=0.5,
            structured_n=args.wanda_structured_n,
            structured_m=args.wanda_structured_m,
            batch_size=4,
        )
        ctx = nullcontext()
    else:
        if not args.expert_selections:
            print("ERROR: --expert-selections required for freq/recon/random")
            sys.exit(1)
        with open(args.expert_selections) as f:
            sel_data = json.load(f)
        expert_selections = {int(k): v for k, v in sel_data["expert_selections"].items()}
        print(f"Loaded expert selections: {len(expert_selections)} layers, top_k={args.top_k}")
        ctx = patch_router_for_fixed_experts(model, expert_selections, args.top_k)

    t0 = time.time()
    benchmark_name = args.benchmark.upper()
    with ctx:
        results = evaluate_mmlu_baseline(
            model, tokenizer, questions,
            benchmark_name=benchmark_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    elapsed = time.time() - t0
    tag = f"_{args.tag}" if args.tag else ""
    output_file = os.path.join(args.output_dir, f"{args.benchmark}{tag}.json")
    with open(output_file, "w") as f:
        json.dump({"results": results, "config": vars(args),
                    "elapsed_seconds": elapsed}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS ({benchmark_name}, {args.method}):")
    print(f"{'='*70}")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"\nTime: {elapsed/60:.1f} min")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
