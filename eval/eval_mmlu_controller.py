#!/usr/bin/env python
"""Evaluate controller checkpoints on MMLU and MMMLU.

Uses the same controller loading pattern as eval_math.py:
  1. load_model(controller_enabled=True, controller_type, lora_path)
  2. load_controller_checkpoint(model, checkpoint_path, controller_type)
  3. generate_response(model, tokenizer, prompt, controller_runtime=...)

MMLU: English multiple-choice (57 categories, parquet format)
MMMLU: Multilingual MMLU (14 languages, CSV format)
"""
import os, sys, json, random, re, argparse, time, glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_math import load_model, load_controller_checkpoint, generate_response, RouterSwitchTracker

ANSWER_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(data_dir: str, num_samples: int = 200, seed: int = 42) -> List[dict]:
    """Load MMLU test questions from parquet files across all categories."""
    data_dir = Path(data_dir)
    all_questions = []

    categories = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in ("all", "auxiliary_train", ".git")
    ])
    print(f"[MMLU] Found {len(categories)} categories")

    for cat in categories:
        parquet_path = data_dir / cat / "test-00000-of-00001.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            all_questions.append({
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer_idx": int(row["answer"]),
                "answer_letter": ANSWER_LETTERS[int(row["answer"])],
                "subject": row["subject"],
            })

    print(f"[MMLU] Total questions: {len(all_questions)}")

    rng = random.Random(seed)
    if num_samples < len(all_questions):
        all_questions = rng.sample(all_questions, num_samples)
    else:
        rng.shuffle(all_questions)

    print(f"[MMLU] Selected {len(all_questions)} questions")
    subjects = {}
    for q in all_questions:
        subjects[q["subject"]] = subjects.get(q["subject"], 0) + 1
    print(f"[MMLU] Subjects represented: {len(subjects)}")
    return all_questions


def load_mmmlu(data_dir: str, num_samples: int = 200, seed: int = 42) -> List[dict]:
    """Load MMMLU test questions from CSV files across all languages."""
    test_dir = Path(data_dir) / "test"
    all_questions = []

    csv_files = sorted(test_dir.glob("mmlu_*.csv"))
    print(f"[MMMLU] Found {len(csv_files)} language files")

    for csv_path in csv_files:
        lang = csv_path.stem.replace("mmlu_", "")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            answer_letter = str(row["Answer"]).strip().upper()
            all_questions.append({
                "question": str(row["Question"]),
                "choices": [str(row["A"]), str(row["B"]), str(row["C"]), str(row["D"])],
                "answer_idx": ANSWER_LETTERS.index(answer_letter) if answer_letter in ANSWER_LETTERS else 0,
                "answer_letter": answer_letter,
                "subject": str(row.get("Subject", "unknown")),
                "language": lang,
            })

    print(f"[MMMLU] Total questions across all languages: {len(all_questions)}")

    rng = random.Random(seed)
    if num_samples < len(all_questions):
        all_questions = rng.sample(all_questions, num_samples)
    else:
        rng.shuffle(all_questions)

    print(f"[MMMLU] Selected {len(all_questions)} questions")
    langs = {}
    for q in all_questions:
        langs[q.get("language", "?")] = langs.get(q.get("language", "?"), 0) + 1
    print(f"[MMMLU] Languages represented: {dict(sorted(langs.items()))}")
    return all_questions


def format_mmlu_prompt(question: dict) -> str:
    """Format a multiple-choice question as a prompt, using <answer> tags like MATH eval."""
    q = question["question"]
    choices = question["choices"]
    choice_lines = "\n".join(f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(choices))

    prompt = f"""Answer the following multiple-choice question. Show your reasoning step by step.

CRITICAL: You MUST wrap your final answer letter in <answer></answer> tags. Do NOT just write "The answer is A" - you must write "The answer is <answer>A</answer>".

Example of correct format: "Therefore, the answer is <answer>B</answer>."

Question: {q}

{choice_lines}

Answer (remember to use <answer></answer> tags for your final answer letter):"""
    return prompt


def extract_answer_letter(response: str) -> str:
    """Extract the answer letter from <answer>...</answer> tags, matching MATH eval pattern."""
    pattern = r'<answer>((?:(?!<answer>).)*?)</answer>'
    matches = list(re.finditer(pattern, response, re.DOTALL))

    if matches:
        content = matches[-1].group(1).strip().upper()
        for letter in ANSWER_LETTERS:
            if letter in content:
                return letter

    return ""


def evaluate_mmlu(
    model, tokenizer, questions: List[dict], benchmark_name: str = "MMLU",
    max_new_tokens: int = 512, temperature: float = 0.5,
    controller_sampling: bool = False, q_based_selection: bool = False,
    q_selection_init_w: float = 2.0, termination_mode: str = "sampling",
    termination_threshold: float = 0.5,
    router_switch_tracker: Optional['RouterSwitchTracker'] = None,
) -> dict:
    """Evaluate model on MMLU/MMMLU questions."""
    correct = 0
    total = 0
    by_subject = {}
    switch_rates = []

    for idx, q in enumerate(tqdm(questions, desc=benchmark_name)):
        prompt = format_mmlu_prompt(q)
        response, switch_rate = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
            controller_sampling=controller_sampling,
            q_based_selection=q_based_selection,
            q_selection_init_w=q_selection_init_w,
            termination_mode=termination_mode,
            termination_threshold=termination_threshold,
            router_switch_tracker=router_switch_tracker,
        )
        if switch_rate is not None:
            switch_rates.append(switch_rate)

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

        sr_str = f" | Switch rate: {switch_rate:.4f}" if switch_rate is not None else ""
        lang_str = f" | Lang: {q['language']}" if "language" in q else ""
        print(f"\n{'='*60}", flush=True)
        print(f"[{benchmark_name}] Sample {idx+1}/{len(questions)} | Subject: {subj}{lang_str}{sr_str}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"PROMPT:\n{prompt}", flush=True)
        print(f"\n--- MODEL RESPONSE ---", flush=True)
        print(response, flush=True)
        print(f"\n--- END RESPONSE ---", flush=True)
        print(f"Predicted: {predicted} | Correct: {q['answer_letter']} | "
              f"{'PASS' if is_correct else 'FAIL'}", flush=True)
        avg_sr_str = f" | Avg switch rate: {sum(switch_rates)/len(switch_rates):.4f}" if switch_rates else ""
        print(f"Running: {correct}/{total}={correct/total:.2%}{avg_sr_str}", flush=True)

    accuracy = correct / max(total, 1)
    subject_acc = {
        subj: data["correct"] / max(data["total"], 1)
        for subj, data in sorted(by_subject.items())
    }
    avg_switch_rate = sum(switch_rates) / len(switch_rates) if switch_rates else None
    if avg_switch_rate is not None:
        print(f"\n[{benchmark_name}] Average switch rate: {avg_switch_rate:.4f}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_subject": subject_acc,
        "switch_rate": avg_switch_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/scratch/gpfs/HENDERSON/transformer_cache/gpt-oss-20b")
    parser.add_argument("--benchmark", type=str, required=True, choices=["mmlu", "mmmlu"])
    parser.add_argument("--mmlu-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/mmlu")
    parser.add_argument("--mmmlu-dir", type=str,
                        default="/scratch/gpfs/HENDERSON/zs7353/rl_moe/MMMLU")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--base-model", action="store_true",
                        help="Evaluate base model (no controller, track router switch rate)")
    parser.add_argument("--cache-k", type=int, nargs="+", default=None,
                        help="Cache sizes for base model switch rate (e.g., --cache-k 8 16)")
    parser.add_argument("--controller-type", type=str, default="activation",
                        choices=["rnn", "activation"])
    parser.add_argument("--controller-allowed-experts", type=int, default=16)
    parser.add_argument("--with-lora", action="store_true")
    parser.add_argument("--q-based-selection", action="store_true")
    parser.add_argument("--q-selection-init-w", type=float, default=1.0)
    parser.add_argument("--controller-sampling", action="store_true")
    parser.add_argument("--termination-mode", type=str, default="sampling",
                        choices=["sampling", "threshold"])
    parser.add_argument("--termination-threshold", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate for MMLU reasoning + answer")
    parser.add_argument("--temperature", type=float, default=0.5)
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

    if args.base_model:
        cache_k_list = args.cache_k if args.cache_k else [8, 16]
        max_cache_k = max(cache_k_list)

        print(f"\nLoading base model from {args.model_path} (no controller)")
        print(f"  Cache k values: {cache_k_list}")
        model, tokenizer = load_model(args.model_path, controller_enabled=False)

        switch_tracker = RouterSwitchTracker(model, store_top_n=max_cache_k)
        switch_tracker.install_hooks()
        print(f"  Installed router switch tracker on {len(switch_tracker.layer_selections)} layers")

        t0 = time.time()
        benchmark_name = args.benchmark.upper()

        correct, total = 0, 0
        by_subject = {}
        switch_rates_by_k = {ck: [] for ck in cache_k_list}

        for idx, q in enumerate(tqdm(questions, desc=benchmark_name)):
            prompt = format_mmlu_prompt(q)

            # Tokenize to get prompt_len (same logic as generate_response)
            fmt_prompt = prompt
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                fmt_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
            prompt_inputs = tokenizer(fmt_prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            response, _ = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )

            sr_parts = []
            for ck in cache_k_list:
                csr = switch_tracker.compute_cache_switch_rate(ck, prompt_len)
                if csr is not None:
                    switch_rates_by_k[ck].append(csr)
                    sr_parts.append(f"k={ck}: {csr:.4f}")
            switch_tracker.reset()

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

            sr_str = " | Switch: " + ", ".join(sr_parts) if sr_parts else ""
            lang_str = f" | Lang: {q['language']}" if "language" in q else ""
            print(f"\n{'='*60}", flush=True)
            print(f"[{benchmark_name}] Sample {idx+1}/{len(questions)} | Subject: {subj}{lang_str}{sr_str}", flush=True)
            print(f"PROMPT:\n{prompt}", flush=True)
            print(f"\n--- MODEL RESPONSE ---", flush=True)
            print(response, flush=True)
            print(f"--- END RESPONSE ---", flush=True)
            print(f"Predicted: {predicted} | Correct: {q['answer_letter']} | "
                  f"{'PASS' if is_correct else 'FAIL'}", flush=True)
            avg_parts = []
            for ck in cache_k_list:
                if switch_rates_by_k[ck]:
                    avg_parts.append(f"k={ck}: {sum(switch_rates_by_k[ck])/len(switch_rates_by_k[ck]):.4f}")
            avg_sr_str = f" | Avg switch: {', '.join(avg_parts)}" if avg_parts else ""
            print(f"Running: {correct}/{total}={correct/total:.2%}{avg_sr_str}", flush=True)

        switch_tracker.remove_hooks()
        elapsed = time.time() - t0

        accuracy = correct / max(total, 1)
        subject_acc = {
            subj: data["correct"] / max(data["total"], 1)
            for subj, data in sorted(by_subject.items())
        }
        results = {
            "accuracy": accuracy, "correct": correct, "total": total,
            "accuracy_by_subject": subject_acc,
        }
        for ck in cache_k_list:
            if switch_rates_by_k[ck]:
                avg_sr = sum(switch_rates_by_k[ck]) / len(switch_rates_by_k[ck])
                results[f"switch_rate_k{ck}"] = avg_sr
                results[f"switch_rates_k{ck}"] = switch_rates_by_k[ck]

        tag = f"_{args.tag}" if args.tag else ""
        cache_tag = "_".join(f"k{ck}" for ck in cache_k_list)
        output_file = os.path.join(args.output_dir, f"{args.benchmark}_base_cache_sr_{cache_tag}{tag}.json")
        with open(output_file, "w") as f:
            json.dump({"results": results, "config": vars(args),
                        "elapsed_seconds": elapsed}, f, indent=2)

        print(f"\n{'='*70}")
        print(f"RESULTS ({benchmark_name}, base model):")
        print(f"{'='*70}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        for ck in cache_k_list:
            if switch_rates_by_k[ck]:
                avg_sr = sum(switch_rates_by_k[ck]) / len(switch_rates_by_k[ck])
                print(f"  Cache switch rate (k={ck}): {avg_sr:.4f}")
        print(f"\nAccuracy by subject:")
        for subj, acc in sorted(subject_acc.items()):
            print(f"  {subj}: {acc:.2%}")
        print(f"\nTime: {elapsed/60:.1f} min")
        print(f"Saved to {output_file}")
    else:
        if args.checkpoint_dir is None:
            print("ERROR: --checkpoint-dir required for controller evaluation (use --base-model for base model)")
            sys.exit(1)

        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"activation_controller_step_{args.step}.pt"
        )
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"controller_step_{args.step}.pt"
            )
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        lora_path = None
        if args.with_lora:
            lora_dir = os.path.join(args.checkpoint_dir, f"lora_step_{args.step}")
            if os.path.exists(lora_dir):
                lora_path = lora_dir
                print(f"Found LoRA adapter at {lora_dir}")
            else:
                print(f"WARNING: --with-lora specified but no adapter at {lora_dir}")

        print(f"\nLoading model from {args.model_path}")
        print(f"  controller_type={args.controller_type}")
        print(f"  controller_allowed_experts={args.controller_allowed_experts}")
        print(f"  with_lora={args.with_lora}, lora_path={lora_path}")
        model, tokenizer = load_model(
            args.model_path,
            controller_enabled=True,
            lora_path=lora_path,
            controller_type=args.controller_type,
            controller_allowed_experts=args.controller_allowed_experts,
        )
        load_controller_checkpoint(model, checkpoint_path,
                                   controller_type=args.controller_type)

        n_ctrl = sum(1 for layer in model.model.layers
                     if hasattr(layer.mlp, 'controller') and layer.mlp.controller is not None)
        print(f"  Controllers present: {n_ctrl}/{len(model.model.layers)} layers")

        t0 = time.time()
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            controller_sampling=args.controller_sampling,
            q_based_selection=args.q_based_selection,
            q_selection_init_w=args.q_selection_init_w,
            termination_mode=args.termination_mode,
            termination_threshold=args.termination_threshold,
        )

        benchmark_name = args.benchmark.upper()
        results = evaluate_mmlu(model, tokenizer, questions,
                                benchmark_name=benchmark_name, **gen_kwargs)

        elapsed = time.time() - t0
        tag = f"_{args.tag}" if args.tag else ""
        output_file = os.path.join(args.output_dir, f"{args.benchmark}{tag}.json")
        with open(output_file, "w") as f:
            json.dump({"results": results, "config": vars(args),
                        "elapsed_seconds": elapsed}, f, indent=2)

        print(f"\n{'='*70}")
        print(f"RESULTS ({benchmark_name}, controller step {args.step}):")
        print(f"{'='*70}")
        print(f"  Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        if results.get("switch_rate") is not None:
            print(f"  Switch rate: {results['switch_rate']:.4f}")
        print(f"\nAccuracy by subject:")
        for subj, acc in sorted(results.get("accuracy_by_subject", {}).items()):
            print(f"  {subj}: {acc:.2%}")
        print(f"\nTime: {elapsed/60:.1f} min")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
