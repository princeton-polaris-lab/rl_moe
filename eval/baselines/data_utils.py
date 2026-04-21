"""
Data utilities for loading calibration data for expert pruning baselines.

Supports two modes:
1. "general" - Load raw C4 sequences (pretraining-style text)
2. "math" - Generate rollouts on MATH problems using the base model
"""

import os
import sys
import gzip
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from tqdm import tqdm

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_c4_sequences(
    c4_dir: str,
    num_sequences: int = 64,
    seq_length: int = 2048,
    tokenizer=None,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Load and tokenize sequences from C4 dataset.
    
    Follows NAEE paper: concatenate short documents to form longer sequences.
    
    Args:
        c4_dir: Path to C4 data directory (e.g., /scratch/gpfs/.../c4/en.noblocklist)
        num_sequences: Number of sequences to create (default 64)
        seq_length: Target sequence length in tokens (default 2048)
        tokenizer: Tokenizer to use
        seed: Random seed
        
    Returns:
        List of token tensors, each of shape (seq_length,)
    """
    random.seed(seed)
    
    c4_path = Path(c4_dir)
    if not c4_path.exists():
        raise FileNotFoundError(f"C4 directory not found: {c4_dir}")
    
    # Find all json.gz files
    shard_files = sorted(c4_path.glob("*.json.gz"))
    if not shard_files:
        # Try looking in subdirectory
        shard_files = sorted(c4_path.glob("*/*.json.gz"))
    
    if not shard_files:
        raise FileNotFoundError(f"No C4 shard files found in {c4_dir}")
    
    print(f"[C4] Found {len(shard_files)} shard files")
    
    # Read documents from the first shard (should be more than enough)
    documents = []
    target_chars = num_sequences * seq_length * 5  # Rough estimate: 5 chars per token
    
    for shard_file in shard_files:
        if sum(len(d) for d in documents) >= target_chars:
            break
        
        print(f"[C4] Reading from {shard_file.name}...")
        with gzip.open(shard_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    documents.append(doc['text'])
                    if sum(len(d) for d in documents) >= target_chars:
                        break
                except (json.JSONDecodeError, KeyError) as e:
                    continue
    
    print(f"[C4] Loaded {len(documents)} documents")
    
    # Concatenate documents into long sequences
    all_text = " ".join(documents)
    
    # Tokenize all text at once
    print(f"[C4] Tokenizing {len(all_text)} characters...")
    all_tokens = tokenizer.encode(all_text, add_special_tokens=False)
    print(f"[C4] Got {len(all_tokens)} tokens")
    
    # Split into sequences
    sequences = []
    for i in range(0, len(all_tokens) - seq_length + 1, seq_length):
        if len(sequences) >= num_sequences:
            break
        seq = torch.tensor(all_tokens[i:i + seq_length], dtype=torch.long)
        sequences.append(seq)
    
    if len(sequences) < num_sequences:
        print(f"[C4] Warning: Only got {len(sequences)} sequences (requested {num_sequences})")
    
    print(f"[C4] Created {len(sequences)} sequences of length {seq_length}")
    return sequences


def load_math_problems(
    data_dir: Path,
    num_problems: int = 64,
    seed: int = 42,
    levels: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load MATH problems for calibration.
    
    Args:
        data_dir: Path to hendrycks_math directory
        num_problems: Number of problems to sample
        seed: Random seed
        levels: Optional difficulty level filter
        
    Returns:
        List of problem dicts with 'problem', 'solution', 'answer' keys
    """
    import pandas as pd
    random.seed(seed)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"MATH data directory not found: {data_dir}")
    
    # Get all category directories
    category_dirs = [
        d for d in sorted(data_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    all_samples = []
    
    for cat_dir in tqdm(category_dirs, desc="Loading MATH categories"):
        # Load train split for calibration
        train_files = list(cat_dir.glob("train-*.parquet"))
        if not train_files:
            # Fall back to test if no train
            train_files = list(cat_dir.glob("test-*.parquet"))
        if not train_files:
            continue
        
        try:
            df = pd.read_parquet(train_files[0])
        except Exception as e:
            print(f"  Warning: Could not load {train_files[0]}: {e}")
            continue
        
        if not all(col in df.columns for col in ["problem", "solution"]):
            continue
        
        # Filter by level if specified
        if levels is not None and "level" in df.columns:
            df = df[df["level"].isin(levels)]
        
        for _, row in df.iterrows():
            # Extract answer from solution
            answer = extract_boxed_answer(row["solution"])
            all_samples.append({
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
                "level": row.get("level", "Unknown"),
                "type": row.get("type", cat_dir.name),
            })
    
    # Randomly sample
    if len(all_samples) > num_problems:
        all_samples = random.sample(all_samples, num_problems)
    
    print(f"[MATH] Loaded {len(all_samples)} problems for calibration")
    return all_samples


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in solution text."""
    import re
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


def generate_math_rollouts(
    model,
    tokenizer,
    problems: List[Dict],
    max_length: int = 2048,
    temperature: float = 0.7,
) -> List[torch.Tensor]:
    """
    Generate rollouts on MATH problems using the base model.
    
    This creates calibration sequences that represent the model's behavior
    on math problems (as used in task-specific calibration).
    
    Args:
        model: The model to generate with
        tokenizer: Tokenizer
        problems: List of problem dicts
        max_length: Maximum total sequence length (prompt + generation)
        temperature: Sampling temperature
        
    Returns:
        List of token tensors (full sequences including prompt)
    """
    sequences = []
    
    model.eval()
    
    for problem in tqdm(problems, desc="Generating MATH rollouts"):
        # Format prompt
        prompt = format_math_prompt(problem["problem"])
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length // 2)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[1]
        max_new_tokens = max_length - prompt_len
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Get full sequence
        full_seq = outputs[0]  # Shape: (seq_len,)
        
        # Pad or truncate to exact length
        if len(full_seq) < max_length:
            # Pad with pad token
            padding = torch.full(
                (max_length - len(full_seq),),
                tokenizer.pad_token_id,
                dtype=full_seq.dtype,
                device=full_seq.device
            )
            full_seq = torch.cat([full_seq, padding])
        else:
            full_seq = full_seq[:max_length]
        
        sequences.append(full_seq.cpu())
    
    print(f"[MATH] Generated {len(sequences)} rollout sequences")
    return sequences


def format_math_prompt(problem: str) -> str:
    """Format a MATH problem as a prompt."""
    return f"""Solve the following math problem. Show your reasoning step by step.

CRITICAL: You MUST wrap your final answer in <answer></answer> tags.

Problem: {problem}

Solution:"""


def load_nemotron_prompts(
    data_dir: str = "/scratch/gpfs/HENDERSON/zs7353/Nemotron-Post-Training-Dataset-v2/data",
    num_prompts: int = 128,
    seed: int = 42,
) -> List[str]:
    """
    Load user prompts from the Nemotron Post-Training Dataset.
    
    Pools prompts from ALL categories (chat, code, math, stem, multilingual_*)
    and randomly samples from the entire pool, matching training behavior.
    
    Args:
        data_dir: Path to Nemotron data directory
        num_prompts: Number of prompts to randomly sample
        seed: Random seed
    
    Returns:
        List of prompt strings
    """
    import pandas as pd
    import numpy as np
    
    rng = random.Random(seed)
    data_path = Path(data_dir)
    
    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Discover all category prefixes
    # Filenames are like: chat-00000-of-00012.parquet, multilingual_de-00003-of-00038.parquet
    # Category is everything before the first "-DIGITS-of-" pattern
    import re
    categories = {}
    for pf in parquet_files:
        m = re.match(r'^(.+?)-\d+-of-\d+', pf.stem)
        prefix = m.group(1) if m else pf.stem
        if prefix not in categories:
            categories[prefix] = []
        categories[prefix].append(pf)
    
    print(f"[Nemotron] Found {len(categories)} categories: {sorted(categories.keys())}")
    
    all_prompts = []
    for cat, files in sorted(categories.items()):
        df = pd.read_parquet(files[0])
        cat_count = 0
        
        for _, row in df.iterrows():
            msgs = row.get("messages", [])
            if msgs is None:
                continue
            for msg in msgs:
                if isinstance(msg, (dict, np.void)):
                    role = msg["role"] if isinstance(msg, dict) else msg[0]
                    content = msg["content"] if isinstance(msg, dict) else msg[1]
                    if role == "user" and isinstance(content, str) and len(content) > 20:
                        all_prompts.append(content)
                        cat_count += 1
                        break
            if cat_count >= 500:
                break
        
        print(f"[Nemotron] Pooled {cat_count} prompts from '{cat}'")
    
    print(f"[Nemotron] Total pool: {len(all_prompts)} prompts")
    rng.shuffle(all_prompts)
    all_prompts = all_prompts[:num_prompts]
    print(f"[Nemotron] Randomly sampled {len(all_prompts)} prompts")
    return all_prompts


def generate_nemotron_rollouts(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int = 2048,
    max_prompt_tokens: int = 2048,
    temperature: float = 0.5,
) -> List[torch.Tensor]:
    """
    Generate rollouts on Nemotron prompts using the base model.
    
    Total sequence (prompt + response) is capped at max_length, matching
    the paper's approach of using fixed-length 2048-token sequences.
    max_new_tokens = max_length - prompt_len for each prompt.
    
    Args:
        model: The model to generate with
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_length: Maximum total sequence length (prompt + response)
        max_prompt_tokens: Maximum prompt length in tokens (skip longer prompts)
        temperature: Sampling temperature
    
    Returns:
        List of token tensors, each of shape (max_length,)
    """
    sequences = []
    model.eval()
    skipped = 0
    
    for prompt in tqdm(prompts, desc="Generating Nemotron rollouts"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[1]
        if prompt_len >= max_prompt_tokens:
            skipped += 1
            continue
        
        max_new_tokens = max_length - prompt_len
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        full_seq = outputs[0]
        
        if len(full_seq) < max_length:
            padding = torch.full(
                (max_length - len(full_seq),),
                tokenizer.pad_token_id,
                dtype=full_seq.dtype,
                device=full_seq.device
            )
            full_seq = torch.cat([full_seq, padding])
        else:
            full_seq = full_seq[:max_length]
        
        sequences.append(full_seq.cpu())
    
    if skipped:
        print(f"[Nemotron] Skipped {skipped} prompts exceeding {max_prompt_tokens} tokens")
    print(f"[Nemotron] Generated {len(sequences)} rollout sequences")
    return sequences


def get_calibration_data(
    mode: str,
    model,
    tokenizer,
    num_sequences: int = 64,
    seq_length: int = 2048,
    c4_dir: str = "/scratch/gpfs/HENDERSON/zs7353/rl_moe/c4/en.noblocklist",
    math_dir: str = "/scratch/gpfs/HENDERSON/zs7353/rl_moe/hendrycks_math",
    seed: int = 42,
    cache_dir: str = "/scratch/gpfs/HENDERSON/zs7353/rl_moe/eval/baselines/cache",
    use_cache: bool = True,
) -> Tuple[List[torch.Tensor], Optional[List[Dict]]]:
    """
    Get calibration data for expert pruning.
    
    Args:
        mode: "general" for C4, "math" for MATH rollouts, or "nemotron" for Nemotron rollouts
        model: Model (needed for MATH rollouts)
        tokenizer: Tokenizer
        num_sequences: Number of sequences
        seq_length: Sequence length
        c4_dir: Path to C4 data
        math_dir: Path to MATH data
        seed: Random seed
        cache_dir: Directory to cache calibration data
        use_cache: Whether to use cached data if available
        
    Returns:
        (sequences, problems_or_None)
        - sequences: List of token tensors
        - problems: List of problem dicts (only for math mode, None for general)
    """
    import pickle
    
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filename based on parameters
    cache_file = os.path.join(
        cache_dir,
        f"calibration_{mode}_n{num_sequences}_len{seq_length}_seed{seed}.pkl"
    )
    
    # Try loading from cache
    if use_cache and os.path.exists(cache_file):
        print(f"[CACHE] Loading calibration data from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            sequences = cached_data['sequences']
            problems = cached_data.get('problems', None)
            print(f"[CACHE] Loaded {len(sequences)} sequences from cache")
            return sequences, problems
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}")
            print(f"[CACHE] Regenerating calibration data...")
    
    # Generate fresh calibration data
    if mode == "general":
        sequences = load_c4_sequences(
            c4_dir=c4_dir,
            num_sequences=num_sequences,
            seq_length=seq_length,
            tokenizer=tokenizer,
            seed=seed,
        )
        problems = None
    
    elif mode == "math":
        problems = load_math_problems(
            data_dir=Path(math_dir),
            num_problems=num_sequences,
            seed=seed,
        )
        sequences = generate_math_rollouts(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            max_length=seq_length,
        )
    
    elif mode == "nemotron":
        prompts = load_nemotron_prompts(
            num_prompts=num_sequences,
            seed=seed,
        )
        sequences = generate_nemotron_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=seq_length,
            max_prompt_tokens=seq_length,
            temperature=0.5,
        )
        problems = None
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'general', 'math', or 'nemotron'")
    
    # Save to cache
    if use_cache:
        print(f"[CACHE] Saving calibration data to {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'sequences': sequences,
                    'problems': problems,
                    'mode': mode,
                    'num_sequences': num_sequences,
                    'seq_length': seq_length,
                    'seed': seed,
                }, f)
            print(f"[CACHE] Saved {len(sequences)} sequences to cache")
        except Exception as e:
            print(f"[CACHE] Failed to save cache: {e}")
    
    return sequences, problems
