"""
Analyze verifiability of boxed answers in Nemotron-Math-v2 dataset.

Computes:
- How many samples have verifiable boxed answers (parseable by math_verify or sympy)
- How many samples match ground truth (verified by math_verify or sympy)
- Sequence length distribution for samples matching ground truth

Usage:
    python analyze_verifiability.py [path_to_jsonl_file] [--max_samples N] [--use_sympy]
"""

import argparse
import json
import random
import re
from collections import Counter
from multiprocessing import cpu_count, Pool
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer

# Import sympy-based grader (will be used if --use_sympy is set)
import sys
sys.path.insert(0, "/home/devvrit03/tinker-cookbook-custom")
from tinker_cookbook.recipes.math_rl.math_grading import grade_answer as sympy_grade_answer

# --- CONFIGURATION ---
DEFAULT_FILE = "/home/devvrit03/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/f837702092f10dc3039d33e08cd1839f2a2f986a/data/medium.jsonl"
# DEFAULT_FILE = "/home/devvrit03/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/f837702092f10dc3039d33e08cd1839f2a2f986a/data/high.part_01.jsonl"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_WORKERS = min(cpu_count(), 32)

# Sequence length buckets (in tokens)
BUCKET_RANGES = [
    (0, 2000, "<2k"),
    (2000, 4000, "2k-4k"),
    (4000, 6000, "4k-6k"),
    (6000, 8000, "6k-8k"),
    (8000, 10000, "8k-10k"),
    (10000, 12000, "10k-12k"),
    (12000, 14000, "12k-14k"),
    (14000, 16000, "14k-16k"),
    (16000, 18000, "16k-18k"),
    (18000, float("inf"), ">18k"),
]

# Pattern to detect Python code blocks (indicates TIR usage)
PYTHON_CODE_PATTERN = re.compile(r'```python|```py|<\|python_start\|>|<tool_call>', re.IGNORECASE)

# Global tokenizer for worker processes
_tokenizer = None

# Global flag for using sympy (set in main, read in workers)
_use_sympy = False


def get_bucket_label(length: int) -> str:
    for low, high, label in BUCKET_RANGES:
        if low <= length < high:
            return label
    return ">18k"


def has_python_tir(text: str) -> bool:
    """Check if text contains Python TIR patterns."""
    return bool(PYTHON_CODE_PATTERN.search(text))


def extract_text(example: dict) -> str:
    """Extract all text from messages field.
    
    For Nemotron format, includes both 'reasoning_content' and 'content' fields
    since the full rollout is reasoning followed by the summarized answer.
    """
    messages = example.get("messages", [])
    text_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            # Include reasoning_content first (if present), then content
            reasoning = msg.get("reasoning_content", "")
            content = msg.get("content", "")
            if reasoning:
                text_parts.append(reasoning)
            if content:
                text_parts.append(content)
    return "\n".join(text_parts)


def should_filter_sample(example: dict) -> bool:
    """Check if sample should be filtered out (TIR or has tools field)."""
    # Check for tools field
    tools_val = example.get("tools") or example.get("tool")
    if tools_val:
        return True
    
    # Check for Python code blocks
    text = extract_text(example)
    if has_python_tir(text):
        return True
    
    return False


def extract_assistant_response(example: dict) -> Optional[str]:
    """Extract assistant response from messages."""
    messages = example.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def extract_ground_truth(example: dict) -> Optional[str]:
    """Extract ground truth answer from example."""
    for field in ["answer", "Answer", "ground_truth", "solution", "expected", "expected_answer"]:
        if field in example:
            return str(example[field])
    return None


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} expression."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def is_verifiable(boxed_content: str) -> bool:
    """Check if a boxed expression is verifiable using math_verify."""
    if not boxed_content:
        return False
    try:
        from math_verify import parse
        # Wrap in $ for proper parsing
        if not boxed_content.startswith("$"):
            boxed_content = f"${boxed_content}$"
        parsed = parse(boxed_content)
        return parsed is not None and len(parsed) > 0
    except Exception:
        return False


def matches_ground_truth(boxed_content: str, ground_truth: str) -> bool:
    """Check if a boxed expression matches the ground truth using math_verify."""
    if not boxed_content or not ground_truth:
        return False
    try:
        from math_verify import parse, verify
        # Wrap in $ for proper parsing
        if not boxed_content.startswith("$"):
            boxed_content = f"${boxed_content}$"
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        
        parsed_answer = parse(boxed_content)
        parsed_gt = parse(ground_truth)
        
        return verify(parsed_answer, parsed_gt)
    except Exception:
        return False


def matches_ground_truth_sympy(boxed_content: str, ground_truth: str) -> bool:
    """Check if a boxed expression matches the ground truth using sympy.
    
    Note: This function may hang on certain expressions. Use with pebble.ProcessPool
    which can timeout and kill stuck workers.
    """
    if not boxed_content or not ground_truth:
        return False
    
    # Skip very long expressions that may cause sympy to hang
    MAX_EXPR_LEN = 200
    if len(boxed_content) > MAX_EXPR_LEN or len(ground_truth) > MAX_EXPR_LEN:
        return False
    
    try:
        return sympy_grade_answer(boxed_content, ground_truth)
    except Exception:
        return False


def is_verifiable_sympy(boxed_content: str) -> bool:
    """Check if sympy can parse the expression (always returns True for non-empty content).
    
    With sympy approach, we just check if the content is non-empty.
    The actual verification happens in matches_ground_truth_sympy.
    """
    return boxed_content is not None and len(boxed_content.strip()) > 0


def process_line(line: str) -> Optional[dict]:
    """Process a single JSON line. Returns stats dict or None if filtered/invalid."""
    try:
        example = json.loads(line)
        
        # Check if should be filtered (TIR/tools)
        if should_filter_sample(example):
            return {"status": "filtered"}
        
        # Extract assistant response and ground truth
        assistant_response = extract_assistant_response(example)
        ground_truth = extract_ground_truth(example)
        text = extract_text(example)
        
        if not assistant_response:
            return {"status": "no_assistant"}
        
        # Extract boxed content
        boxed_content = extract_boxed_content(assistant_response)
        
        # Compute stats based on verification method
        has_boxed = boxed_content is not None
        has_ground_truth = ground_truth is not None
        
        if _use_sympy:
            # Sympy-based verification
            is_verifiable_result = is_verifiable_sympy(boxed_content) if has_boxed else False
            matches_gt = matches_ground_truth_sympy(boxed_content, ground_truth) if has_boxed and has_ground_truth else False
        else:
            # math_verify-based verification
            is_verifiable_result = is_verifiable(boxed_content) if has_boxed else False
            matches_gt = matches_ground_truth(boxed_content, ground_truth) if has_boxed and has_ground_truth else False
        
        return {
            "status": "ok",
            "has_boxed": has_boxed,
            "boxed_content": boxed_content,
            "ground_truth": ground_truth,
            "is_verifiable": is_verifiable_result,
            "has_ground_truth": has_ground_truth,
            "matches_gt": matches_gt,
            "text": text if matches_gt else None,  # Only keep text for samples matching GT (for tokenization)
        }
    except Exception:
        return None


def init_tokenizer():
    """Initialize tokenizer in worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def tokenize_text(text: str) -> int:
    """Tokenize a single text. Uses global tokenizer initialized in init_tokenizer."""
    global _tokenizer
    return len(_tokenizer.encode(text, add_special_tokens=False))


def main():
    global _use_sympy
    
    parser = argparse.ArgumentParser(description="Analyze verifiability of Nemotron-Math-v2 dataset")
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE, help="Path to jsonl file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to analyze")
    parser.add_argument("--use_sympy", action="store_true", help="Use sympy-based grading instead of math_verify")
    args = parser.parse_args()
    
    # Set global flag for workers
    _use_sympy = args.use_sympy
    
    data_file = args.file
    
    print("=" * 60)
    print("Nemotron-Math-v2 Verifiability Analysis")
    print(f"File: {data_file.split('/')[-1]}")
    print(f"Verifier: {'sympy' if args.use_sympy else 'math_verify'}")
    print(f"Workers: {NUM_WORKERS}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,}")
    print("=" * 60)
    
    # Load lines (stream if max_samples is set to avoid loading entire file)
    print(f"\nLoading file...")
    all_lines = []
    with open(data_file, 'r') as f:
        if args.max_samples:
            for i, line in enumerate(f):
                if i >= args.max_samples:
                    break
                all_lines.append(line)
            print(f"Loaded {len(all_lines):,} samples (limited by --max_samples)")
        else:
            all_lines = f.readlines()
            print(f"Total lines loaded: {len(all_lines):,}")
    
    
    # Process in parallel using pebble for timeout support
    print(f"\nProcessing samples in parallel (with timeout support)...")
    
    from pebble import ProcessPool
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    
    results = []
    timeout_count = 0
    TASK_TIMEOUT = 5  # seconds per task
    
    with ProcessPool(max_workers=NUM_WORKERS) as pool:
        future = pool.map(process_line, all_lines, timeout=TASK_TIMEOUT, chunksize=100)
        iterator = future.result()
        
        with tqdm(total=len(all_lines), desc="Analyzing") as pbar:
            while True:
                try:
                    result = next(iterator)
                    results.append(result)
                except StopIteration:
                    break
                except FuturesTimeoutError:
                    # Task timed out, append None and continue
                    results.append(None)
                    timeout_count += 1
                except Exception as e:
                    results.append(None)
                finally:
                    pbar.update(1)
    
    if timeout_count > 0:
        print(f"  (Timed out on {timeout_count:,} samples)")
    
    # Aggregate stats
    total_processed = 0
    filtered_count = 0
    no_assistant_count = 0
    parse_errors = 0
    
    ok_samples = 0
    has_boxed_count = 0
    is_verifiable_count = 0
    has_ground_truth_count = 0
    matches_gt_count = 0
    
    # For detailed breakdown
    verifiable_and_matches = 0
    verifiable_but_no_match = 0
    not_verifiable_count = 0
    no_gt_count = 0
    
    # Collect texts for samples matching GT (for tokenization)
    texts_matching_gt = []
    
    # Collect samples matching GT for preview
    samples_matching_gt = []

    
    # Collect some examples for display
    example_matches = []
    example_mismatches = []
    MAX_EXAMPLES = 10
    
    for result in results:
        total_processed += 1
        
        if result is None:
            parse_errors += 1
            continue
        
        status = result.get("status")
        
        if status == "filtered":
            filtered_count += 1
            continue
        
        if status == "no_assistant":
            no_assistant_count += 1
            continue
        
        if status == "ok":
            ok_samples += 1
            
            # Collect for preview (with boxed content)
            if result.get("has_boxed") and result.get("matches_gt") and len(samples_matching_gt) < 100:
                samples_matching_gt.append({
                    "boxed": result.get("boxed_content"),
                    "gt": result.get("ground_truth"),
                })
            
            if result.get("has_boxed"):
                has_boxed_count += 1
                
                if result.get("is_verifiable"):
                    is_verifiable_count += 1
                    
                    if result.get("has_ground_truth"):
                        has_ground_truth_count += 1
                        if result.get("matches_gt"):
                            matches_gt_count += 1
                            verifiable_and_matches += 1
                            # Collect text for tokenization
                            if result.get("text"):
                                texts_matching_gt.append(result["text"])
                            # Collect examples
                            if len(example_matches) < MAX_EXAMPLES:
                                example_matches.append({
                                    "boxed": result.get("boxed_content"),
                                    "gt": result.get("ground_truth"),
                                })
                        else:
                            verifiable_but_no_match += 1
                            # Collect mismatch examples
                            if len(example_mismatches) < MAX_EXAMPLES:
                                example_mismatches.append({
                                    "boxed": result.get("boxed_content"),
                                    "gt": result.get("ground_truth"),
                                })
                    else:
                        no_gt_count += 1
                else:
                    not_verifiable_count += 1
                    if result.get("has_ground_truth"):
                        has_ground_truth_count += 1
    
    # Print preview from samples matching GT
    if samples_matching_gt:
        print("\n" + "=" * 60)
        print(f"PREVIEW: ~20 Random examples (non-TIR, matches GT)")
        print("=" * 60)
        preview_samples = random.sample(samples_matching_gt, min(20, len(samples_matching_gt)))
        for i, ex in enumerate(preview_samples, 1):
            print(f"\n[{i}] GT: {ex['gt']}")
            print(f"    Boxed: {ex['boxed']}")
    
    # Print results
    print("\n" + "=" * 60)
    print("VERIFIABILITY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n--- Dataset Overview ---")
    print(f"Total lines processed:    {total_processed:,}")
    print(f"Parse errors:             {parse_errors:,}")
    print(f"Filtered (TIR/tools):     {filtered_count:,}")
    print(f"No assistant response:    {no_assistant_count:,}")
    print(f"Valid samples analyzed:   {ok_samples:,}")
    
    if ok_samples > 0:
        print(f"\n--- Boxed Answer Stats ---")
        print(f"Has \\boxed{{}}:            {has_boxed_count:,} ({100*has_boxed_count/ok_samples:.1f}%)")
        
        if has_boxed_count > 0:
            print(f"\n--- Verifiability (of samples with \\boxed{{}}) ---")
            print(f"Verifiable (parseable):   {is_verifiable_count:,} ({100*is_verifiable_count/has_boxed_count:.1f}%)")
            print(f"Not verifiable:           {not_verifiable_count:,} ({100*not_verifiable_count/has_boxed_count:.1f}%)")
            
            print(f"\n--- Ground Truth Matching ---")
            print(f"Has ground truth field:   {has_ground_truth_count:,}")
            
            if is_verifiable_count > 0:
                print(f"\nOf the {is_verifiable_count:,} verifiable samples:")
                print(f"  Matches ground truth:   {matches_gt_count:,} ({100*matches_gt_count/is_verifiable_count:.1f}%)")
                print(f"  Does NOT match GT:      {verifiable_but_no_match:,} ({100*verifiable_but_no_match/is_verifiable_count:.1f}%)")
                print(f"  No GT available:        {no_gt_count:,}")
            
            print(f"\n--- Summary ---")
            print(f"Verifiable & matches GT:  {verifiable_and_matches:,}")
            if has_boxed_count > 0:
                print(f"  (% of boxed samples):   {100*verifiable_and_matches/has_boxed_count:.1f}%")
            if ok_samples > 0:
                print(f"  (% of all valid):       {100*verifiable_and_matches/ok_samples:.1f}%")
    
    # Print example matches
    if example_matches:
        print("\n" + "=" * 60)
        print(f"EXAMPLE MATCHES (first {len(example_matches)} samples matching GT)")
        print("=" * 60)
        for i, ex in enumerate(example_matches, 1):
            print(f"\n[{i}] GT: {ex['gt']}")
            print(f"    Boxed: {ex['boxed']}")
    
    # Print example mismatches
    if example_mismatches:
        print("\n" + "=" * 60)
        print(f"EXAMPLE MISMATCHES (first {len(example_mismatches)} samples NOT matching GT)")
        print("=" * 60)
        for i, ex in enumerate(example_mismatches, 1):
            print(f"\n[{i}] GT: {ex['gt']}")
            print(f"    Boxed: {ex['boxed']}")
    
    # Tokenize and compute sequence length stats for samples matching GT
    if texts_matching_gt:
        print("\n" + "=" * 60)
        print("SEQUENCE LENGTH ANALYSIS (samples matching GT)")
        print("=" * 60)
        print(f"\nTokenizing {len(texts_matching_gt):,} samples...")
        
        with Pool(NUM_WORKERS, initializer=init_tokenizer) as pool:
            sequence_lengths = list(tqdm(
                pool.imap(tokenize_text, texts_matching_gt, chunksize=1000),
                total=len(texts_matching_gt),
                desc="Tokenizing"
            ))
        
        # Compute bucket distribution
        bucket_counts = Counter()
        for length in sequence_lengths:
            bucket_counts[get_bucket_label(length)] += 1
        
        min_len = min(sequence_lengths)
        max_len = max(sequence_lengths)
        avg_len = sum(sequence_lengths) / len(sequence_lengths)
        
        sorted_lengths = sorted(sequence_lengths)
        n = len(sorted_lengths)
        median_len = (sorted_lengths[n//2 - 1] + sorted_lengths[n//2]) / 2 if n % 2 == 0 else sorted_lengths[n//2]
        
        print(f"\n--- Sequence Length Statistics (in tokens) ---")
        print(f"Tokenizer: {TOKENIZER_NAME}")
        print(f"Samples:        {len(sequence_lengths):,}")
        print(f"Min length:     {min_len:,}")
        print(f"Max length:     {max_len:,}")
        print(f"Average length: {avg_len:,.1f}")
        print(f"Median length:  {median_len:,.1f}")
        
        print(f"\n--- Sequence Length Distribution ---")
        print(f"{'Bucket':<12} {'Count':>10} {'Percentage':>12}")
        print("-" * 36)
        
        for low, high, label in BUCKET_RANGES:
            count = bucket_counts.get(label, 0)
            pct = (count / len(sequence_lengths) * 100) if sequence_lengths else 0
            print(f"{label:<12} {count:>10,} {pct:>11.2f}%")
        
        print("-" * 36)
        print(f"{'TOTAL':<12} {len(sequence_lengths):>10,} {100.0:>11.2f}%")


if __name__ == "__main__":
    main()
