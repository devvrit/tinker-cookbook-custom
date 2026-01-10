"""
Analyze dataset files for sequence length statistics.
Supports both JSONL (Nemotron) and Parquet (OpenThoughts) formats.
Filters for samples without Python TIR (optional).

Usage:
    python analyze_nemotron_math.py [path_to_file] [--skip-tir-filter]
    
Examples:
    python analyze_nemotron_math.py medium.jsonl
    python analyze_nemotron_math.py train-00000-of-00120.parquet --skip-tir-filter
"""

import argparse
import re
import json
import random
from collections import Counter
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from transformers import AutoTokenizer

# --- CONFIGURATION ---
DEFAULT_FILE = "/home/devvrit03/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/f837702092f10dc3039d33e08cd1839f2a2f986a/data/medium.jsonl"
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
_skip_tir_filter = False


def get_bucket_label(length: int) -> str:
    for low, high, label in BUCKET_RANGES:
        if low <= length < high:
            return label
    return ">18k"


def has_python_tir(text: str) -> bool:
    return bool(PYTHON_CODE_PATTERN.search(text))


def extract_boxed_ending(content: str) -> str | None:
    """Extract the ending around \\boxed{} - from 20 chars before it to the end."""
    # Find the last occurrence of \boxed{
    boxed_idx = content.rfind(r'\boxed{')
    if boxed_idx == -1:
        # Try alternative format: \\boxed{
        boxed_idx = content.rfind('\\boxed{')
    
    if boxed_idx != -1:
        # Start 20 chars before \boxed{
        start_idx = max(0, boxed_idx - 20)
        return content[start_idx:]
    else:
        # Fallback: return last 150 chars
        return content[-150:] if len(content) > 150 else content


def extract_text(example: dict) -> str:
    """Extract text from messages field (Nemotron) or conversations field (OpenThoughts).
    
    For Nemotron format, includes both 'reasoning_content' and 'content' fields
    since the full rollout is reasoning followed by the summarized answer.
    """
    # Try Nemotron format first (messages with role/content/reasoning_content)
    messages = example.get("messages", [])
    if messages:
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
    
    # Try OpenThoughts format (conversations with from/value)
    conversations = example.get("conversations", [])
    if conversations:
        text_parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                value = turn.get("value", "")
                if value:
                    text_parts.append(value)
        return "\n".join(text_parts)
    
    return ""


def init_tokenizer():
    """Initialize tokenizer in worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def get_assistant_content(example: dict) -> str | None:
    """Get assistant/gpt response content from either schema."""
    # Try Nemotron format (messages with role: assistant)
    messages = example.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    
    # Try OpenThoughts format (conversations with from: gpt)
    conversations = example.get("conversations", [])
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            return turn.get("value", "")
    
    return None


def process_line(line: str) -> tuple:
    """Process a single JSON line. Returns (status, text, has_tools_field, has_code, sample_preview, assistant_ending) or None."""
    global _skip_tir_filter
    try:
        example = json.loads(line)
        text = extract_text(example)
        if not text:
            return None
        
        # Check if 'tools' field exists and is non-empty
        tools_val = example.get("tools") or example.get("tool")
        has_tools_field = bool(tools_val)
        
        # Check for Python code blocks in the text
        has_code_blocks = has_python_tir(text)
        
        # For debugging: save a preview of samples with tools but no code blocks
        sample_preview = None
        if has_tools_field and not has_code_blocks:
            sample_preview = text[:500] if text else None
        
        # Get the \boxed{} ending from assistant/gpt response
        assistant_ending = None
        content = get_assistant_content(example)
        if content:
            assistant_ending = extract_boxed_ending(content)
        
        # Skip TIR filtering if flag is set (for datasets without tool use)
        if not _skip_tir_filter and (has_code_blocks or has_tools_field):
            return ("tir", None, has_tools_field, has_code_blocks, sample_preview, assistant_ending)
        
        return ("text", text, has_tools_field, has_code_blocks, None, assistant_ending)
    except:
        return None


def tokenize_text(text: str) -> int:
    """Tokenize a single text. Uses global tokenizer initialized in init_tokenizer."""
    global _tokenizer
    return len(_tokenizer.encode(text, add_special_tokens=False))


def load_data_file(data_file: str) -> list[str]:
    """Load data from JSONL or Parquet file, return list of JSON strings."""
    if data_file.endswith('.parquet'):
        import pandas as pd
        import numpy as np
        print(f"Loading parquet file...")
        df = pd.read_parquet(data_file)
        
        # Convert numpy arrays to Python lists for JSON serialization
        def convert_row(row):
            result = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                elif isinstance(v, list):
                    # Handle nested numpy arrays in lists
                    result[k] = [
                        {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv 
                         for kk, vv in item.items()} if isinstance(item, dict) else item
                        for item in v
                    ]
                else:
                    result[k] = v
            return result
        
        return [json.dumps(convert_row(row)) for row in df.to_dict(orient='records')]
    else:
        # Assume JSONL
        with open(data_file, 'r') as f:
            return f.readlines()


def main():
    global _skip_tir_filter
    parser = argparse.ArgumentParser(description="Analyze dataset for sequence length statistics")
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE, help="Path to jsonl or parquet file")
    parser.add_argument("--skip-tir-filter", action="store_true",
                        help="Skip TIR filtering (for datasets without tool use like OpenThoughts)")
    args = parser.parse_args()
    
    _skip_tir_filter = args.skip_tir_filter
    
    data_file = args.file
    
    print("=" * 60)
    print("Dataset Sequence Length Analysis")
    print(f"File: {data_file.split('/')[-1]}")
    if _skip_tir_filter:
        print("Filter: None (TIR filtering disabled)")
    else:
        print("Filter: No Python TIR (no code blocks)")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 60)
    
    # Load data (supports both JSONL and Parquet)
    print(f"\nLoading data...")
    all_lines = load_data_file(data_file)
    total_lines = len(all_lines)
    print(f"Total samples: {total_lines:,}")
    
    # Early display: Show 20 random boxed endings BEFORE any preprocessing
    print("\n" + "=" * 60)
    print("EARLY PREVIEW: 20 Random \\boxed{} endings (before preprocessing)")
    print("=" * 60)
    sample_indices = random.sample(range(total_lines), min(20, total_lines))
    for idx in sample_indices:
        try:
            example = json.loads(all_lines[idx])
            content = get_assistant_content(example)
            if content:
                ending = extract_boxed_ending(content)
                if ending:
                    print(f"\n[Line {idx}]:")
                    print(ending)
        except:
            pass
    print("\n" + "=" * 60)
    
    # Step 1: Read and filter in parallel
    print("\n" + "=" * 60)
    print("Step 1: Reading & filtering (parallel)")
    print("=" * 60 + "\n")
    
    texts_to_tokenize = []
    skipped_total = 0
    has_tools_field_count = 0
    has_code_blocks_count = 0
    has_both = 0
    has_tools_only = 0
    has_code_only = 0
    example_previews = []  # For debugging samples with tools but no code
    assistant_endings = []  # For showing \boxed{} formats
    
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_line, all_lines, chunksize=1000),
            total=total_lines,
            desc="Reading & filtering"
        ))
    
    for result in results:
        if result is None:
            continue
        status, text, has_tools, has_code, preview, ending = result
        
        if has_tools:
            has_tools_field_count += 1
        if has_code:
            has_code_blocks_count += 1
        
        if status == "tir":
            skipped_total += 1
            if has_tools and has_code:
                has_both += 1
            elif has_tools and not has_code:
                has_tools_only += 1
                if preview and len(example_previews) < 3:
                    example_previews.append(preview)
            elif has_code and not has_tools:
                has_code_only += 1
        else:
            texts_to_tokenize.append(text)
            # Collect some assistant endings for display
            if ending and len(assistant_endings) < 20:
                assistant_endings.append(ending)
    
    filtered_samples = len(texts_to_tokenize)
    print(f"\nFiltered: {filtered_samples:,} samples")
    print(f"Skipped (TIR or tools field): {skipped_total:,}")
    
    print(f"\n--- TIR Detection Comparison ---")
    print(f"Samples with 'tools'/'tool' field:     {has_tools_field_count:,}")
    print(f"Samples with code blocks detected:     {has_code_blocks_count:,}")
    print(f"  - Has BOTH tools field AND code:     {has_both:,}")
    print(f"  - Has tools field ONLY (no code):    {has_tools_only:,}")
    print(f"  - Has code blocks ONLY (no tools):   {has_code_only:,}")
    
    if example_previews:
        print(f"\n--- Examples with tools field but NO code blocks ---")
        for i, preview in enumerate(example_previews, 1):
            print(f"\nExample {i}:")
            print("-" * 40)
            print(preview[:400] + "..." if len(preview) > 400 else preview)
            print("-" * 40)
    
    if assistant_endings:
        print(f"\n--- Sample \\boxed{{}} endings (from non-TIR samples) ---")
        for i, ending in enumerate(assistant_endings, 1):
            print(f"\nSample {i}:")
            print(ending)
    
    # Step 2: Tokenize in parallel
    print("\n" + "=" * 60)
    print("Step 2: Tokenizing sequences (parallel)")
    print("=" * 60 + "\n")
    
    with Pool(NUM_WORKERS, initializer=init_tokenizer) as pool:
        sequence_lengths = list(tqdm(
            pool.imap(tokenize_text, texts_to_tokenize, chunksize=1000),
            total=len(texts_to_tokenize),
            desc="Tokenizing"
        ))
    
    # Compute bucket distribution
    bucket_counts = Counter()
    for length in sequence_lengths:
        bucket_counts[get_bucket_label(length)] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n--- Dataset Overview ---")
    print(f"Total samples: {total_lines:,}")
    print(f"Samples without Python TIR: {filtered_samples:,}")
    print(f"Skipped (TIR/tools field): {skipped_total:,}")
    
    if sequence_lengths:
        min_len = min(sequence_lengths)
        max_len = max(sequence_lengths)
        avg_len = sum(sequence_lengths) / len(sequence_lengths)
        
        sorted_lengths = sorted(sequence_lengths)
        n = len(sorted_lengths)
        median_len = (sorted_lengths[n//2 - 1] + sorted_lengths[n//2]) / 2 if n % 2 == 0 else sorted_lengths[n//2]
        
        print(f"\n--- Sequence Length Statistics (in tokens) ---")
        print(f"Tokenizer: {TOKENIZER_NAME}")
        print(f"Min length:     {min_len:,}")
        print(f"Max length:     {max_len:,}")
        print(f"Average length: {avg_len:,.1f}")
        print(f"Median length:  {median_len:,.1f}")
        
        print(f"\n--- Sequence Length Distribution ---")
        print(f"{'Bucket':<12} {'Count':>10} {'Percentage':>12}")
        print("-" * 36)
        
        for low, high, label in BUCKET_RANGES:
            count = bucket_counts.get(label, 0)
            pct = (count / filtered_samples * 100) if filtered_samples > 0 else 0
            print(f"{label:<12} {count:>10,} {pct:>11.2f}%")
        
        print("-" * 36)
        print(f"{'TOTAL':<12} {filtered_samples:>10,} {100.0:>11.2f}%")


if __name__ == "__main__":
    main()
