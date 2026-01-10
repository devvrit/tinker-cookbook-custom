"""
Reprocess Nemotron-Math-v2 dataset to normalize \boxed{} answers using Gemini.

This script:
1. Loads and filters dataset (removes TIR/tool samples)
2. Sends filtered samples to Gemini for boxed answer normalization
3. Saves results with resume capability

Usage:
    python reprocess_boxed_answers.py \
        --input_file /path/to/medium.jsonl \
        --output_file /path/to/output.jsonl \
        --batch_size 100 \
        --seed 42
"""

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Optional

from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DEFAULT_INPUT_FILE = "/home/devvrit03/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/f837702092f10dc3039d33e08cd1839f2a2f986a/data/high.part_01.jsonl"

# Pattern to detect Python code blocks (indicates TIR usage)
PYTHON_CODE_PATTERN = re.compile(r'```python|```py|<\|python_start\|>|য়', re.IGNORECASE)

# Gemini prompt template
NORMALIZATION_PROMPT = """You are given an LLM assistant response to a math problem. But the final answer within \\boxed{{}} might not be verifiable using tools like math verify or sympy. Your work is to re-write just the \\boxed{{}} expression such that it becomes verifiable.

User Problem:
{user_prompt}

Ground Truth Answer: {ground_truth}

Assistant's Response (contains the derivation and a \\boxed{{}} answer):
{assistant_response}

Your task:
1. Extract the answer the assistant arrived at from their \\boxed{{}} expression
2. If it matches the ground truth, output a clean \\boxed{{}} format that can be verified (e.g., simple numeric, fraction, or expression) using math verify/sympy.
3. If the assistant's answer within \\boxed{{}} doesn't match ground truth, still output what the assistant computed in a clean verifiable \\boxed{{}} format.

Output ONLY the \\boxed{{}} expression at the end in a single new line, nothing else. For example: \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."""



def has_python_tir(text: str) -> bool:
    """Check if text contains Python TIR patterns."""
    return bool(PYTHON_CODE_PATTERN.search(text))


def extract_text(example: dict) -> str:
    """Extract all text from messages field."""
    messages = example.get("messages", [])
    text_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
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


def extract_user_prompt(example: dict) -> Optional[str]:
    """Extract user prompt from messages."""
    messages = example.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return None


def extract_assistant_response(example: dict) -> Optional[str]:
    """Extract assistant response from messages."""
    messages = example.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def extract_ground_truth(example: dict) -> Optional[str]:
    """Extract ground truth answer from example.
    
    The ground truth is typically in the dataset metadata or can be
    extracted from the user message.
    """
    # Try common field names
    for field in ["answer", "Answer", "ground_truth", "solution", "expected", "expected_answer"]:
        if field in example:
            return str(example[field])
    return None


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \boxed{...} expression."""
    # Find the last \boxed{ occurrence
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def is_verifiable(boxed_content: str) -> bool:
    """Check if a boxed expression is verifiable using math_verify.
    
    Returns True if math_verify can parse the expression, False otherwise.
    """
    try:
        from math_verify import parse
        # Wrap in $ for proper parsing
        if not boxed_content.startswith("$"):
            boxed_content = f"${boxed_content}$"
        parsed = parse(boxed_content)
        # If parse returns something non-empty, it's verifiable
        return parsed is not None and len(parsed) > 0
    except Exception:
        return False


def matches_ground_truth(boxed_content: str, ground_truth: str) -> bool:
    """Check if a boxed expression matches the ground truth using math_verify.
    
    Returns True if math_verify verifies them as equal, False otherwise.
    """
    if not boxed_content or not ground_truth or ground_truth == "(not available)":
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


def replace_boxed_answer(original_response: str, new_boxed: str) -> str:
    """Replace the last \boxed{} in the response with the new one."""
    # Find the last \boxed{...} pattern
    pattern = r'(\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    matches = list(re.finditer(pattern, original_response))
    
    if not matches:
        # No boxed found, append the new one
        return original_response + "\n" + new_boxed
    
    # Replace the last occurrence
    last_match = matches[-1]
    return original_response[:last_match.start()] + new_boxed + original_response[last_match.end():]


async def call_gemini_with_retry(
    client: genai.Client,
    prompt: str,
    model: str,
    max_retries: int = 3,
    retry_delay: float = 30.0,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Optional[str]:
    """Call Gemini API with retry logic."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
            )
            return response.text
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Gemini call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Gemini call failed after {max_retries} attempts: {e}")
    
    return None


async def process_single_sample(
    client: genai.Client,
    example: dict,
    index: int,
    model: str,
    max_retries: int,
    retry_delay: float,
    semaphore: asyncio.Semaphore,
) -> tuple[int, Optional[dict], Optional[str], Optional[dict]]:
    """Process a single sample through Gemini.
    
    Returns: (index, processed_example, error_message, stats_info)
    stats_info contains: original_boxed, new_boxed, original_verifiable, new_verifiable
    """
    async with semaphore:
        user_prompt = extract_user_prompt(example)
        assistant_response = extract_assistant_response(example)
        ground_truth = extract_ground_truth(example)
        
        if not user_prompt or not assistant_response:
            return index, None, "Missing user prompt or assistant response", None
        
        # Extract original boxed content for stats
        original_boxed = extract_boxed_content(assistant_response)
        original_verifiable = is_verifiable(original_boxed) if original_boxed else False
        
        if not ground_truth:
            # If no ground truth, we still process but note it
            ground_truth = "(not available)"
        
        # Build the Gemini prompt
        prompt = NORMALIZATION_PROMPT.format(
            user_prompt=user_prompt,
            ground_truth=ground_truth,
            assistant_response=assistant_response,
        )
        
        # Call Gemini
        gemini_response = await call_gemini_with_retry(
            client=client,
            prompt=prompt,
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        if gemini_response is None:
            return index, None, "Gemini call failed after retries", None
        
        # Extract the normalized boxed answer from Gemini response
        # Gemini may output multiple lines, with the answer on the LAST line
        response_lines = gemini_response.strip().split('\n')
        last_line = response_lines[-1].strip()
        
        # Try to extract boxed from the last line first
        if last_line.startswith("\\boxed{"):
            normalized_boxed = last_line
        else:
            # Try to extract boxed from the last line
            extracted = extract_boxed_content(last_line)
            if extracted:
                normalized_boxed = f"\\boxed{{{extracted}}}"
            else:
                # Fall back to extracting from the full response
                extracted = extract_boxed_content(gemini_response)
                if extracted:
                    normalized_boxed = f"\\boxed{{{extracted}}}"
                else:
                    # Last resort: wrap the last line content
                    normalized_boxed = f"\\boxed{{{last_line}}}"
        
        # Extract new boxed content for stats
        new_boxed_content = extract_boxed_content(normalized_boxed)
        new_verifiable = is_verifiable(new_boxed_content) if new_boxed_content else False
        
        # Check ground truth matching
        original_matches_gt = matches_ground_truth(original_boxed, ground_truth) if original_boxed else False
        new_matches_gt = matches_ground_truth(new_boxed_content, ground_truth) if new_boxed_content else False
        
        # Create stats info
        stats_info = {
            "original_boxed": original_boxed,
            "new_boxed": new_boxed_content,
            "ground_truth": ground_truth,
            "original_verifiable": original_verifiable,
            "new_verifiable": new_verifiable,
            "original_matches_gt": original_matches_gt,
            "new_matches_gt": new_matches_gt,
        }
        
        # Create a modified copy of the example
        modified_example = json.loads(json.dumps(example))  # Deep copy
        
        # Replace the boxed answer in the assistant response
        messages = modified_example.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                original_content = msg.get("content", "")
                msg["content"] = replace_boxed_answer(original_content, normalized_boxed)
                break
        
        return index, modified_example, None, stats_info


async def process_batch(
    client: genai.Client,
    samples: list[tuple[int, dict]],  # List of (index, example) tuples
    model: str,
    max_retries: int,
    retry_delay: float,
    concurrent_limit: int,
) -> tuple[list[tuple[int, dict, dict]], list[tuple[int, str]]]:
    """Process a batch of samples concurrently.
    
    Returns: (successful_results, failed_results)
    successful_results contains (index, processed_example, stats_info)
    """
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        process_single_sample(
            client=client,
            example=example,
            index=idx,
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
            semaphore=semaphore,
        )
        for idx, example in samples
    ]
    
    successful = []
    failed = []
    
    for coro in tqdm_asyncio.as_completed(tasks, desc="Processing samples"):
        idx, result, error, stats_info = await coro
        if result is not None:
            successful.append((idx, result, stats_info))
        else:
            failed.append((idx, error))
    
    return successful, failed


def stream_and_filter_chunks(
    input_file: str, 
    chunk_size: int, 
    max_samples: Optional[int] = None
) -> tuple[list[tuple[int, dict]], dict]:
    """Stream dataset and yield filtered chunks.
    
    Yields: List of (original_line_index, example) tuples for each chunk
    Returns stats dict at the end via the final yield
    """
    logger.info(f"Streaming dataset from {input_file} (chunk_size={chunk_size})")
    
    current_chunk = []
    total_lines = 0
    filtered_out = 0
    total_kept = 0
    
    with open(input_file, 'r') as f:
        for line_idx, line in enumerate(f):
            total_lines += 1
            
            # Check if we've reached max_samples
            if max_samples is not None and total_kept >= max_samples:
                break
            
            try:
                example = json.loads(line)
                if should_filter_sample(example):
                    filtered_out += 1
                else:
                    current_chunk.append((line_idx, example))
                    total_kept += 1
                    
                    # Yield chunk when full
                    if len(current_chunk) >= chunk_size:
                        logger.info(f"Read {total_lines:,} lines, kept {total_kept:,}, filtered {filtered_out:,}")
                        yield current_chunk, None
                        current_chunk = []
                        
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line {line_idx}")
                continue
    
    # Yield final partial chunk if any
    if current_chunk:
        logger.info(f"Read {total_lines:,} lines, kept {total_kept:,}, filtered {filtered_out:,}")
        yield current_chunk, None
    
    # Final stats
    stats = {
        "total_lines": total_lines,
        "filtered_out": filtered_out,
        "total_kept": total_kept,
    }
    yield [], stats


async def main():
    parser = argparse.ArgumentParser(description="Reprocess boxed answers using Gemini")
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of samples per batch for progress saving (default: 10000)"
    )
    parser.add_argument(
        "--concurrent_limit",
        type=int,
        default=100,
        help="Maximum concurrent Gemini requests (default: 100)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per request (default: 3)"
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=30.0,
        help="Delay in seconds between retries (default: 30)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model to use (default: gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skips already-processed samples)"
    )
    parser.add_argument(
        "--preview_only",
        action="store_true",
        help="Only load and filter, don't process (for testing)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Load existing results if resuming
    existing_results = {}
    processed_indices = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if "_original_index" in result:
                        existing_results[result["_original_index"]] = result
                        processed_indices.add(result["_original_index"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resuming: loaded {len(existing_results)} existing results")
    
    # Stats tracking
    all_results = dict(existing_results)
    all_stats = []
    example_conversions = []
    MAX_EXAMPLES = 5
    total_processed = len(existing_results)
    total_failed = 0
    chunk_idx = 0
    file_stats = None
    
    # Stream and process chunks
    for chunk, stats in stream_and_filter_chunks(
        args.input_file, 
        args.batch_size, 
        args.max_samples
    ):
        # Final stats yield
        if stats is not None:
            file_stats = stats
            continue
            
        if not chunk:
            continue
        
        # Filter out already processed samples (for resume)
        if processed_indices:
            chunk = [(idx, ex) for idx, ex in chunk if idx not in processed_indices]
            if not chunk:
                continue
        
        chunk_idx += 1
        logger.info(f"\nProcessing chunk {chunk_idx} ({len(chunk)} samples)")
        
        # Preview mode - just show first chunk samples
        if args.preview_only:
            logger.info("Preview mode - showing first 5 samples from first chunk:")
            for i, (line_idx, example) in enumerate(chunk[:5]):
                user_prompt = extract_user_prompt(example)
                assistant_resp = extract_assistant_response(example)
                ground_truth = extract_ground_truth(example)
                boxed = extract_boxed_content(assistant_resp) if assistant_resp else None
                
                logger.info(f"\n--- Sample {i} (line {line_idx}) ---")
                logger.info(f"User prompt: {user_prompt[:200] if user_prompt else 'N/A'}...")
                logger.info(f"Ground truth: {ground_truth}")
                logger.info(f"Current boxed: {boxed}")
            return
        
        # Process this chunk
        successful, failed = await process_batch(
            client=client,
            samples=chunk,
            model=args.gemini_model,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            concurrent_limit=args.concurrent_limit,
        )
        
        # Update results and collect stats
        for idx, result, stats_info in successful:
            result["_original_index"] = idx
            all_results[idx] = result
            total_processed += 1
            
            if stats_info:
                all_stats.append(stats_info)
                
                # Collect example conversions (first few with actual changes)
                if len(example_conversions) < MAX_EXAMPLES:
                    if stats_info.get("original_boxed") != stats_info.get("new_boxed"):
                        example_conversions.append(stats_info)
        
        for idx, error in failed:
            total_failed += 1
            logger.warning(f"Sample {idx} failed: {error}")
        
        # Write results after each chunk (append mode for efficiency)
        with open(args.output_file, 'w') as f:
            for idx in sorted(all_results.keys()):
                result = all_results[idx]
                result_clean = {k: v for k, v in result.items() if k != "_original_index"}
                f.write(json.dumps(result_clean) + "\n")
        
        logger.info(f"Chunk {chunk_idx} complete. "
                   f"Total processed: {total_processed}, "
                   f"Total failed: {total_failed}")
        
        # Print examples after first chunk for quick feedback
        if chunk_idx == 1 and example_conversions:
            logger.info("\n" + "=" * 60)
            logger.info("EARLY EXAMPLES (from first chunk)")
            logger.info("=" * 60)
            for i, stats in enumerate(example_conversions):
                logger.info(f"\n--- Example {i + 1} ---")
                logger.info(f"Ground Truth: {stats.get('ground_truth', 'N/A')}")
                logger.info(f"Original boxed: {stats.get('original_boxed', 'N/A')}")
                logger.info(f"New boxed:      {stats.get('new_boxed', 'N/A')}")
                logger.info(f"Original verifiable: {stats.get('original_verifiable', False)} | matches GT: {stats.get('original_matches_gt', False)}")
                logger.info(f"New verifiable:      {stats.get('new_verifiable', False)} | matches GT: {stats.get('new_matches_gt', False)}")
            logger.info("=" * 60 + "\n")
    
    # Print example conversions
    if example_conversions:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE CONVERSIONS (first few samples with changes)")
        logger.info("=" * 60)
        for i, stats in enumerate(example_conversions[:MAX_EXAMPLES]):
            logger.info(f"\n--- Example {i + 1} ---")
            logger.info(f"Ground Truth: {stats.get('ground_truth', 'N/A')}")
            logger.info(f"Original boxed: {stats.get('original_boxed', 'N/A')}")
            logger.info(f"New boxed:      {stats.get('new_boxed', 'N/A')}")
            logger.info(f"Original verifiable: {stats.get('original_verifiable', False)} | matches GT: {stats.get('original_matches_gt', False)}")
            logger.info(f"New verifiable:      {stats.get('new_verifiable', False)} | matches GT: {stats.get('new_matches_gt', False)}")
    
    # Calculate final stats
    if all_stats:
        total = len(all_stats)
        orig_verifiable = sum(1 for s in all_stats if s.get("original_verifiable", False))
        new_verifiable = sum(1 for s in all_stats if s.get("new_verifiable", False))
        orig_matches_gt = sum(1 for s in all_stats if s.get("original_matches_gt", False))
        new_matches_gt = sum(1 for s in all_stats if s.get("new_matches_gt", False))
        
        # Conversion tracking
        became_verifiable = sum(1 for s in all_stats 
                                if not s.get("original_verifiable", False) and s.get("new_verifiable", False))
        lost_verifiable = sum(1 for s in all_stats 
                              if s.get("original_verifiable", False) and not s.get("new_verifiable", False))
        became_matching_gt = sum(1 for s in all_stats 
                                  if not s.get("original_matches_gt", False) and s.get("new_matches_gt", False))
        lost_matching_gt = sum(1 for s in all_stats 
                                if s.get("original_matches_gt", False) and not s.get("new_matches_gt", False))
        
        logger.info("\n" + "=" * 60)
        logger.info("VERIFIABILITY STATS")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total}")
        logger.info(f"")
        logger.info(f"--- Verifiable (parseable by math_verify) ---")
        logger.info(f"Originally verifiable:   {orig_verifiable:,} ({100*orig_verifiable/total:.1f}%)")
        logger.info(f"Now verifiable:          {new_verifiable:,} ({100*new_verifiable/total:.1f}%)")
        logger.info(f"Became verifiable:       {became_verifiable:,}")
        logger.info(f"Lost verifiability:      {lost_verifiable:,}")
        logger.info(f"Net change:              {became_verifiable - lost_verifiable:+,}")
        logger.info(f"")
        logger.info(f"--- Matches Ground Truth (verified by math_verify) ---")
        logger.info(f"Originally matching GT:  {orig_matches_gt:,} ({100*orig_matches_gt/total:.1f}%)")
        logger.info(f"Now matching GT:         {new_matches_gt:,} ({100*new_matches_gt/total:.1f}%)")
        logger.info(f"Became matching GT:      {became_matching_gt:,}")
        logger.info(f"Lost GT matching:        {lost_matching_gt:,}")
        logger.info(f"Net change:              {became_matching_gt - lost_matching_gt:+,}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    if file_stats:
        logger.info(f"File stats: {file_stats['total_lines']:,} lines read, "
                   f"{file_stats['filtered_out']:,} filtered (TIR/tools), "
                   f"{file_stats['total_kept']:,} kept")
    logger.info(f"Total samples processed: {total_processed}")
    logger.info(f"Total failed: {total_failed}")
    logger.info(f"Output file: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
