"""
Recipe to evaluate instruct models with and without feedback on AIME24 and AIME25.

This is the v2 version designed for instruct models (like Qwen3 instruct) that don't
use a separate thinking phase. Instead:
1. Generate baseline samples with a prompt asking for a summary inside <summary> tags
2. Grade baseline samples and extract summaries from the <summary> tags
3. Generate feedback based on the extracted summaries
4. Generate new samples conditioned on feedback
5. Grade the feedback-conditioned samples

Example usage:
    # Evaluate on full dataset
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback_instruct \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        eval_aime25=True \\
        n_samples=4 \\
        temperature=0.6 \\
        max_tokens=4096
    
    # Evaluate on first 10 problems
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback_instruct \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        max_problems=10
    
    # Evaluate on problems 5-15 (inclusive)
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback_instruct \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        start_idx=5 \\
        end_idx=16
    
    # Evaluate on random 20 problems
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback_instruct \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        max_problems=20 \\
        random_seed=42
"""

import asyncio
import json
import logging
import math
import os
import random
import sys
import time

import chz
import tinker
from datasets import load_dataset
from tinker.types import SamplingParams
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import model_info, renderers
from tinker_cookbook.display import format_text
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.external_feedback import (
    extract_external_feedback,
    get_external_feedback,
)
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)

# Tags for summary extraction
SUMMARY_START_TAG = "<summary>"
SUMMARY_END_TAG = "</summary>"

# Default prompt templates for instruct models
DEFAULT_INSTRUCT_STUDENT_SUFFIX = """

Solve this problem step by step. Write your final answer in \\boxed{} format.

After your solution, provide a very thorough and detailed summary of your approach and answer inside <summary> and </summary> tags. The summary should describe in detail each step, calculation, and reasoning you used to solve the problem. Make sure to also include the final answer in \\boxed{} format in this summary."""

DEFAULT_INSTRUCT_FEEDBACK_PROMPT_TEMPLATE = """You are analyzing student attempts at solving a math problem to create helpful feedback for a NEW student who will attempt this problem for the first time.

Problem: {problem}
Ground Truth Answer: {answer}

Student Solution Summaries:
{summaries}

First, reason through each student summary carefully. Analyze what each student did correctly and incorrectly. Consider whether different students may have taken valid alternative approaches.

Then, based on your analysis and the ground truth, create feedback specifically designed to help a NEW student who has never seen this problem before. The feedback should:
1. Warn about common mistakes, misconceptions, and pitfalls to avoid (learned from the attempts above)
2. Suggest effective problem-solving strategies and key concepts to consider (note: there may be multiple valid solution paths - do not assume only one correct method exists)
3. Provide hints about important reasoning steps without giving away the solution

Important guidelines:
- Do not leak the final answer in your feedback
- Be aware that multiple correct approaches may exist - avoid insisting on a single "correct" method if alternatives are valid
- Write the feedback as actionable guidance that will help a first-time solver improve their problem-solving process
- Frame the feedback as forward-looking advice (e.g., "Consider...", "Watch out for...", "A useful approach is...") rather than commentary on past attempts
- Warn about common mistakes, misconceptions, and pitfalls to avoid.

After your reasoning, provide your final summarized feedback inside <feedback> and </feedback> tags. This feedback will be given directly to a new student, so write it in second person (e.g., "You should consider...") and make it immediately useful for someone approaching this problem fresh.
"""

DEFAULT_INSTRUCT_PROXY_TEACHER_TEMPLATE = """You are solving a math problem.
Problem: {problem}

You have received the following feedback from reviewing multiple solution attempts:
Feedback: {feedback}

Now solve the problem step by step. Write your final answer in \\boxed{{}} format.

After your solution, provide a very thorough and detailed summary of your approach and answer inside <summary> and </summary> tags. The summary should describe in detail each step, calculation, and reasoning you used to solve the problem. Make sure to also include the final answer in \\boxed{{}} format in this summary."""


def extract_summary_from_response_instruct(
    response_text: str,
    start_tag: str = SUMMARY_START_TAG,
    end_tag: str = SUMMARY_END_TAG,
) -> str | None:
    """
    Extract the summary from inside <summary>...</summary> tags.
    
    Args:
        response_text: Full response text from the model
        start_tag: Opening tag for summary (default: <summary>)
        end_tag: Closing tag for summary (default: </summary>)
        
    Returns:
        The summary text inside the tags, or None if start tag not found
    """
    start_idx = response_text.find(start_tag)
    end_idx = response_text.find(end_tag)
    
    if start_idx == -1:
        # Start tag not found, no summary available
        return None
    elif end_idx == -1:
        # Start tag found but end tag missing, return everything after start tag
        summary = response_text[start_idx + len(start_tag):].strip()
        return summary if summary else None
    else:
        # Both tags found, return content between them
        summary = response_text[start_idx + len(start_tag):end_idx].strip()
        return summary if summary else None


@chz.chz
class CLIConfig:
    """Command-line configuration for feedback evaluation (instruct model version)."""

    # Model/checkpoint configuration
    model_path: str | None = None  # tinker:// path to checkpoint
    model_name: str | None = None  # Base model name (auto-detected if not provided)
    renderer_name: str | None = "qwen3"  # Renderer name

    # Dataset configuration
    eval_aime24: bool = False
    eval_aime25: bool = False
    max_problems: int | None = None  # Limit number of problems to evaluate (takes first N problems)
    start_idx: int | None = None  # Start index for subset evaluation (0-based)
    end_idx: int | None = None  # End index for subset evaluation (exclusive, if None uses all after start_idx)
    random_seed: int | None = None  # Random seed for random subset selection (if set, randomly samples max_problems)

    # Generation hyperparameters
    temperature: float = 0.6
    max_tokens: int = 16000  # Lower default since no thinking phase

    # Feedback generation parameters
    feedback_max_tokens: int = 2048
    feedback_temperature: float = 0.0  # Lower temperature for more deterministic, reliable feedback
    use_external_api: bool = True
    external_feedback_model: str = "gemini-2.5-flash-lite"  # Model to use for external feedback API
    
    # Prompt templates
    student_prompt_suffix: str = DEFAULT_INSTRUCT_STUDENT_SUFFIX
    feedback_prompt_template: str = DEFAULT_INSTRUCT_FEEDBACK_PROMPT_TEMPLATE
    proxy_teacher_template: str = DEFAULT_INSTRUCT_PROXY_TEACHER_TEMPLATE

    # Evaluation parameters
    n_samples: int = 4  # Number of samples per problem for pass@k calculation

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    preview_responses: bool = True  # If True, truncate long responses in logs
    preview_feedback: bool = True  # If True, truncate feedback in logs
    max_trajectories_to_log: int = 5  # Max number of detailed trajectories to log

    # Service configuration
    base_url: str | None = None


async def generate_feedback_instruct(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    problem: str,
    answer: str,
    summaries: list[str],
    feedback_prompt_template: str,
    feedback_max_tokens: int,
    feedback_temperature: float,
    use_external_api: bool,
    external_feedback_model: str = "gemini-2.0-flash-lite",
    max_retries: int = 3,
    retry_delay_seconds: int = 120,
) -> str | None:
    """Generate feedback based on problem, answer, and student summaries.
    
    Args:
        max_retries: Maximum number of retry attempts on failure (default: 3)
        retry_delay_seconds: Delay in seconds between retry attempts (default: 60)
    """
    if summaries:
        summaries_text = "\n".join(
            [f"Student solution {i+1}: {summary}" for i, summary in enumerate(summaries)]
        )
    else:
        summaries_text = "(No valid summaries available - students did not include summary tags)"
    
    feedback_prompt = feedback_prompt_template.format(
        problem=problem,
        answer=answer,
        summaries=summaries_text,
    )
    
    if use_external_api:
        last_error = None
        for attempt in range(max_retries):
            try:
                feedback_text = await get_external_feedback(
                    feedback_prompt,
                    feedback_temperature=feedback_temperature,
                    feedback_max_tokens=feedback_max_tokens,
                    model=external_feedback_model,
                )
                feedback_text = extract_external_feedback(
                    feedback_text, start_tag="<feedback>", end_tag="</feedback>"
                )
                return feedback_text
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Feedback generation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay_seconds} seconds..."
                    )
                    await asyncio.sleep(retry_delay_seconds)
                else:
                    logger.error(
                        f"Feedback generation failed after {max_retries} attempts: {e}"
                    )
        return None
    else:
        feedback_convo = [
            {"role": "user", "content": feedback_prompt},
        ]
        feedback_input = renderer.build_generation_prompt(feedback_convo)
        
        feedback_response = await sampling_client.sample_async(
            prompt=feedback_input,
            num_samples=1,
            sampling_params=SamplingParams(
                max_tokens=feedback_max_tokens,
                temperature=feedback_temperature,
                stop=renderer.get_stop_sequences(),
            ),
        )
        
        feedback_tokens = feedback_response.sequences[0].tokens
        feedback_message, _ = renderer.parse_response(feedback_tokens)
        parsed_feedback_text = feedback_message["content"]
        feedback_text = extract_external_feedback(
            parsed_feedback_text, start_tag="<feedback>", end_tag="</feedback>"
        )
    
    return feedback_text


async def generate_with_feedback_instruct(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    problem: str,
    feedback: str,
    proxy_teacher_template: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
) -> list[list[int]]:
    """
    Generate responses conditioned on feedback using proxy teacher prompt.
    
    For instruct models, this is a single-phase generation.
    
    Returns:
        List of token sequences
    """
    proxy_prompt = proxy_teacher_template.format(
        problem=problem,
        feedback=feedback,
    )
    
    prompt_input = renderer.build_generation_prompt(
        [renderers.Message(role="user", content=proxy_prompt)]
    )
    
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    
    response = await sampling_client.sample_async(
        prompt=prompt_input,
        num_samples=n_samples,
        sampling_params=params,
    )
    
    return [list(seq.tokens) for seq in response.sequences]


async def evaluate_dataset_combined(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer,
    dataset_name: str,
    split: str,
    max_problems: int | None,
    start_idx: int | None,
    end_idx: int | None,
    random_seed: int | None,
    temperature: float,
    max_tokens: int,
    n_samples: int,
    student_prompt_suffix: str,
    feedback_prompt_template: str,
    proxy_teacher_template: str,
    feedback_max_tokens: int,
    feedback_temperature: float,
    use_external_api: bool,
    external_feedback_model: str = "gemini-2.0-flash-lite",
    log_path: str | None = None,
    ml_logger: ml_log.Logger | None = None,
    preview_responses: bool = True,
    preview_feedback: bool = True,
    max_trajectories_to_log: int = 5,
) -> dict[str, float]:
    """
    Efficiently evaluate a dataset with both baseline and feedback-conditioned generation.
    
    For instruct models (single-phase generation):
    1. Generate baseline samples with prompt asking for summary in <summary> tags
    2. Grade baseline samples and extract summaries
    3. Generate feedback from those summaries
    4. Generate feedback-conditioned samples (single phase)
    5. Grade feedback-conditioned samples
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    total_size = len(dataset)
    logger.info(f"Dataset has {total_size} total problems")
    
    # Select subset based on provided options
    if random_seed is not None and max_problems is not None:
        # Random sampling
        random.seed(random_seed)
        indices = random.sample(range(total_size), min(max_problems, total_size))
        indices.sort()  # Keep sorted for reproducibility
        dataset = dataset.select(indices)
        logger.info(f"Randomly selected {len(dataset)} problems (seed={random_seed})")
    elif start_idx is not None:
        # Range selection
        start = start_idx
        end = end_idx if end_idx is not None else total_size
        end = min(end, total_size)
        if start < 0 or start >= total_size:
            raise ValueError(f"start_idx {start} is out of range [0, {total_size})")
        if end <= start:
            raise ValueError(f"end_idx {end} must be greater than start_idx {start}")
        dataset = dataset.select(range(start, end))
        logger.info(f"Selected problems {start} to {end-1} (inclusive): {len(dataset)} problems")
    elif max_problems is not None:
        # Take first N problems
        dataset = dataset.select(range(min(max_problems, total_size)))
        logger.info(f"Selected first {len(dataset)} problems")
    else:
        logger.info(f"Evaluating all {total_size} problems")
    
    prompts = []
    references = []
    
    for sample in dataset:
        if "Problem" in sample:  # AIME 2024
            problem = sample["Problem"]
            answer = str(sample["Answer"])
        elif "problem" in sample:  # AIME 2025
            problem = sample["problem"]
            answer = sample["answer"]
        else:
            logger.warning(f"Unknown sample format: {sample.keys()}")
            continue
        
        prompts.append(problem)
        references.append(answer)
    
    logger.info(f"Evaluating {len(prompts)} problems (combined baseline + feedback)")
    
    # Track metrics for both baseline and feedback
    baseline_total_correct = 0
    baseline_total_samples = 0
    baseline_pass_at_k_sums = {k: 0.0 for k in [1, 2, 4, 8, 16, 32, 64] if k <= n_samples}
    
    feedback_total_correct = 0
    feedback_total_samples = 0
    feedback_pass_at_k_sums = {k: 0.0 for k in [1, 2, 4, 8, 16, 32, 64] if k <= n_samples}
    
    # Track trajectories for logging
    trajectories_logged = 0
    logged_trajectories_baseline = []
    logged_trajectories_feedback = []
    
    # Track timing
    eval_start_time = time.time()
    
    async def process_problem_combined(idx: int, problem: str, answer: str):
        """
        Process a single problem for instruct models:
        1. Generate baseline samples with summary prompt
        2. Extract summaries from <summary> tags
        3. Generate feedback
        4. Generate feedback-conditioned samples
        
        Returns both baseline and feedback results.
        """
        # Build prompt with suffix asking for summary
        full_prompt = problem + student_prompt_suffix
        prompt_input = renderer.build_generation_prompt(
            [renderers.Message(role="user", content=full_prompt)]
        )
        
        # Single-phase generation for baseline
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        
        response = await sampling_client.sample_async(
            prompt=prompt_input,
            num_samples=n_samples,
            sampling_params=params,
        )
        
        baseline_tokens_list = []
        baseline_summaries = []
        
        for seq in response.sequences:
            tokens = list(seq.tokens)
            baseline_tokens_list.append(tokens)
            
            # Decode and extract summary from <summary> tags
            response_text = tokenizer.decode(tokens, skip_special_tokens=True)
            summary = extract_summary_from_response_instruct(response_text)
            if summary:
                baseline_summaries.append(summary)
        
        # Generate feedback from baseline summaries
        feedback = await generate_feedback_instruct(
            sampling_client,
            renderer,
            problem,
            answer,
            baseline_summaries,
            feedback_prompt_template,
            feedback_max_tokens,
            feedback_temperature,
            use_external_api,
            external_feedback_model,
        )
        
        # Generate feedback-conditioned samples
        feedback_tokens_list = []
        if feedback is not None:
            feedback_tokens_list = await generate_with_feedback_instruct(
                sampling_client,
                renderer,
                problem,
                feedback,
                proxy_teacher_template,
                n_samples,
                temperature,
                max_tokens,
            )
        else:
            logger.warning(f"Problem {idx}: Failed to generate feedback, skipping feedback evaluation")
        
        return idx, baseline_tokens_list, feedback_tokens_list, answer, feedback, baseline_summaries
    
    def grade_samples(tokens_list: list[list[int]], ref: str) -> tuple[int, list[dict]]:
        """Grade a list of token sequences and return (correct_count, sample_results)."""
        correct_count = 0
        sample_results = []
        for sample_idx, tokens in enumerate(tokens_list):
            response = renderer.parse_response(tokens)[0]
            decoded_text = response["content"]
            
            extracted = None
            is_correct = False
            try:
                extracted = extract_boxed(decoded_text)
                is_correct = grade_answer(extracted, ref)
                if is_correct:
                    correct_count += 1
            except Exception:
                pass
            
            sample_results.append({
                "sample_idx": sample_idx,
                "response": decoded_text,
                "extracted_answer": extracted,
                "is_correct": is_correct,
            })
        return correct_count, sample_results
    
    def update_pass_at_k(pass_at_k_sums: dict, n: int, c: int):
        """Update pass@k sums given n samples and c correct."""
        for k in pass_at_k_sums:
            if n - c < k:
                pass_at_k_sums[k] += 1.0
            else:
                prob_fail = math.comb(n - c, k) / math.comb(n, k)
                pass_at_k_sums[k] += (1.0 - prob_fail)
    
    # Process all problems
    all_futures = [
        process_problem_combined(i, problem, ref)
        for i, (problem, ref) in enumerate(zip(prompts, references))
    ]
    
    pbar = tqdm(total=len(all_futures), desc=f"Evaluating {dataset_name} (instruct, combined)")
    for future in tqdm_asyncio.as_completed(all_futures):
        idx, baseline_tokens_list, feedback_tokens_list, ref, feedback_text, summaries = await future
        pbar.update(1)
        
        problem_text = prompts[idx]
        
        # Grade baseline samples
        baseline_correct, baseline_results = grade_samples(baseline_tokens_list, ref)
        baseline_n = len(baseline_tokens_list)
        baseline_total_correct += baseline_correct
        baseline_total_samples += baseline_n
        update_pass_at_k(baseline_pass_at_k_sums, baseline_n, baseline_correct)
        
        # Grade feedback samples
        feedback_correct, feedback_results = grade_samples(feedback_tokens_list, ref)
        feedback_n = len(feedback_tokens_list)
        feedback_total_correct += feedback_correct
        feedback_total_samples += feedback_n
        if feedback_n > 0:
            update_pass_at_k(feedback_pass_at_k_sums, feedback_n, feedback_correct)
        
        # Log trajectories
        if trajectories_logged < max_trajectories_to_log:
            # Baseline trajectory
            baseline_traj = {
                "problem_idx": idx,
                "problem": problem_text,
                "reference_answer": ref,
                "use_feedback": False,
                "feedback": None,
                "extracted_summaries": summaries,
                "samples": baseline_results,
                "correct_count": baseline_correct,
                "total_samples": baseline_n,
            }
            logged_trajectories_baseline.append(baseline_traj)
            
            # Feedback trajectory
            feedback_traj = {
                "problem_idx": idx,
                "problem": problem_text,
                "reference_answer": ref,
                "use_feedback": True,
                "feedback": feedback_text,
                "samples": feedback_results,
                "correct_count": feedback_correct,
                "total_samples": feedback_n,
            }
            logged_trajectories_feedback.append(feedback_traj)
            trajectories_logged += 1
            
            # Log to console with preview formatting
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAJECTORY {trajectories_logged}/{max_trajectories_to_log}")
            logger.info(f"{'='*80}")
            logger.info(f"Problem {idx}:")
            logger.info(f"  Problem: {format_text(problem_text, preview_responses)}")
            logger.info(f"  Reference Answer: {ref}")
            
            logger.info(f"\n  --- BASELINE (no feedback) ---")
            logger.info(f"  Extracted Summaries ({len(summaries)}/{baseline_n}):")
            for i, summary in enumerate(summaries[:2]):  # Show first 2 summaries
                logger.info(f"    Summary {i+1}: {format_text(summary, preview_responses)}")
            logger.info(f"  Baseline Correct: {baseline_correct}/{baseline_n}")
            for sample in baseline_results[:2]:  # Log first 2 for brevity
                response_preview = format_text(sample['response'], preview_responses)
                logger.info(f"    Sample {sample['sample_idx']}: {sample['extracted_answer']} ({'✓' if sample['is_correct'] else '✗'})")
                logger.info(f"      Response: {response_preview}")
            
            logger.info(f"\n  --- WITH FEEDBACK ---")
            if feedback_text:
                logger.info(f"  Feedback: {format_text(feedback_text, preview_feedback)}")
            logger.info(f"  Feedback Correct: {feedback_correct}/{feedback_n}")
            for sample in feedback_results[:2]:  # Log first 2 for brevity
                response_preview = format_text(sample['response'], preview_responses)
                logger.info(f"    Sample {sample['sample_idx']}: {sample['extracted_answer']} ({'✓' if sample['is_correct'] else '✗'})")
                logger.info(f"      Response: {response_preview}")
            logger.info(f"{'='*80}\n")
        
        # Update progress bar
        current_processed = baseline_total_samples / n_samples if n_samples > 0 else 0
        if current_processed > 0:
            running_baseline_pass1 = baseline_pass_at_k_sums[1] / current_processed
            running_feedback_pass1 = feedback_pass_at_k_sums[1] / current_processed if feedback_total_samples > 0 else 0
            pbar.set_postfix({
                "base@1": f"{running_baseline_pass1:.2%}",
                "fb@1": f"{running_feedback_pass1:.2%}",
            })
    
    pbar.close()
    
    # Write trajectories to file if log_path is provided
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        
        if logged_trajectories_baseline:
            baseline_file = os.path.join(log_path, f"trajectories_{dataset_name.replace('/', '_')}_baseline.jsonl")
            with open(baseline_file, "w") as f:
                for traj in logged_trajectories_baseline:
                    f.write(json.dumps(traj) + "\n")
            logger.info(f"Logged {len(logged_trajectories_baseline)} baseline trajectories to {baseline_file}")
        
        if logged_trajectories_feedback:
            feedback_file = os.path.join(log_path, f"trajectories_{dataset_name.replace('/', '_')}_with_feedback.jsonl")
            with open(feedback_file, "w") as f:
                for traj in logged_trajectories_feedback:
                    f.write(json.dumps(traj) + "\n")
            logger.info(f"Logged {len(logged_trajectories_feedback)} feedback trajectories to {feedback_file}")
    
    # Compute final metrics
    eval_duration = time.time() - eval_start_time
    metrics = {}
    num_problems = len(references)
    
    # Use cleaner dataset key
    dataset_key = dataset_name.replace("/", "_").replace("-", "_")
    
    if num_problems > 0:
        # Baseline metrics
        for k, total_pass in baseline_pass_at_k_sums.items():
            metrics[f"eval/{dataset_key}/pass@{k}_baseline"] = total_pass / num_problems
        metrics[f"eval/{dataset_key}/accuracy_baseline"] = (
            baseline_total_correct / baseline_total_samples if baseline_total_samples > 0 else 0.0
        )
        
        # Feedback metrics
        for k, total_pass in feedback_pass_at_k_sums.items():
            metrics[f"eval/{dataset_key}/pass@{k}_with_feedback"] = total_pass / num_problems
        metrics[f"eval/{dataset_key}/accuracy_with_feedback"] = (
            feedback_total_correct / feedback_total_samples if feedback_total_samples > 0 else 0.0
        )
        
        # Improvement metrics
        baseline_pass1 = baseline_pass_at_k_sums[1] / num_problems if num_problems > 0 else 0
        feedback_pass1 = feedback_pass_at_k_sums[1] / num_problems if num_problems > 0 else 0
        metrics[f"eval/{dataset_key}/pass@1_improvement"] = feedback_pass1 - baseline_pass1
        
        # Meta metrics
        metrics[f"eval/{dataset_key}/num_problems"] = num_problems
        metrics[f"eval/{dataset_key}/n_samples"] = n_samples
        metrics[f"time/{dataset_key}_eval_seconds"] = eval_duration
    
    # Log metrics via ml_logger if available
    if ml_logger is not None:
        ml_logger.log_metrics(metrics)
    
    return metrics


async def main(config: CLIConfig):
    """Main evaluation function for instruct models."""
    # Set up logging infrastructure
    if config.log_path:
        log_dir = os.path.expanduser(config.log_path)
        os.makedirs(log_dir, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=log_dir,
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name,
            config=config,
        )
        
        # Save the command used to launch this evaluation
        command_str = " ".join(sys.argv)
        logger.info(f"Launch command: {command_str}")
        command_file = os.path.join(log_dir, "command.txt")
        with open(command_file, "w") as f:
            f.write(command_str + "\n")
        logger.info(f"Command saved to {command_file}")
    else:
        # Fallback to basic logging if no log_path
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        ml_logger = None
    
    eval_start_time = time.time()
    
    if config.model_path is None and config.model_name is None:
        raise ValueError("Either model_path or model_name must be provided")
    
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    # Auto-detect base model from checkpoint if not provided
    base_model = config.model_name
    if config.model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(
            config.model_path
        )
        if base_model:
            if base_model != training_run.base_model:
                raise ValueError(
                    f"Provided model_name {base_model} does not match "
                    f"checkpoint's base model {training_run.base_model}"
                )
        else:
            base_model = training_run.base_model
            logger.info(f"Auto-detected base model from checkpoint: {base_model}")
    
    if base_model is None:
        raise ValueError("Could not determine base model. Please provide model_name.")
    
    # Get renderer name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        base_model
    )
    logger.info(f"Using renderer: {renderer_name}")
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    # Create sampling client
    if config.model_path:
        # Convert training checkpoint path (weights/) to sampler path (sampler_weights/) if needed
        sampler_path = config.model_path
        if "/weights/" in sampler_path:
            sampler_path = sampler_path.replace("/weights/", "/sampler_weights/")
            logger.info(
                f"Converted training checkpoint path to sampler path: {config.model_path} -> {sampler_path}"
            )
        
        logger.info(f"Creating sampling client with checkpoint: {sampler_path}")
        sampling_client = service_client.create_sampling_client(
            model_path=sampler_path, base_model=base_model
        )
    else:
        logger.info(f"Creating sampling client for base model: {base_model}")
        sampling_client = service_client.create_sampling_client(base_model=base_model)
    
    all_metrics = {}
    
    # Evaluate each dataset with combined baseline + feedback evaluation
    datasets_to_eval = []
    if config.eval_aime24:
        datasets_to_eval.append(("Maxwell-Jia/AIME_2024", "train"))
    if config.eval_aime25:
        datasets_to_eval.append(("math-ai/aime25", "test"))
    
    if not datasets_to_eval:
        raise ValueError(
            "No evaluations specified. Use eval_aime24=True or eval_aime25=True"
        )
    
    for dataset_name, split in datasets_to_eval:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {dataset_name} - INSTRUCT MODEL (single-phase generation)")
        logger.info(f"Baseline samples will be reused for summary extraction and feedback")
        logger.info(f"{'='*80}")
        
        # Combined evaluation: baseline samples are reused for feedback
        dataset_metrics = {}
        with timed(f"eval_{dataset_name.replace('/', '_')}", dataset_metrics):
            combined_metrics = await evaluate_dataset_combined(
                sampling_client,
                renderer,
                tokenizer,
                dataset_name,
                split,
                config.max_problems,
                config.start_idx,
                config.end_idx,
                config.random_seed,
                config.temperature,
                config.max_tokens,
                config.n_samples,
                student_prompt_suffix=config.student_prompt_suffix,
                feedback_prompt_template=config.feedback_prompt_template,
                proxy_teacher_template=config.proxy_teacher_template,
                feedback_max_tokens=config.feedback_max_tokens,
                feedback_temperature=config.feedback_temperature,
                use_external_api=config.use_external_api,
                external_feedback_model=config.external_feedback_model,
                log_path=config.log_path,
                ml_logger=ml_logger,
                preview_responses=config.preview_responses,
                preview_feedback=config.preview_feedback,
                max_trajectories_to_log=config.max_trajectories_to_log,
            )
        all_metrics.update(combined_metrics)
        all_metrics.update(dataset_metrics)
    
    # Add total timing
    total_eval_time = time.time() - eval_start_time
    all_metrics["time/total_eval_seconds"] = total_eval_time
    all_metrics["progress/complete"] = 1.0
    
    # Add model info to metrics
    all_metrics["config/model_path"] = config.model_path or ""
    all_metrics["config/base_model"] = base_model
    all_metrics["config/temperature"] = config.temperature
    all_metrics["config/n_samples"] = config.n_samples
    all_metrics["config/max_tokens"] = config.max_tokens
    
    # Log final metrics via ml_logger
    if ml_logger is not None:
        ml_logger.log_metrics(all_metrics)
    
    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS (INSTRUCT MODEL)")
    logger.info("=" * 80)
    if config.model_path:
        logger.info(f"Checkpoint: {config.model_path}")
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"N Samples: {config.n_samples}")
    logger.info("(Single-phase generation with <summary> tag extraction)")
    logger.info("-" * 80)
    
    # Log metrics grouped by category
    for metric_name, metric_value in sorted(all_metrics.items()):
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")
    
    logger.info("=" * 80)
    logger.info(f"Total evaluation time: {total_eval_time:.1f}s")
    
    # Cleanup logging
    if ml_logger is not None:
        if hasattr(ml_logger, 'get_logger_url'):
            url = ml_logger.get_logger_url()
            if url:
                logger.info(f"Results logged to: {url}")
        ml_logger.close()
        logger.info("Evaluation completed successfully")
    
    return all_metrics


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(main(config))

