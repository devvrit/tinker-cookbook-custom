"""
Recipe to evaluate model with and without feedback on AIME24 and AIME25.

This script efficiently evaluates both baseline and feedback-conditioned performance
by reusing baseline samples as rollouts for feedback generation:
1. Generate baseline samples and grade them
2. Extract summaries from those same baseline samples
3. Generate feedback based on the summaries
4. Generate new samples conditioned on feedback
5. Grade the feedback-conditioned samples

This is more efficient than generating separate rollouts for feedback since
baseline samples are reused for both evaluation AND feedback generation.

Example usage:
    # Evaluate on full dataset
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        eval_aime25=True \\
        n_samples=4 \\
        temperature=0.6 \\
        max_tokens=16384
    
    # Evaluate on first 10 problems
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        max_problems=10
    
    # Evaluate on problems 5-15 (inclusive)
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback \\
        model_path=tinker://YOUR_CHECKPOINT_PATH \\
        eval_aime24=True \\
        start_idx=5 \\
        end_idx=16
    
    # Evaluate on random 20 problems
    python -m tinker_cookbook.recipes.distillation.eval_with_feedback \\
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
from tinker.types import ModelInput, SamplingParams
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import model_info, renderers
from tinker_cookbook.display import format_text
from tinker_cookbook.distillation.feedback_self_distillation_datasets import (
    DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    DEFAULT_PROXY_TEACHER_TEMPLATE,
    DEFAULT_STUDENT_SUFFIX,
    DEFAULT_THINK_CONTINUATION_TEXT,
    FeedbackSelfDistillationEnv,
    extract_summary_from_response,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.external_feedback import (
    extract_external_feedback,
    get_external_feedback,
)
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)

THINK_END_TOKEN = "</think>"


@chz.chz
class CLIConfig:
    """Command-line configuration for feedback evaluation."""

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
    max_tokens: int = 16384
    max_tokens_turn1: int | None = None  # Max tokens for thinking phase
    max_tokens_turn2: int | None = None  # Max tokens for answer phase

    # Feedback generation parameters
    filter_incomplete_traces: bool = True
    feedback_max_tokens: int = 2048
    feedback_temperature: float = 0.0  # Lower temperature for more deterministic, reliable feedback
    use_external_api: bool = True
    external_feedback_model: str = "gemini-2.0-flash-lite"  # Model to use for external feedback API
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE
    proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE
    student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX
    think_continuation_text: str = DEFAULT_THINK_CONTINUATION_TEXT

    # Evaluation parameters
    n_samples: int = 4  # Number of samples per problem for pass@k calculation (for feedback-conditioned generation)

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    preview_responses: bool = True  # If True, truncate long responses in logs
    preview_feedback: bool = True  # If True, truncate feedback in logs
    max_trajectories_to_log: int = 5  # Max number of detailed trajectories to log

    # Service configuration
    base_url: str | None = None


async def generate_feedback(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer,
    problem: str,
    answer: str,
    summaries: list[str],
    feedback_prompt_template: str,
    feedback_max_tokens: int,
    feedback_temperature: float,
    filter_incomplete_traces: bool,
    use_external_api: bool,
    external_feedback_model: str = "gemini-2.0-flash-lite",
) -> str | None:
    """Generate feedback based on problem, answer, and student summaries."""
    summaries_text = "\n".join(
        [f"Student solution {i+1}: {summary}" for i, summary in enumerate(summaries)]
    )
    
    if not summaries:
        summaries_text = "(No valid summaries available - students did not complete their responses)"
    
    # Create a temporary env to get the feedback prompt
    temp_env = FeedbackSelfDistillationEnv(
        problem=problem,
        answer=answer,
        renderer=renderer,
        tokenizer=tokenizer,
        feedback_prompt_template=feedback_prompt_template,
    )
    feedback_prompt = temp_env.get_feedback_prompt(summaries_text)
    
    if use_external_api:
        feedback_text = await get_external_feedback(
            feedback_prompt,
            feedback_temperature=feedback_temperature,
            feedback_max_tokens=feedback_max_tokens,
            model=external_feedback_model,
        )
        feedback_text = extract_external_feedback(
            feedback_text, start_tag="<feedback>", end_tag="</feedback>"
        )
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
        feedback_text = extract_summary_from_response(
            parsed_feedback_text, filter_incomplete_traces
        )
    
    return feedback_text


async def generate_with_feedback(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer,
    problem: str,
    feedback: str,
    proxy_teacher_template: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    max_tokens_turn1: int | None,
    max_tokens_turn2: int | None,
    think_continuation_text: str,
) -> list[list[int]]:
    """
    Generate responses conditioned on feedback using proxy teacher prompt.
    
    Returns:
        List of token sequences
    """
    # Create a temporary env to get the proxy teacher prompt
    temp_env = FeedbackSelfDistillationEnv(
        problem=problem,
        answer="",  # Not needed for proxy teacher prompt
        renderer=renderer,
        tokenizer=tokenizer,
        proxy_teacher_template=proxy_teacher_template,
    )
    temp_env.generated_feedback = feedback
    proxy_prompt = temp_env.get_proxy_teacher_prompt()
    
    prompt_input = renderer.build_generation_prompt(
        [renderers.Message(role="user", content=proxy_prompt)]
    )
    
    max_tokens_turn1 = max_tokens_turn1 or max_tokens
    max_tokens_turn2 = max_tokens_turn2 or max_tokens
    
    # Step 1: Generate thinking
    think_stop = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)
    step1_params = SamplingParams(
        max_tokens=max_tokens_turn1,
        temperature=temperature,
        stop=think_stop,
    )
    step1_resp = await sampling_client.sample_async(
        prompt=prompt_input,
        num_samples=n_samples,
        sampling_params=step1_params,
    )
    
    # Step 2: Continue each sample
    eos_stop = renderer.get_stop_sequences()
    step2_params = SamplingParams(
        max_tokens=max_tokens_turn2,
        temperature=temperature,
        stop=eos_stop,
    )
    
    all_tokens = []
    for seq in step1_resp.sequences:
        step1_tokens = seq.tokens
        step1_text = tokenizer.decode(step1_tokens)
        
        if THINK_END_TOKEN in step1_text:
            step2_prompt_tokens = prompt_input.to_ints() + step1_tokens
        else:
            continuation_tokens = tokenizer.encode(
                think_continuation_text, add_special_tokens=False
            )
            step2_prompt_tokens = prompt_input.to_ints() + step1_tokens + continuation_tokens
            step1_tokens = step1_tokens + continuation_tokens
        
        step2_prompt = ModelInput.from_ints(step2_prompt_tokens)
        step2_resp = await sampling_client.sample_async(
            prompt=step2_prompt,
            num_samples=1,
            sampling_params=step2_params,
        )
        step2_tokens = step2_resp.sequences[0].tokens
        
        full_tokens = list(step1_tokens) + list(step2_tokens)
        all_tokens.append(full_tokens)
    
    return all_tokens


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
    max_tokens_turn1: int | None,
    max_tokens_turn2: int | None,
    think_continuation_text: str,
    n_samples: int,
    feedback_prompt_template: str,
    proxy_teacher_template: str,
    feedback_max_tokens: int,
    feedback_temperature: float,
    filter_incomplete_traces: bool,
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
    
    This function reuses baseline samples as rollouts for feedback generation:
    1. Generate baseline samples (n_samples per problem)
    2. Grade baseline samples for baseline metrics
    3. Extract summaries from baseline samples
    4. Generate feedback from those summaries
    5. Generate feedback-conditioned samples (n_samples per problem)
    6. Grade feedback-conditioned samples
    
    This eliminates redundant generation by reusing baseline samples for feedback.
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
        
        prompt_text = problem + "\nWrite your answer in \\boxed{} format."
        prompts.append(prompt_text)
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
        Process a single problem efficiently:
        1. Generate baseline samples
        2. Extract summaries from baseline samples (reuse for feedback)
        3. Generate feedback
        4. Generate feedback-conditioned samples
        
        Returns both baseline and feedback results.
        """
        # Step 1: Generate baseline samples (these will also be used for feedback)
        prompt_input = renderer.build_generation_prompt(
            [renderers.Message(role="user", content=problem + DEFAULT_STUDENT_SUFFIX)]
        )
        
        local_max_tokens_turn1 = max_tokens_turn1 or max_tokens
        local_max_tokens_turn2 = max_tokens_turn2 or max_tokens
        
        think_stop = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)
        step1_params = SamplingParams(
            max_tokens=local_max_tokens_turn1,
            temperature=temperature,
            stop=think_stop,
        )
        step1_resp = await sampling_client.sample_async(
            prompt=prompt_input,
            num_samples=n_samples,
            sampling_params=step1_params,
        )
        
        eos_stop = renderer.get_stop_sequences()
        step2_params = SamplingParams(
            max_tokens=local_max_tokens_turn2,
            temperature=temperature,
            stop=eos_stop,
        )
        
        baseline_tokens_list = []
        baseline_summaries = []  # Extract summaries for feedback generation
        
        for seq in step1_resp.sequences:
            step1_tokens = seq.tokens
            step1_text = tokenizer.decode(step1_tokens)
            
            if THINK_END_TOKEN in step1_text:
                step2_prompt_tokens = prompt_input.to_ints() + step1_tokens
            else:
                continuation_tokens = tokenizer.encode(
                    think_continuation_text, add_special_tokens=False
                )
                step2_prompt_tokens = prompt_input.to_ints() + step1_tokens + continuation_tokens
                step1_tokens = step1_tokens + continuation_tokens
            
            step2_prompt = ModelInput.from_ints(step2_prompt_tokens)
            step2_resp = await sampling_client.sample_async(
                prompt=step2_prompt,
                num_samples=1,
                sampling_params=step2_params,
            )
            step2_tokens = step2_resp.sequences[0].tokens
            full_tokens = list(step1_tokens) + list(step2_tokens)
            baseline_tokens_list.append(full_tokens)
            
            # Extract summary from step2 (turn 2 is the summary/answer)
            summary_text = tokenizer.decode(step2_tokens, skip_special_tokens=True).strip()
            if summary_text:
                baseline_summaries.append(summary_text)
        
        # Step 2: Generate feedback from baseline summaries (reusing baseline samples!)
        feedback = await generate_feedback(
            sampling_client,
            renderer,
            tokenizer,
            problem,
            answer,
            baseline_summaries,
            feedback_prompt_template,
            feedback_max_tokens,
            feedback_temperature,
            filter_incomplete_traces,
            use_external_api,
            external_feedback_model,
        )
        
        # Step 3: Generate feedback-conditioned samples
        feedback_tokens_list = []
        if feedback is not None:
            feedback_tokens_list = await generate_with_feedback(
                sampling_client,
                renderer,
                tokenizer,
                problem,
                feedback,
                proxy_teacher_template,
                n_samples,
                temperature,
                max_tokens,
                max_tokens_turn1,
                max_tokens_turn2,
                think_continuation_text,
            )
        else:
            logger.warning(f"Problem {idx}: Failed to generate feedback, skipping feedback evaluation")
        
        return idx, baseline_tokens_list, feedback_tokens_list, answer, feedback
    
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
    
    pbar = tqdm(total=len(all_futures), desc=f"Evaluating {dataset_name} (combined)")
    for future in tqdm_asyncio.as_completed(all_futures):
        idx, baseline_tokens_list, feedback_tokens_list, ref, feedback_text = await future
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
    """Main evaluation function."""
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
    if config.model_path is not None and base_model is None:
        # Only fetch training run info if we need to auto-detect base_model
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(
            config.model_path
        )
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
        logger.info(f"Evaluating {dataset_name} - COMBINED (baseline + feedback)")
        logger.info(f"Baseline samples will be reused as rollouts for feedback generation")
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
                config.max_tokens_turn1,
                config.max_tokens_turn2,
                config.think_continuation_text,
                config.n_samples,
                feedback_prompt_template=config.feedback_prompt_template,
                proxy_teacher_template=config.proxy_teacher_template,
                feedback_max_tokens=config.feedback_max_tokens,
                feedback_temperature=config.feedback_temperature,
                filter_incomplete_traces=config.filter_incomplete_traces,
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
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    if config.model_path:
        logger.info(f"Checkpoint: {config.model_path}")
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"N Samples: {config.n_samples}")
    logger.info("(Baseline samples reused as rollouts for feedback generation)")
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

