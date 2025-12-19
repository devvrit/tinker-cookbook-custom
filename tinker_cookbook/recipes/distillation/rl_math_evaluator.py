import asyncio
import logging
import math
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import chz
import tinker
from datasets import load_dataset
from tinker.types import ModelInput, SamplingParams
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# Two-step generation constants
THINK_END_TOKEN = "</think>"
DEFAULT_THINK_CONTINUATION_TEXT = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"

@chz.chz
class RLMathEvaluatorBuilder:
    dataset_name: str
    split: str = "train"
    max_samples: int | None = None
    n_samples: int = 1
    temperature: float = 0.6
    max_tokens: int = 16384  # Default/fallback max_tokens
    model_name: str = "Qwen/Qwen3-8B-Base"  # Needed for tokenizer
    renderer_name: str | None = None
    
    # Two-step generation with interruption
    use_two_step_generation: bool = False  # If True, use two-step generation with </think> interruption
    max_tokens_turn1: int | None = None  # Max tokens for thinking phase; None uses max_tokens
    max_tokens_turn2: int | None = None  # Max tokens for answer phase; None uses max_tokens
    think_continuation_text: str = DEFAULT_THINK_CONTINUATION_TEXT
    
    def __call__(self) -> SamplingClientEvaluator:
        return RLMathEvaluator(self)

class RLMathEvaluator(SamplingClientEvaluator):
    def __init__(self, config: RLMathEvaluatorBuilder):
        self.config = config
        # Load dataset
        self.dataset = load_dataset(self.config.dataset_name, split=self.config.split)
        if self.config.max_samples:
            self.dataset = self.dataset.select(range(min(len(self.dataset), self.config.max_samples)))
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(self.config.model_name)
        
        # Initialize renderer if provided, or try to infer/default
        if self.config.renderer_name:
            self.renderer = renderers.get_renderer(self.config.renderer_name, self.tokenizer)
        else:
            # Fallback or default behavior if needed, though explicit is better
            # For now let's assume if not provided we don't use stop tokens (backward compatibility)
            raise ValueError("Renderer name must be provided")
            
    async def _sample_single_step(
        self,
        sampling_client: tinker.SamplingClient,
        prompt_inp: ModelInput,
    ) -> list[list[int]]:
        """Single-step generation (original behavior)."""
        stop_sequences = self.renderer.get_stop_sequences() if self.renderer else None
        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=stop_sequences,
        )
        resp = await sampling_client.sample_async(
            prompt=prompt_inp,
            num_samples=self.config.n_samples,
            sampling_params=sampling_params,
        )
        return [seq.tokens for seq in resp.sequences]

    async def _sample_two_step(
        self,
        sampling_client: tinker.SamplingClient,
        prompt_inp: ModelInput,
    ) -> list[list[int]]:
        """
        Two-step generation with </think> interruption.
        
        Step 1: Generate thinking until </think> or max_tokens_turn1
        Step 2: Generate answer until EOS or max_tokens_turn2
                If </think> was NOT found in step 1, prepend continuation text
        """
        max_tokens_turn1 = self.config.max_tokens_turn1 or self.config.max_tokens
        max_tokens_turn2 = self.config.max_tokens_turn2 or self.config.max_tokens
        
        # Step 1: Generate thinking with </think> as stop sequence
        think_stop = self.tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)
        step1_params = SamplingParams(
            max_tokens=max_tokens_turn1,
            temperature=self.config.temperature,
            stop=think_stop,
        )
        step1_resp = await sampling_client.sample_async(
            prompt=prompt_inp,
            num_samples=self.config.n_samples,
            sampling_params=step1_params,
        )
        
        # Step 2: Continue each sample
        eos_stop = self.renderer.get_stop_sequences() if self.renderer else None
        step2_params = SamplingParams(
            max_tokens=max_tokens_turn2,
            temperature=self.config.temperature,
            stop=eos_stop,
        )
        
        all_full_tokens = []
        for seq in step1_resp.sequences:
            step1_tokens = seq.tokens
            step1_text = self.tokenizer.decode(step1_tokens)
            
            # Check if </think> was generated
            if THINK_END_TOKEN in step1_text:
                # </think> found - continue from current tokens
                step2_prompt_tokens = prompt_inp.to_ints() + step1_tokens
            else:
                # </think> NOT found - append continuation text
                continuation_tokens = self.tokenizer.encode(
                    self.config.think_continuation_text, add_special_tokens=False
                )
                step2_prompt_tokens = prompt_inp.to_ints() + step1_tokens + continuation_tokens
                # Update step1_tokens to include continuation for final output
                step1_tokens = step1_tokens + continuation_tokens
            
            step2_prompt = ModelInput.from_ints(step2_prompt_tokens)
            step2_resp = await sampling_client.sample_async(
                prompt=step2_prompt,
                num_samples=1,  # One continuation per step1 sample
                sampling_params=step2_params,
            )
            step2_tokens = step2_resp.sequences[0].tokens
            
            # Combine step1 + step2 tokens
            full_tokens = list(step1_tokens) + list(step2_tokens)
            all_full_tokens.append(full_tokens)
        
        return all_full_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        logger.info(f"Starting evaluation on {self.config.dataset_name} with {len(self.dataset)} samples")
        if self.config.use_two_step_generation:
            logger.info("Using two-step generation with </think> interruption")
        
        prompts = []
        references = []
        
        for sample in self.dataset:
            # Handle different dataset formats
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
        
        async def sample_with_index(idx: int, prompt_inp: ModelInput):
            if self.config.use_two_step_generation:
                tokens_list = await self._sample_two_step(sampling_client, prompt_inp)
            else:
                tokens_list = await self._sample_single_step(sampling_client, prompt_inp)
            return idx, tokens_list, prompt_inp

        all_futures = []
        for i, prompt_text in enumerate(prompts):
            inp = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=prompt_text)]
            )
            all_futures.append(sample_with_index(i, inp))
        
        # Grade answers and calculate pass@k
        total_correct = 0
        total_samples = 0
        pass_at_k_sums = {k: 0.0 for k in [1, 2, 4, 8, 16, 32, 64] if k <= self.config.n_samples}
        
        pbar = tqdm(total=len(all_futures), desc=f"Evaluating {self.config.dataset_name}")
        iterations = 0
        for future in tqdm_asyncio.as_completed(all_futures):
            i, tokens_list, prompt_inp = await future
            pbar.update(1)
            iterations += 1
            
            ref = references[i]
            
            # Grade each sample for this prompt
            correct_count = 0
            for seq_idx, tokens in enumerate(tokens_list):
                response = self.renderer.parse_response(tokens)[0]
                decoded_text = response["content"]
                
                # Log first few samples for debugging
                if iterations <= 3 and seq_idx == 0:
                    prompt_with_special = self.tokenizer.decode(prompt_inp.to_ints(), skip_special_tokens=False)
                    output_with_special = self.tokenizer.decode(tokens, skip_special_tokens=False)
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Sample {i} (Input with special tokens):\n{prompt_with_special}")
                    logger.info(f"Sample {i} (Output with special tokens):\n{output_with_special}")
                
                try:
                    extracted = extract_boxed(decoded_text)
                    is_correct = grade_answer(extracted, ref)
                    if is_correct:
                        correct_count += 1
                    
                    if iterations <= 3 and seq_idx == 0:
                        logger.info(f"Sample {i} (Extracted): {extracted}, Ref: {ref}, Correct: {is_correct}")
                        logger.info(f"{'='*80}")
                        
                except Exception:
                    if iterations <= 3 and seq_idx == 0:
                        logger.info(f"Sample {i}: Failed to extract/grade answer")
                        logger.info(f"{'='*80}")
            
            n = len(tokens_list)
            c = correct_count
            
            total_correct += c
            total_samples += n
            
            for k in pass_at_k_sums:
                if n - c < k:
                    pass_at_k_sums[k] += 1.0
                else:
                    prob_fail = math.comb(n - c, k) / math.comb(n, k)
                    pass_at_k_sums[k] += (1.0 - prob_fail)
            
            current_processed = total_samples / self.config.n_samples
            if current_processed > 0:
                running_pass1 = pass_at_k_sums[1] / current_processed
                running_acc = total_correct / total_samples
                pbar.set_postfix({"pass@1": f"{running_pass1:.2%}", "acc": f"{running_acc:.2%}"})
        
        pbar.close()
        
        metrics = {}
        num_problems = len(references)
        
        if num_problems > 0:
            for k, total_pass in pass_at_k_sums.items():
                metrics[f"{self.config.dataset_name}/pass@{k}"] = total_pass / num_problems
            
            metrics[f"{self.config.dataset_name}/accuracy_per_sample"] = total_correct / total_samples if total_samples > 0 else 0.0
            metrics[f"{self.config.dataset_name}/num_problems"] = num_problems
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics
