import logging
import chz
import tinker
from datasets import load_dataset
from tinker.types import ModelInput, SamplingParams
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

@chz.chz
class RLMathEvaluatorBuilder:
    dataset_name: str
    split: str = "train"
    max_samples: int | None = None
    n_samples: int = 1
    temperature: float = 0.6
    max_tokens: int = 16384
    model_name: str = "Qwen/Qwen3-8B-Base" # Needed for tokenizer
    
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
            
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        logger.info(f"Starting evaluation on {self.config.dataset_name} with {len(self.dataset)} samples")
        
        prompts = []
        references = []
        
        for sample in self.dataset:
            # Handle different dataset formats
            if "Problem" in sample: # AIME 2024
                problem = sample["Problem"]
                answer = str(sample["Answer"])
            elif "problem" in sample: # AIME 2025
                problem = sample["problem"]
                answer = sample["answer"]
            else:
                logger.warning(f"Unknown sample format: {sample.keys()}")
                continue
                
            prompt_text = problem + "\nWrite your answer in \\boxed{} format."
            prompts.append(prompt_text)
            references.append(answer)
            
        # Generate samples
        import math
        
        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        # Let's rewrite the loop to use asyncio.gather
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        
        async def sample_with_index(idx, prompt_inp):
            resp = await sampling_client.sample_async(
                prompt=prompt_inp,
                num_samples=self.config.n_samples,
                sampling_params=sampling_params
            )
            return idx, resp

        all_futures = []
        for i, prompt_text in enumerate(prompts):
            inp = ModelInput.from_ints(self.tokenizer.encode(prompt_text))
            all_futures.append(sample_with_index(i, inp))
            
        # Run all generations (tinker client handles concurrency/batching internally)
        
        # Grade answers and calculate pass@k
        total_correct = 0
        total_samples = 0
        pass_at_k_sums = {k: 0.0 for k in [1, 2, 4, 8, 16, 32, 64] if k <= self.config.n_samples}
        
        # We need to store responses to return them if needed, or just for final check? 
        # Actually we just need metrics.
        
        # tqdm_asyncio.as_completed returns a generator. To use set_postfix, we need the pbar instance.
        # We can iterate over the generator, but we can't call set_postfix on it directly.
        # Instead, let's use a manual tqdm bar and update it manually.
        from tqdm import tqdm
        
        pbar = tqdm(total=len(all_futures), desc=f"Evaluating {self.config.dataset_name}")
        for future in tqdm_asyncio.as_completed(all_futures):
            i, resp = await future
            pbar.update(1)
            
            ref = references[i]
            sequences = resp.sequences
            
            # Grade each sample for this prompt
            correct_count = 0
            for seq_idx, seq in enumerate(sequences):
                decoded_text = self.tokenizer.decode(seq.tokens)
                try:
                    extracted = extract_boxed(decoded_text)
                    is_correct = grade_answer(extracted, ref)
                    if is_correct:
                        correct_count += 1
                    
                    # Log first few samples for debugging
                    if i < 3 and seq_idx == 0:
                        logger.info(f"\nSample {i} (Problem): {prompts[i][:100]}...")
                        logger.info(f"Sample {i} (Gen): {decoded_text[:200]}...")
                        logger.info(f"Sample {i} (Extracted): {extracted}, Ref: {ref}, Correct: {is_correct}")
                        
                except Exception:
                    pass
            
            n = len(sequences)
            c = correct_count
            
            # Update global accuracy (micro-average of samples)
            total_correct += c
            total_samples += n
            
            # Calculate pass@k for this problem
            for k in pass_at_k_sums:
                if n - c < k:
                    pass_at_k_sums[k] += 1.0
                else:
                    prob_fail = math.comb(n - c, k) / math.comb(n, k)
                    pass_at_k_sums[k] += (1.0 - prob_fail)
            
            # Update progress bar with running metrics
            # We can show pass@1 and accuracy
            current_processed = total_samples / self.config.n_samples # Approximation of how many problems processed
            if current_processed > 0:
                running_pass1 = pass_at_k_sums[1] / current_processed
                running_acc = total_correct / total_samples
                pbar.set_postfix({"pass@1": f"{running_pass1:.2%}", "acc": f"{running_acc:.2%}"})
        
        metrics = {}
        num_problems = len(references)
        
        if num_problems > 0:
            # Average pass@k across problems
            for k, total_pass in pass_at_k_sums.items():
                metrics[f"{self.config.dataset_name}/pass@{k}"] = total_pass / num_problems
            
            # Also log raw accuracy (per sample)
            metrics[f"{self.config.dataset_name}/accuracy_per_sample"] = total_correct / total_samples if total_samples > 0 else 0.0
            metrics[f"{self.config.dataset_name}/num_problems"] = num_problems
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics
