"""
Dataset utilities for on-policy self-distillation.

This module contains dataset configuration classes and environment definitions
for self-distillation where the teacher is the same model conditioned on
a ground-truth-enhanced prompt.

Supports two-step generation with interruption:
1. Step 1: Generate thinking until </think> token (stop sequence)
2. Step 2: Generate the answer until EOS
   - If </think> was found in step 1: continue from step 1 tokens
   - If </think> was NOT found: append continuation text then continue
"""

import math
from functools import partial
from typing import Literal, Sequence

import chz
import tinker
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


# Default prompt templates
DEFAULT_STUDENT_SUFFIX = " Write your answer in \\boxed{} format."

# Two-step generation defaults
DEFAULT_THINK_CONTINUATION_TEXT = "Now I have to give the answer </think>\n"
THINK_END_TOKEN = "</think>"

DEFAULT_PROXY_TEACHER_TEMPLATE = """Question: {problem}
Ground Truth Answer: {answer}

Instruction: You are an expert tutor who knows the final answer provided above. Your goal is to generate a valid, step-by-step derivation that logically leads to the result. Write the solution as if you are solving it from scratch. Do not simply state the ground truth answer at the start. Do not mention that you were provided the answer in your output text. Just produce the reasoning trace.
Output the full reasoning and summarized answer now.
"""


class SelfDistillationEnv(ProblemEnv):
    """
    Environment for self-distillation with two-step generation:
    
    Step 1: Generate thinking until </think> token (stop sequence)
    Step 2: Generate the answer until EOS
            - If </think> was found in step 1: continue from step 1 tokens
            - If </think> was NOT found: append continuation text then continue
    
    Following the multi-turn RL pattern, step() returns episode_done=False
    after step 1 to proceed to step 2 for answer generation.
    
    The environment returns zero reward since training signal comes from KL only.
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        tokenizer,
        convo_prefix: list[renderers.Message] | None = None,
        student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX,
        proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE,
        think_continuation_text: str = DEFAULT_THINK_CONTINUATION_TEXT,
    ):
        # Set format_coef to 0 since we don't use format rewards
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.problem = problem
        self.answer = answer
        self.tokenizer = tokenizer
        self.student_prompt_suffix = student_prompt_suffix
        self.proxy_teacher_template = proxy_teacher_template
        self.think_continuation_text = think_continuation_text
        
        # Two-step generation state
        self._step_count: int = 0
        self._step1_action_tokens: list[int] = []  # Store step 1 tokens for building step 2 observation
        self._initial_observation: tinker.ModelInput | None = None  # Store initial obs for step 2

    def get_question(self) -> str:
        """Returns the student prompt (just the problem + suffix)."""
        return self.problem + self.student_prompt_suffix

    def _get_step1_stop_condition(self) -> list[int]:
        """
        Stop condition for step 1: </think> token.
        Generation will stop when </think> is generated or max_tokens is reached.
        """
        return self.tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)

    def _get_step2_stop_condition(self) -> list[int]:
        """Stop condition for step 2: use renderer's stop sequences (EOS token)."""
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[int]]:
        """
        Returns initial observation for step 1 with </think> as stop sequence.
        Step 1 generates thinking and stops at </think> or max_tokens.
        """
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self._initial_observation = self.renderer.build_generation_prompt(convo)
        self._step_count = 0
        return self._initial_observation, self._get_step1_stop_condition()

    def get_proxy_teacher_prompt(self) -> str:
        """Returns the proxy teacher prompt (includes ground truth answer)."""
        return self.proxy_teacher_template.format(
            problem=self.problem,
            answer=self.answer,
        )

    def check_format(self, sample_str: str) -> bool:
        """Check if sample contains \\boxed{} format."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Not used for self-distillation - always return False."""
        return False

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        return self.answer

    async def step(self, action: Action) -> StepResult:
        """
        Two-step generation logic following multi-turn RL pattern:
        
        Step 1: Generate thinking (stops at </think> or max_tokens)
                - If </think> found: proceed to step 2 with current context
                - If </think> NOT found: proceed to step 2 with continuation text appended
        Step 2: Generate answer (stops at EOS), always episode_done=True
        """
        self._step_count += 1
        message, parse_success = self.renderer.parse_response(action)
        response_text = message["content"]
        
        if self._step_count == 1:
            # Step 1: Check for </think> token
            self._step1_action_tokens = list(action)  # Store for step 2
            assert self._initial_observation is not None, "initial_observation must be called first"
            
            if THINK_END_TOKEN in response_text:
                # </think> found - thinking complete, proceed to step 2 for answer generation
                logger.debug(f"</think> found in step 1 after {len(action)} tokens, proceeding to step 2")
                
                # Build observation for step 2: original prompt + step 1 tokens
                full_prompt_tokens = self._initial_observation.to_ints() + self._step1_action_tokens
                next_observation = tinker.ModelInput.from_ints(full_prompt_tokens)
                
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_observation,
                    next_stop_condition=self._get_step2_stop_condition(),
                    metrics={},
                )
            else:
                # </think> not found - append continuation text, then proceed to step 2
                logger.debug(f"</think> NOT found after {len(action)} tokens, appending continuation and proceeding to step 2")
                
                # Build observation: original prompt + step 1 tokens + continuation text
                continuation_tokens = self.tokenizer.encode(
                    self.think_continuation_text, add_special_tokens=False
                )
                full_prompt_tokens = (
                    self._initial_observation.to_ints() 
                    + self._step1_action_tokens 
                    + continuation_tokens
                )
                next_observation = tinker.ModelInput.from_ints(full_prompt_tokens)
                
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_observation,
                    next_stop_condition=self._get_step2_stop_condition(),
                    metrics={},
                )
        else:
            # Step 2: Answer generation complete
            logger.debug(f"Step 2 complete with {len(action)} additional tokens")
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._get_step2_stop_condition(),
                metrics={},
            )


class SelfDistillationDataset(RLDataset):
    """Dataset for self-distillation using Polaris math problems."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        tokenizer,
        convo_prefix: list[renderers.Message] | None = None,
        student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX,
        proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE,
        think_continuation_text: str = DEFAULT_THINK_CONTINUATION_TEXT,
        seed: int = 0,
        dataset_name: str = "polaris_selfdistill",
    ):
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(
            seed=seed
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.convo_prefix = convo_prefix
        self.student_prompt_suffix = student_prompt_suffix
        self.proxy_teacher_template = proxy_teacher_template
        self.think_continuation_text = think_continuation_text
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the Polaris dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                SelfDistillationEnv,
                problem,
                answer,
                self.renderer,
                self.tokenizer,
                convo_prefix=self.convo_prefix,
                student_prompt_suffix=self.student_prompt_suffix,
                proxy_teacher_template=self.proxy_teacher_template,
                think_continuation_text=self.think_continuation_text,
            ),
            num_envs=group_size,
            dataset_name=self.dataset_name,
        )


@chz.chz
class SelfDistillationDatasetBuilder(RLDatasetBuilder):
    """Builder for self-distillation dataset."""

    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    convo_prefix: list[renderers.Message] | None = None
    student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX
    proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE
    think_continuation_text: str = DEFAULT_THINK_CONTINUATION_TEXT
    seed: int = 0

    async def __call__(self) -> tuple[SelfDistillationDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = SelfDistillationDataset(
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            convo_prefix=self.convo_prefix,
            student_prompt_suffix=self.student_prompt_suffix,
            proxy_teacher_template=self.proxy_teacher_template,
            think_continuation_text=self.think_continuation_text,
            seed=self.seed,
        )

        # No test dataset for now
        return train_dataset, None
