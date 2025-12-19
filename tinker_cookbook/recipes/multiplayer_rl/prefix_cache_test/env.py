"""
Prefix Cache Test Environment - Different System Prompt Version

This recipe tests prefix caching when the SYSTEM PROMPT changes between turns:
- Turn 1: system_prompt1 → user_msg1 → response1
- Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2

The key insight: Turn 2 has a DIFFERENT system prompt but the SAME conversation.
This tests whether prefix caching can handle partial overlaps or requires 
exact prefix matches from the very beginning.

Expected results:
- If cache requires exact prefix match: Turn 2 cannot use cached prefix
- If cache can handle partial matches: Turn 2 might partially benefit
"""

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import chz
from tinker import ModelInput
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPTS - Different for Turn 1 vs Turn 2
# ============================================================================
SYSTEM_PROMPT_1 = """You are a helpful assistant specializing in technology and science. 
Provide detailed, accurate, and well-structured responses."""

SYSTEM_PROMPT_2 = """You are a knowledgeable expert in artificial intelligence and machine learning.
Provide comprehensive and insightful responses with technical depth."""

# ============================================================================
# USER MESSAGES - user_msg1 is shared, user_msg2 is only for Turn 2
# ============================================================================
USER_MSG_1 = """Write a very detailed, comprehensive essay about the history of artificial intelligence. 
Cover the following topics in depth:
1. The origins of AI research in the 1950s
2. The AI winters and their causes
3. The rise of machine learning
4. Deep learning breakthroughs
5. Large language models and their impact
6. Future directions and challenges

Make this essay as long and detailed as possible. Include historical dates, key researchers, 
important papers, technical concepts, and real-world applications. Do not stop until you have 
covered all topics thoroughly."""

USER_MSG_2 = "Now summarize the above essay in exactly one sentence."


class PrefixCacheTestEnv(Env):
    """
    Environment testing prefix caching with different system prompts.
    
    Turn 1: system_prompt1 → user_msg1 → response1
    Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2
    
    Turn 2 has a DIFFERENT system prompt but includes the SAME conversation from Turn 1.
    """
    
    def __init__(self, renderer: Renderer, env_id: int = 0):
        self.renderer: Renderer = renderer
        self.env_id: int = env_id
        self._step_count: int = 0
        self._turn_timings: list[float] = []
        self._turn_token_counts: list[int] = []
        self._turn_prefix_lengths: list[int] = []
        self._generation_start_time: float | None = None
        
        # Store response1 from Turn 1 to use in Turn 2
        self._response1_content: str | None = None

    def _get_stop_condition_for_turn(self, turn_num: int) -> StopCondition:
        """
        Return stop condition based on turn number.
        Turn 1: No early stopping (generate full essay)
        Turn 2: Use renderer's stop sequences for normal stopping
        """
        if turn_num == 1:
            return []  # Generate until max_tokens
        else:
            return self.renderer.get_stop_sequences()

    def _build_turn1_obs(self) -> ModelInput:
        """Build observation for Turn 1: system_prompt1 → user_msg1"""
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT_1},
            {"role": "user", "content": USER_MSG_1},
        ]
        return self.renderer.build_generation_prompt(convo)

    def _build_turn2_obs(self) -> ModelInput:
        """Build observation for Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2"""
        assert self._response1_content is not None, "response1 must be set before building Turn 2 obs"
        
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT_1},  # DIFFERENT system prompt
            {"role": "user", "content": USER_MSG_1},          # Same user_msg1
            {"role": "assistant", "content": self._response1_content},  # response1 from Turn 1
            {"role": "user", "content": USER_MSG_2},          # New user_msg2
        ]
        return self.renderer.build_generation_prompt(convo)

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        obs = self._build_turn1_obs()
        obs_len = len(obs) if hasattr(obs, '__len__') else 0
        self._turn_prefix_lengths.append(obs_len)
        
        self._generation_start_time = time.time()
        
        logger.info(f"Env {self.env_id}: Turn 1 START")
        logger.info(f"  Prefix: system_prompt1 → user_msg1 ({obs_len} tokens)")
        logger.info(f"  Expecting ~8k token generation")
        return obs, self._get_stop_condition_for_turn(1)

    async def step(self, action: Action) -> StepResult:
        step_receive_time = time.time()
        self._step_count += 1
        
        # Calculate generation time
        generation_time = None
        if self._generation_start_time is not None:
            generation_time = step_receive_time - self._generation_start_time
            self._turn_timings.append(generation_time)
        
        # Track tokens generated
        tokens_generated = len(action) if hasattr(action, '__len__') else 0
        self._turn_token_counts.append(tokens_generated)
        
        # Parse the response
        (action_message, _parse_success) = self.renderer.parse_response(action)
        
        gen_time_str = f"{generation_time:.3f}s" if generation_time else "N/A"
        
        if self._step_count == 1:
            # Turn 1 done - save response1 for use in Turn 2
            self._response1_content = action_message["content"]
            
            logger.info(f"Env {self.env_id}: Turn 1 DONE")
            logger.info(f"  Generation time: {gen_time_str}")
            logger.info(f"  Tokens generated (response1): {tokens_generated}")
            
            # Build Turn 2 observation with DIFFERENT system prompt but SAME conversation
            obs = self._build_turn2_obs()
            obs_len = len(obs) if hasattr(obs, '__len__') else 0
            self._turn_prefix_lengths.append(obs_len)
            
            self._generation_start_time = time.time()
            
            logger.info(f"Env {self.env_id}: Turn 2 START")
            logger.info(f"  Prefix: system_prompt2 → user_msg1 → response1 → user_msg2 ({obs_len} tokens)")
            logger.info(f"  NOTE: system_prompt2 is DIFFERENT from system_prompt1!")
            logger.info(f"  Expecting ~100 token generation")
            
            return StepResult(
                next_observation=obs,
                next_stop_condition=self._get_stop_condition_for_turn(2),
                episode_done=False,
                reward=0.0,
            )
        else:
            # Turn 2 done - episode complete
            logger.info(f"Env {self.env_id}: Turn 2 DONE")
            logger.info(f"  Generation time: {gen_time_str}")
            logger.info(f"  Tokens generated (response2): {tokens_generated}")
            
            # Log summary
            if len(self._turn_timings) >= 2:
                t1_time = self._turn_timings[0]
                t2_time = self._turn_timings[1]
                t1_tokens = self._turn_token_counts[0]
                t2_tokens = self._turn_token_counts[1]
                t1_prefix = self._turn_prefix_lengths[0]
                t2_prefix = self._turn_prefix_lengths[1]
                
                t1_tps = t1_tokens / t1_time if t1_time > 0 else 0
                t2_tps = t2_tokens / t2_time if t2_time > 0 else 0
                
                t1_time_per_token = t1_time / t1_tokens if t1_tokens > 0 else 0
                t2_time_per_token = t2_time / t2_tokens if t2_tokens > 0 else 0
                
                logger.info("=" * 70)
                logger.info(f"Env {self.env_id}: EPISODE SUMMARY")
                logger.info(f"  Turn 1: Prefix={t1_prefix} tokens, Generated={t1_tokens} tokens")
                logger.info(f"          Time={t1_time:.3f}s ({t1_tps:.1f} tok/s)")
                logger.info(f"  Turn 2: Prefix={t2_prefix} tokens, Generated={t2_tokens} tokens")
                logger.info(f"          Time={t2_time:.3f}s ({t2_tps:.1f} tok/s)")
                logger.info("")
                logger.info(f"  Time per token - T1: {t1_time_per_token*1000:.2f}ms, T2: {t2_time_per_token*1000:.2f}ms")
                
                if t2_tps > 0 and t1_tps > 0:
                    throughput_diff = (t2_tps / t1_tps - 1) * 100
                    logger.info(f"  Turn 2 vs Turn 1 throughput: {throughput_diff:+.1f}%")
                
                logger.info("")
                logger.info("  STRUCTURE:")
                logger.info("    Turn 1: system_prompt1 → user_msg1 → response1")
                logger.info("    Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2")
                logger.info("")
                logger.info("  KEY INSIGHT: Turn 2 has DIFFERENT system prompt but SAME conversation.")
                logger.info("  If prefix caching requires exact match from start, Turn 2 cannot use cache.")
                logger.info("=" * 70)
            
            obs = self._build_turn2_obs()
            return StepResult(
                next_observation=obs,
                next_stop_condition=self._get_stop_condition_for_turn(2),
                episode_done=True,
                reward=1.0,
            )


@dataclass(frozen=True)
class PrefixCacheTestEnvGroupBuilder(EnvGroupBuilder):
    """Builder for a group of PrefixCacheTestEnv instances."""
    renderer: Renderer
    num_envs: int
    group_id: int = 0

    async def make_envs(self) -> Sequence[Env]:
        envs = [
            PrefixCacheTestEnv(
                renderer=self.renderer,
                env_id=self.group_id * self.num_envs + i
            ) 
            for i in range(self.num_envs)
        ]
        logger.info(f"Created {self.num_envs} envs (group {self.group_id})")
        return envs


@dataclass(frozen=True)
class PrefixCacheTestDataset(RLDataset):
    """Dataset for prefix cache testing."""
    renderer: Renderer
    batch_size: int
    group_size: int
    num_batches: int = 1

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            PrefixCacheTestEnvGroupBuilder(
                renderer=self.renderer,
                num_envs=self.group_size,
                group_id=index * self.batch_size + i,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.num_batches


@chz.chz
class PrefixCacheTestDatasetBuilder(RLDatasetBuilder):
    """Builder for the prefix cache test dataset."""
    batch_size: int
    renderer_name: str
    train_group_size: int
    model_name: str
    base_url: str | None = None
    test_group_size: int = 4
    num_batches: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        player_renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        
        logger.info("=" * 70)
        logger.info("PREFIX CACHE TEST - DIFFERENT SYSTEM PROMPT")
        logger.info("=" * 70)
        logger.info("Test structure:")
        logger.info("  Turn 1: system_prompt1 → user_msg1 → response1")
        logger.info("  Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2")
        logger.info("")
        logger.info("Key difference:")
        logger.info("  - Turn 2 uses DIFFERENT system prompt (system_prompt2)")
        logger.info("  - But SAME conversation (user_msg1, response1)")
        logger.info("")
        logger.info("This tests if prefix caching requires exact prefix match from start.")
        logger.info("=" * 70)
        
        training_dataset = PrefixCacheTestDataset(
            renderer=player_renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            num_batches=self.num_batches,
        )
        test_dataset = PrefixCacheTestDataset(
            renderer=player_renderer,
            batch_size=1,
            group_size=self.test_group_size,
            num_batches=1,
        )
        return training_dataset, test_dataset
