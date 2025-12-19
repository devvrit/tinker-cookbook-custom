"""
Training script for DIFFERENT SYSTEM PROMPT prefix cache test.

This tests prefix caching when the system prompt changes between turns:
- Turn 1: system_prompt1 → user_msg1 → response1
- Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2

Turn 2 has a DIFFERENT system prompt but includes the SAME conversation from Turn 1.
This tests whether prefix caching requires exact prefix match from the very beginning.

Run with:
    python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train

With wandb logging:
    python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train wandb_project=prefix-caching

Note: This uses chz for CLI args, so use key=value syntax (not --key value)

Expected results:
- If cache requires exact prefix match: Turn 2 CANNOT use cached prefix
- Turn 2 throughput will be similar to Turn 1 (no caching benefit)
"""

import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.env import (
    PrefixCacheTestDatasetBuilder,
)
from tinker_cookbook.rl import train

# Enable info logging to see timing info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    group_size: int = 1  # Keep small since each env generates 8k tokens
    batch_size: int = 1  # Keep small for clearer timing
    learning_rate: float = 3e-5
    max_tokens: int = 8192  # Turn 1 generates ~8k tokens
    eval_every: int = 1
    save_every: int = 1
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    num_batches: int = 1  # How many batches to run


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"prefix-cache-8k-{model_name.split('/')[-1]}-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/prefix-cache-test/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = PrefixCacheTestDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name=model_name,
        renderer_name=renderer_name,
        train_group_size=cli_config.group_size,
        num_batches=cli_config.num_batches,
    )

    logger.info("=" * 70)
    logger.info("DIFFERENT SYSTEM PROMPT TEST")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {cli_config.batch_size}")
    logger.info(f"Group size: {cli_config.group_size}")
    logger.info(f"Max tokens: {cli_config.max_tokens}")
    logger.info("=" * 70)
    logger.info("Test structure:")
    logger.info("  Turn 1: system_prompt1 → user_msg1 → response1")
    logger.info("  Turn 2: system_prompt2 → user_msg1 → response1 → user_msg2 → response2")
    logger.info("")
    logger.info("Turn 2 has DIFFERENT system prompt but SAME conversation content.")
    logger.info("Tests if prefix caching requires exact match from start.")
    logger.info("=" * 70)

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
