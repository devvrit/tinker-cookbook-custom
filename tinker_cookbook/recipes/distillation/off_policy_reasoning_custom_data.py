"""
Supervised fine-tuning for reasoning tasks using a custom JSON dataset.

This script implements standard supervised learning on a custom dataset provided in a JSON file.
The expected format for each line in the JSON file is:
{
    "conversations": [
        {"from": "user", "value": "..."},
        {"from": "assistant", "value": "..."}
    ],
    "question": "...",
    "source": null,
    "id": null
}

Example usage:
    python -m tinker_cookbook.recipes.distillation.off_policy_reasoning_custom_data \
        dataset_path=/home/devvrit03/filtered_openthoughts3.jsonl \
        max_prompts=128 \
        model_name=Qwen/Qwen3-8B-Base \
        learning_rate=1e-3 \
        batch_size=128 \
        lora_rank=128 \
        eval_aime24=True \
        eval_aime25=True \
        wandb_project=sft
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime


import chz
import datasets
import tinker
from datasets import load_dataset
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluatorBuilder
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.recipes.distillation.rl_math_evaluator import RLMathEvaluatorBuilder
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)

# os.environ['WANDB_API_KEY'] = <your_key>
# os.environ['TINKER_API_KEY'] = <your_key>

@chz.chz
class CustomJSONBuilder(ChatDatasetBuilder):
    """Builder for custom JSON dataset."""

    dataset_path: str
    test_size: int = 0
    shuffle_seed: int = 0
    max_prompts: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSON file
        conversations = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    conversations.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                    continue

        # Randomly sample if max_prompts is set
        if self.max_prompts is not None and len(conversations) > self.max_prompts:
            rng = random.Random(self.shuffle_seed)
            conversations = rng.sample(conversations, self.max_prompts)
            logger.info(f"Randomly sampled {self.max_prompts} conversations from {len(conversations)} total.")

        # Create HuggingFace dataset from the loaded data
        dataset = datasets.Dataset.from_list(conversations)

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split into train and test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            # If test_size is 0 or dataset is too small, use all data for training
            train_ds = dataset
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise default to ALL_ASSISTANT_MESSAGES
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            # Convert custom format (from/value) to standard format (role/content)
            conversations = row.get("conversations", [])
            messages: list[Message] = []
            for msg in conversations:
                role = "user" if msg["from"] == "user" else "assistant"
                # Handle potential "human" label as well just in case, similar to OpenThoughts
                if msg["from"] == "human":
                    role = "user"
                elif msg["from"] == "gpt":
                    role = "assistant"
                
                messages.append({
                    "role": role,
                    "content": msg["value"],
                })
            
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            hf_dataset=train_ds,
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )

        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                hf_dataset=test_ds,
                batch_size=len(test_ds), # Eval usually processes all at once or in larger batches
                map_fn=map_fn,
            )
        else:
            test_dataset = None

        return train_dataset, test_dataset


@chz.chz
class CLIConfig:
    """Command-line configuration for SFT on custom JSON data."""

    # Dataset configuration
    dataset_path: str

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    renderer_name: str | None = "qwen3"
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    lr_schedule: str = "linear"
    num_epochs: int = 1
    max_length: int = 16384

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 500
    save_every: int = 500
    infrequent_eval_every: int = 500
    eval_aime24: bool = False
    eval_aime25: bool = False
    max_prompts: int | None = None

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
        run_name = os.path.basename(log_path)
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"sft-custom-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.batch_size}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Create dataset builder
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=None, 
    )

    dataset_builder = CustomJSONBuilder(
        common_config=common_config,
        dataset_path=cli_config.dataset_path,
        max_prompts=cli_config.max_prompts,
    )

    infrequent_evaluator_builders = []
    if cli_config.eval_aime24:
        infrequent_evaluator_builders.append(
            RLMathEvaluatorBuilder(
                dataset_name="Maxwell-Jia/AIME_2024",
                split="train", # AIME 2024 usually only has train split in this dataset
                temperature=0.6,
                max_tokens=16384,
                n_samples=4,
                renderer_name=renderer_name,
            )
        )

    if cli_config.eval_aime25:
        infrequent_evaluator_builders.append(
            RLMathEvaluatorBuilder(
                dataset_name="math-ai/aime25",
                split="test", # AIME 2025 uses test split
                temperature=0.6,
                max_tokens=16384,
                n_samples=4,
                renderer_name=renderer_name,
            )
        )

    # Create full config
    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        infrequent_evaluator_builders=infrequent_evaluator_builders,
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
    )

    # Run training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
