#!/bin/bash
if [ $# -lt 1 ]; then
  echo "Usage: $0 <num_steps> [feedback_model] [model_name]"
  exit 1
fi

NUM_STEPS=$1
MODEL_NAME="${2:-Qwen/Qwen3-8B-Base}"
TEACHER_MODEL="${3:-Qwen/Qwen3-8B}"

python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=${MODEL_NAME} \
        load_checkpoint_path=tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final \
        dataset=polaris_math \
        learning_rate=1e-4 \
        groups_per_batch=64 \
        group_size=8 \
        lora_rank=128 \
        teacher_model=${TEACHER_MODEL} \
        max_tokens=24576 \
        max_steps=${NUM_STEPS} \
        wandb_project=on_policy_distillation_baseline