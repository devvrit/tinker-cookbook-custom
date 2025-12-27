#!/bin/bash
if [ $# -lt 1 ]; then
  echo "Usage: $0 <num_steps> [feedback_model] [model_name]"
  exit 1
fi

NUM_STEPS=$1
FEEDBACK_MODEL="${2:-gemini-2.0-flash-lite}"
MODEL_NAME="${3:-Qwen/Qwen3-4B-Instruct-2507}"

python -m tinker_cookbook.recipes.distillation.feedback_self_distillation \
  model_name=${MODEL_NAME} \
  is_instruct_model=True \
  learning_rate=1e-4 \
  groups_per_batch=64 \
  group_size=8 \
  lora_rank=128 \
  eval_aime24=True \
  eval_aime25=True \
  max_steps=$NUM_STEPS \
  feedback_max_tokens=16384 \
  use_external_api=True \
  external_feedback_model=${FEEDBACK_MODEL} \
  preview_trajectories=True \
  preview_feedback=True \
  preview_proxy_teacher_prompt=True \
  save_every=50 \
  eval_every=50 \
  infrequent_eval_every=50 \
  wandb_project=feedback_self_distillation
#   max_tokens_turn1=16384 \
#   max_tokens_turn2=8192 \
#   load_checkpoint_path=tinker://6eb6acdc-66a8-54d1-b01c-d6bf5731e098:train:0/weights/final \
