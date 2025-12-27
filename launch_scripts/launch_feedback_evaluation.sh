#!/bin/bash

# Feedback evaluation script
# Baseline samples are now reused as rollouts for feedback generation (more efficient)

# Generate a timestamp for the run name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FEEDBACK_MODEL="${1:-gemini-2.0-flash-lite}"
MODEL_NAME="${2:-Qwen/Qwen3-8B}"

python -m tinker_cookbook.recipes.distillation.eval_with_feedback \
    model_name=${MODEL_NAME} \
    eval_aime24=True \
    eval_aime25=True \
    temperature=0.6 \
    max_tokens=16384 \
    max_tokens_turn1=16384 \
    max_tokens_turn2=8192 \
    n_samples=8 \
    feedback_temperature=0.0 \
    feedback_max_tokens=16384 \
    use_external_api=True \
    external_feedback_model=${FEEDBACK_MODEL} \
    log_path=logs/feedback_evaluation \
    wandb_project=feedback_evaluation \
    wandb_name="${MODEL_NAME}_${FEEDBACK_MODEL}_aime_eval_${TIMESTAMP}" \
    preview_responses=False \
    preview_feedback=False \
    max_trajectories_to_log=5
    # max_problems=10 \
    # model_path=tinker://6eb6acdc-66a8-54d1-b01c-d6bf5731e098:train:0/weights/final \
