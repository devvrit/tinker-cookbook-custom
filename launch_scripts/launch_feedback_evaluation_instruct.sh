#!/bin/bash

# Feedback evaluation script for INSTRUCT models (no thinking phase)
# Uses <summary> tags for extracting approach summaries instead of </think> parsing
# Baseline samples are reused for summary extraction and feedback generation

# Generate a timestamp for the run name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FEEDBACK_MODEL="${1:-gemini-2.5-flash-lite}"
MODEL_NAME="${2:-Qwen/Qwen3-4B-Instruct-2507}"

python -m tinker_cookbook.recipes.distillation.eval_with_feedback_instruct \
    model_name=${MODEL_NAME} \
    eval_aime24=True \
    eval_aime25=True \
    temperature=0.6 \
    max_tokens=16000 \
    n_samples=8 \
    feedback_temperature=0.0 \
    feedback_max_tokens=16000 \
    use_external_api=True \
    external_feedback_model=${FEEDBACK_MODEL} \
    log_path=logs/feedback_evaluation_instruct \
    wandb_project=feedback_evaluation_instruct \
    wandb_name="${MODEL_NAME}_${FEEDBACK_MODEL}_instruct_eval_${TIMESTAMP}" \
    preview_responses=False \
    preview_feedback=False \
    max_trajectories_to_log=5
    # max_problems=10 \
    # model_path=tinker://YOUR_CHECKPOINT_PATH/weights/final \

