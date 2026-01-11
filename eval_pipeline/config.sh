#!/bin/bash
# ========================================
# Evaluation Configuration File
# Usage: source config.sh
# ========================================

# Models to evaluate (name:audio_directory)
MODELS=(
    "levo:xxx/SongGeneration/data/levo"
    # "ace-step:xxx/ACE-Step/data/ace-step"
    # "yue:xxx/YuE/inference/yue"
    # "diffRhythm2:xxx/diffrhythm2/results/diffRhythm2"
    # "..."
)

# Base paths
BASE_DIR="eval_pipeline"
RESULTS_DIR="$BASE_DIR/results"
GT_DIR="$BASE_DIR/gt_lyrics"
PROMPTS_DIR="$BASE_DIR/prompts"

# Model checkpoints
SONGEVAL_CKPT="xxx/SongEval/ckpt/model.safetensors"
SONGEVAL_CONFIG="xxx/SongEval/config.yaml"
MUQ_MODEL="MuQ-large-msd-iter"
MULAN_MODEL="MuQ-MuLan-large"
AUDIOBOX_CKPT="xxx/audiobox-aesthetics_ckpt/checkpoint.pt"

# Conda environments
ENV_SONGEVAL="xxx"
ENV_AUDIOBOX="xxx"
ENV_MULAN="xxx"
ENV_PER="xxx"

# Default GPU
DEFAULT_GPU=0
