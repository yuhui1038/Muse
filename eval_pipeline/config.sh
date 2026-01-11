#!/bin/bash
# ========================================
# 评测配置文件
# 用法: source config.sh
# ========================================

# 待评测模型列表 (名称:音频目录)
MODELS=(
    "levo:xxx/SongGeneration/data/levo"
    # "ace-step:xxx/ACE-Step/data/ace-step"
    # "yue:xxx/YuE/inference/yue"
    # "diffRhythm2:xxx/diffrhythm2/results/diffRhythm2"
    # "..."
)

# 基础路径
BASE_DIR="Muse_Eval_Baseline"
RESULTS_DIR="$BASE_DIR/results"
GT_DIR="$BASE_DIR/gt_lyrics"
PROMPTS_DIR="$BASE_DIR/prompts"

# 模型检查点
SONGEVAL_CKPT="xxx/SongEval/ckpt/model.safetensors"
SONGEVAL_CONFIG="xxx/SongEval/config.yaml"
MUQ_MODEL="MuQ-large-msd-iter"
MULAN_MODEL="MuQ-MuLan-large"
AUDIOBOX_CKPT="xxx/audiobox-aesthetics_ckpt/checkpoint.pt"

# Conda 环境
ENV_SONGEVAL="MuQ"
ENV_AUDIOBOX="sao"
ENV_MULAN="MuQ"
ENV_PER="qwenasr"

# 默认GPU
DEFAULT_GPU=4
