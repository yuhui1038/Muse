#!/bin/bash
# ========================================
# PER评测脚本 (ASR转录 + PER计算)
# 用法: ./run_per.sh <音频目录> <模型名> <语言:cn/en> [输出目录]
# 示例: ./run_per.sh ./audio/sunov4_5_cn sunov4_5_cn cn ./results/per
# ========================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

INPUT_DIR=${1:?"用法: $0 <音频目录> <模型名> <语言:cn/en> [输出目录]"}
MODEL_NAME=${2:?"用法: $0 <音频目录> <模型名> <语言:cn/en> [输出目录]"}
LANG=${3:?"用法: $0 <音频目录> <模型名> <语言:cn/en> [输出目录]"}
OUTPUT_DIR=${4:-"$RESULTS_DIR/per"}

# 选择GT文件
if [ "$LANG" == "cn" ] || [ "$LANG" == "zh" ]; then
    GT_FILE="$GT_DIR/zh.jsonl"
else
    GT_FILE="$GT_DIR/en.jsonl"
fi

# 输出文件
TRANS_FILE="$INPUT_DIR/transcription.jsonl"  # 转录文件保存在音频目录
RESULT_FILE="$OUTPUT_DIR/${MODEL_NAME}.json"

mkdir -p "$OUTPUT_DIR"

echo "=== PER 评测 ==="
echo "音频目录: $INPUT_DIR"
echo "模型名: $MODEL_NAME"
echo "语言: $LANG"

# Step 1: ASR转录
if [ -f "$TRANS_FILE" ]; then
    echo "[1/2] 使用已有转录: $TRANS_FILE"
else
    echo "[1/2] ASR转录..."
    conda run -n $ENV_PER python "$SCRIPT_DIR/transcribe.py" \
        --input_dir "$INPUT_DIR" \
        --output "$TRANS_FILE"
fi

# Step 2: 计算PER
echo "[2/2] 计算PER..."
conda run -n $ENV_PER python "$SCRIPT_DIR/calc_per.py" \
    --hyp_file "$TRANS_FILE" \
    --gt_file "$GT_FILE" \
    --model_name "$MODEL_NAME" \
    --output "$RESULT_FILE"

echo "完成!"

