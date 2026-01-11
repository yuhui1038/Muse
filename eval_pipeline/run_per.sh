#!/bin/bash
# ========================================
# PER Evaluation Script (ASR Transcription + PER Calculation)
# Usage: ./run_per.sh <audio_directory> <model_name> <language:cn/en> [output_directory]
# Example: ./run_per.sh ./audio/sunov4_5_cn sunov4_5_cn cn ./results/per
# ========================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

INPUT_DIR=${1:?"Usage: $0 <audio_directory> <model_name> <language:cn/en> [output_directory]"}
MODEL_NAME=${2:?"Usage: $0 <audio_directory> <model_name> <language:cn/en> [output_directory]"}
LANG=${3:?"Usage: $0 <audio_directory> <model_name> <language:cn/en> [output_directory]"}
OUTPUT_DIR=${4:-"$RESULTS_DIR/per"}

# Select GT file
if [ "$LANG" == "cn" ] || [ "$LANG" == "zh" ]; then
    GT_FILE="$GT_DIR/zh.jsonl"
else
    GT_FILE="$GT_DIR/en.jsonl"
fi

# Output files
TRANS_FILE="$INPUT_DIR/transcription.jsonl"  # Transcription file saved in audio directory
RESULT_FILE="$OUTPUT_DIR/${MODEL_NAME}.json"

mkdir -p "$OUTPUT_DIR"

echo "=== PER Evaluation ==="
echo "Audio directory: $INPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "Language: $LANG"

# Step 1: ASR Transcription
if [ -f "$TRANS_FILE" ]; then
    echo "[1/2] Using existing transcription: $TRANS_FILE"
else
    echo "[1/2] ASR transcription..."
    conda run -n $ENV_PER python "$SCRIPT_DIR/transcribe.py" \
        --input_dir "$INPUT_DIR" \
        --output "$TRANS_FILE"
fi

# Step 2: Calculate PER
echo "[2/2] Calculating PER..."
conda run -n $ENV_PER python "$SCRIPT_DIR/calc_per.py" \
    --hyp_file "$TRANS_FILE" \
    --gt_file "$GT_FILE" \
    --model_name "$MODEL_NAME" \
    --output "$RESULT_FILE"

echo "Completed!"

