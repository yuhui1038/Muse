#!/bin/bash

# Batch music generation example script
# Requires yue source repository, main code change is infer_batch.py replacing infer.py

# Get absolute path of script directory (to avoid filesystem mount issues)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || echo "xxx/YuE/inference")"

# Change to script directory (if possible, otherwise use absolute path)
cd "$SCRIPT_DIR" 2>/dev/null || true


# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Set PyTorch CUDA memory management optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=1

# Set JSONL file path
JSONL_PATH=""

# Set output directory
OUTPUT_DIR=""

# Set processing range (optional)
# Example: only process first 5 songs
START_IDX=0
END_IDX=-1

# Set generation parameters
MAX_NEW_TOKENS=3500
REPETITION_PENALTY=1.1
RUN_N_SEGMENTS=24
STAGE2_BATCH_SIZE=16
CUDA_IDX=0
SEED=42
NO_SAMPLE=0

# Run batch generation (using absolute path)
python "$SCRIPT_DIR/infer_batch.py" \
    --jsonl_path "$JSONL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --max_new_tokens $MAX_NEW_TOKENS \
    --repetition_penalty $REPETITION_PENALTY \
    --run_n_segments $RUN_N_SEGMENTS \
    --stage2_batch_size $STAGE2_BATCH_SIZE \
    --cuda_idx $CUDA_IDX \
    --seed $SEED \
    --rescale \
    $( [ "$NO_SAMPLE" -eq 1 ] && echo "--no_sample" )

