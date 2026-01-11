#!/usr/bin/env bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate mucodec

# =====================================================
# Task Configuration
# Format: GPU_ID:INPUT_DIR
# Note: If parallel processing is enabled, the GPU_ID here determines which GPU the task runs on.
# Different tasks with different GPU_IDs can run in parallel.
# =====================================================
TASKS=(
    "0:xxx/output_main_0.6b_50pct_5e-4_1.3"
)

# Allow command line arguments to override default tasks
if [ $# -gt 0 ]; then
    TASKS=("$@")
fi

# Base output directories
PREPARE_BASE_DIR="./processed_metadata"
EVAL_RESULTS_DIR="./eval_results"
LOG_DIR="./logs_pipeline"

mkdir -p "${PREPARE_BASE_DIR}"
mkdir -p "${EVAL_RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# Define single task processing function
process_task() {
    local task_str=$1
    IFS=':' read -r GPU_ID INPUT_DIR <<< "$task_str"
    local TASK_NAME=$(basename "${INPUT_DIR}")
    local LOG_FILE="${LOG_DIR}/task_${TASK_NAME}_gpu${GPU_ID}.log"
    
    echo "[GPU ${GPU_ID}] Started task: ${TASK_NAME} (Log: ${LOG_FILE})"
    
    (
        echo "=========================================================="
        echo "Processing Task:"
        echo "  GPU: ${GPU_ID}"
        echo "  DIR: ${INPUT_DIR}"
        echo "=========================================================="
        
        if [ ! -d "${INPUT_DIR}" ]; then
            echo "[ERROR] Directory not found: ${INPUT_DIR}"
            exit 1
        fi

        # 1. Generate Metadata
        META_OUT_DIR="${PREPARE_BASE_DIR}/${TASK_NAME}"
        echo "[Step 1] Preparing Metadata..."
        python3 prepare_chunks.py \
            --input_dir "${INPUT_DIR}" \
            --output_dir "${META_OUT_DIR}"
            
        if [ $? -ne 0 ]; then
            echo "[ERROR] Step 1 failed."
            exit 1
        fi
        
        # 2. Split audio
        echo "[Step 2] Splitting Full Audio..."
        python3 split_audio_by_tokens.py "${task_str}"
        
        if [ $? -ne 0 ]; then
            echo "[ERROR] Step 2 failed."
            exit 1
        fi
        
        # 3. Evaluation
        echo "[Step 3] Evaluating..."
        META_FILES=$(find "${META_OUT_DIR}" -name "meta_*.jsonl")
        
        if [ -z "${META_FILES}" ]; then
            echo "[WARN] No valid metadata files found."
            exit 0
        fi
        
        for META_FILE in ${META_FILES}; do
            META_FILENAME=$(basename "${META_FILE}")
            RESULT_FILE="${EVAL_RESULTS_DIR}/result_${TASK_NAME}_${META_FILENAME}"
            
            echo "  Evaluating Metadata: ${META_FILENAME}"
            
            # Use CUDA_VISIBLE_DEVICES to ensure PyTorch only sees the assigned GPU
            # This way, the --gpu parameter inside the script can be set to 0 (as it's a relative index), or directly pass the physical ID
            # For simplicity, we directly pass the physical ID to the --gpu parameter without setting CUDA_VISIBLE_DEVICES environment variable,
            # unless your code only recognizes gpu:0.
            # If the code uses device = f"cuda:{args.gpu}", then directly pass the physical ID.
            
            python3 eval_split_chunks.py \
                --metadata "${META_FILE}" \
                --original_jsonl_dir "${INPUT_DIR}" \
                --task_root_dir "${INPUT_DIR}" \
                --output "${RESULT_FILE}" \
                --gpu "${GPU_ID}"
                
            if [ $? -ne 0 ]; then
                 echo "[ERROR] Evaluation failed for ${META_FILENAME}"
            fi
        done
        
        echo "[SUCCESS] Task ${TASK_NAME} Finished."
    ) > "${LOG_FILE}" 2>&1
}

# =====================================================
# Main loop - parallel launch
# =====================================================
echo "Starting parallel processing..."
echo "Logs are being saved to: ${LOG_DIR}"

PIDS=()

for task in "${TASKS[@]}"; do
    process_task "$task" &
    PID=$!
    PIDS+=($PID)
    echo "Launched PID $PID for task: $task"
done

# Wait for all tasks to complete
echo "Waiting for all tasks to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All parallel tasks finished."
