#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate <conda_env>

export NCCL_DEBUG=WARN

export ARNOLD_WORKER_GPU=8        # Number of GPUs per node
export ARNOLD_WORKER_NUM=1        # Total number of nodes
export ARNOLD_ID=0                # Current node ID
export ARNOLD_WORKER_0_HOST=127.0.0.1
export ARNOLD_WORKER_0_PORT=29500

# Core environment variables for distributed training
export NPROC_PER_NODE=$ARNOLD_WORKER_GPU  # Number of GPUs per node
export MASTER_PORT=${ARNOLD_WORKER_0_PORT:-29500}  # Master node port, default 29500
export NNODES=$ARNOLD_WORKER_NUM  # Total number of nodes
export NODE_RANK=$ARNOLD_ID  # Current node ID (0 is the first node)
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST  # Master node IP address
export LOCAL_WORLD_SIZE=$ARNOLD_WORKER_GPU  # Local GPU count
export WORLD_SIZE=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))  # Total GPU count (nodes × GPUs per node)

# Training task configuration
export RUN_NAME="Muse_0.6b_main_5e-4"
MODEL_PATH="qwen3-0.6B-music"
OUTPUT_DIR="${RUN_NAME}"

# Create output directory (only master node needs to execute, avoid conflicts from simultaneous creation)
if [ $NODE_RANK -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
    echo "Starting multi-node training with $NNODES nodes, $NPROC_PER_NODE GPUs each"
    echo "Total GPUs: $WORLD_SIZE"
fi

# Wait for master node to finish creating directory (simple synchronization mechanism)
sleep 5

# Start distributed training
swift sft \
    --model ${MODEL_PATH} \
    --train_type full \
    --model_type qwen3 \
    --dataset 'train/train_demo.jsonl'\
    --val_dataset 'train/val.jsonl' \
    --num_train_epochs 20 \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps -1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 200 \
    --save_only_model true \
    --logging_steps 1 \
    --max_length 15000 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32 \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
    --report_to tensorboard \
    2>&1 | tee ${OUTPUT_DIR}/train_node_${NODE_RANK}.log
