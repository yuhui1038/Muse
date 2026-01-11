#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate <conda_env>

export NCCL_DEBUG=WARN

export ARNOLD_WORKER_GPU=8        # 本节点 GPU 数
export ARNOLD_WORKER_NUM=1        # 总节点数
export ARNOLD_ID=0                # 当前节点编号
export ARNOLD_WORKER_0_HOST=127.0.0.1
export ARNOLD_WORKER_0_PORT=29500

# 分布式训练核心环境变量设置
export NPROC_PER_NODE=$ARNOLD_WORKER_GPU  # 每节点GPU数量
export MASTER_PORT=${ARNOLD_WORKER_0_PORT:-29500}  # 主节点端口，默认29500
export NNODES=$ARNOLD_WORKER_NUM  # 总节点数量
export NODE_RANK=$ARNOLD_ID  # 当前节点编号（0为第一节点）
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST  # 主节点IP地址
export LOCAL_WORLD_SIZE=$ARNOLD_WORKER_GPU  # 本地GPU数量
export WORLD_SIZE=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))  # 总GPU数量（节点数×每节点GPU数）

# 训练任务配置
export RUN_NAME="Muse_0.6b_main_5e-4"
MODEL_PATH="qwen3-0.6B-music"
OUTPUT_DIR="${RUN_NAME}"

# 创建输出目录（仅主节点需要执行，避免多节点同时创建可能的冲突）
if [ $NODE_RANK -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
    echo "Starting multi-node training with $NNODES nodes, $NPROC_PER_NODE GPUs each"
    echo "Total GPUs: $WORLD_SIZE"
fi

# 等待主节点创建目录完成（简单的同步机制）
sleep 5

# 启动分布式训练
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
