#!/bin/bash

# 批量音乐生成示例脚本
# 需要yue源仓库，代码改动主要是infer_batch.py替换了infer.py

# 获取脚本所在目录的绝对路径（避免文件系统挂载问题）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || echo "xxx/YuE/inference")"

# 切换到脚本目录（如果可能的话，否则使用绝对路径）
cd "$SCRIPT_DIR" 2>/dev/null || true


# 设置 HuggingFace 镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 设置 PyTorch CUDA 内存管理优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=1

# 设置JSONL文件路径
JSONL_PATH=""

# 设置输出目录
OUTPUT_DIR=""

# 设置处理范围 (可选)
# 例如: 只处理前5首歌曲
START_IDX=0
END_IDX=-1

# 设置生成参数
MAX_NEW_TOKENS=3500
REPETITION_PENALTY=1.1
RUN_N_SEGMENTS=24
STAGE2_BATCH_SIZE=16
CUDA_IDX=0
SEED=42
NO_SAMPLE=0

# 运行批量生成（使用绝对路径）
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

