#!/bin/bash
set -euo pipefail

# 定位到脚本所在目录，保证相对路径一致
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SONG_DIR="$SCRIPT_DIR/example/zh_songs"

if [ ! -d "$SONG_DIR" ]; then
  echo "未找到歌曲目录: $SONG_DIR"
  exit 1
fi

# 收集所有 song_*.jsonl 文件
shopt -s nullglob
SONG_FILES=("$SONG_DIR"/song_*.jsonl)
shopt -u nullglob

if [ ${#SONG_FILES[@]} -eq 0 ]; then
  echo "歌曲目录中没有 song_*.jsonl 文件"
  exit 0
fi

export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"

espeak-ng --version

# 可复现性相关设置：
# - 固定随机种子 SEED
# - DO_SAMPLE=0 时尽量走确定性路径（包括固定 style prompt 裁剪起点）
SEED="${SEED:-42}"
DO_SAMPLE="${DO_SAMPLE:-0}"

# 进一步减少 cuBLAS 的非确定性（需要时可启用；若引发报错可注释掉）
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for SONG_FILE in "${SONG_FILES[@]}"; do
  SONG_NAME="$(basename "$SONG_FILE")"
  INPUT_PATH="./example/zh_songs/${SONG_NAME}"
  echo "=============================="
  echo "开始生成: ${SONG_NAME}"
  CMD=(python inference.py
    --repo-id ASLP-lab/DiffRhythm2
    --output-dir ./results/zh
    --input-jsonl "$INPUT_PATH"
    --cfg-strength 3.0
    --max-secs 285.0
    --seed "$SEED"
  )
  if [ "$DO_SAMPLE" -eq 1 ]; then
    CMD+=(--do-sample)
  fi
  "${CMD[@]}"
done

echo "全部歌曲生成完成，共处理 ${#SONG_FILES[@]} 首。"
