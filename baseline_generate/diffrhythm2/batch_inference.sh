#!/bin/bash
set -euo pipefail

# Navigate to script directory to ensure relative paths are consistent
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SONG_DIR="$SCRIPT_DIR/example/zh_songs"

if [ ! -d "$SONG_DIR" ]; then
  echo "Song directory not found: $SONG_DIR"
  exit 1
fi

# Collect all song_*.jsonl files
shopt -s nullglob
SONG_FILES=("$SONG_DIR"/song_*.jsonl)
shopt -u nullglob

if [ ${#SONG_FILES[@]} -eq 0 ]; then
  echo "No song_*.jsonl files in song directory"
  exit 0
fi

export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"

espeak-ng --version

# Reproducibility settings:
# - Fixed random seed SEED
# - DO_SAMPLE=0 tries to follow deterministic path (including fixed style prompt cropping start)
SEED="${SEED:-42}"
DO_SAMPLE="${DO_SAMPLE:-0}"

# Further reduce cuBLAS non-determinism (enable when needed; comment out if causes errors)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for SONG_FILE in "${SONG_FILES[@]}"; do
  SONG_NAME="$(basename "$SONG_FILE")"
  INPUT_PATH="./example/zh_songs/${SONG_NAME}"
  echo "=============================="
  echo "Starting generation: ${SONG_NAME}"
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

echo "All songs generation complete, processed ${#SONG_FILES[@]} songs."
