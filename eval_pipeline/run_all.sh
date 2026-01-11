#!/bin/bash
# ========================================
# Run all evaluations at once
# Usage: ./run_all.sh [GPU]
# ========================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

GPU=${1:-$DEFAULT_GPU}
export CUDA_VISIBLE_DEVICES=$GPU

echo "=========================================="
echo "Using GPU: $GPU"
echo "       Baseline Evaluation Pipeline"
echo "=========================================="

for item in "${MODELS[@]}"; do
    NAME="${item%%:*}"
    DIR="${item#*:}"
    
    echo ">>> Evaluating model: $NAME"
    echo "    Directory: $DIR"
    
    # Split Chinese and English
    echo "[0/4] Splitting audio..."
    python "$SCRIPT_DIR/split_audio.py" "$DIR" "$SCRIPT_DIR/audio"
    
    for LANG in cn en; do
        AUDIO_DIR="$SCRIPT_DIR/audio/${NAME}_${LANG}"
        MODEL_NAME="${NAME}_${LANG}"
        
        if [ ! -d "$AUDIO_DIR" ]; then
            echo "Skipping $MODEL_NAME (directory does not exist)"
            continue
        fi
        
        # SongEval
        echo "[1/4] SongEval ($MODEL_NAME)..."
        conda run --live-stream -n $ENV_SONGEVAL \
            env CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/eval_songeval.py" \
            --input_dir "$AUDIO_DIR" --model_name "$MODEL_NAME" \
            --output "$RESULTS_DIR/songeval/${MODEL_NAME}.json" --gpu 0
        
        # Audiobox
        echo "[2/4] Audiobox ($MODEL_NAME)..."
        conda run --live-stream -n $ENV_AUDIOBOX \
            env CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/eval_audiobox.py" \
            --input_dir "$AUDIO_DIR" --model_name "$MODEL_NAME" \
            --output "$RESULTS_DIR/audiobox/${MODEL_NAME}.json"
        
        # Mulan-T
        echo "[3/4] Mulan-T ($MODEL_NAME)..."
        if [ "$LANG" == "cn" ]; then
            PROMPTS="$PROMPTS_DIR/zh.json"
        else
            PROMPTS="$PROMPTS_DIR/en.json"
        fi
        conda run --live-stream -n $ENV_MULAN \
            env CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/eval_mulan_t.py" \
            --input_dir "$AUDIO_DIR" --prompts "$PROMPTS" --model_name "$MODEL_NAME" \
            --output "$RESULTS_DIR/mulan_t/${MODEL_NAME}.json" --gpu 0
        
        # PER (call independent script)
        echo "[4/4] PER ($MODEL_NAME)..."
        "$SCRIPT_DIR/run_per.sh" "$AUDIO_DIR" "$MODEL_NAME" "$LANG" "$RESULTS_DIR/per"
    done
    echo ""
done

echo "=========================================="
echo "Evaluation completed! Generating summary table..."
python "$SCRIPT_DIR/summarize.py" --results_dir "$RESULTS_DIR" --output "$RESULTS_DIR/summary.md"
echo "=========================================="
