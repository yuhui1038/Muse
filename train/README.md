## Training Framework

The training code uses [ms-swift](https://github.com/modelscope/ms-swift), a scalable lightweight infrastructure for fine-tuning large language models.

## Model Configuration

### `MODEL_PATH` Parameter

The `MODEL_PATH` in `train.sh` should point to the base model. Download the model from [HuggingFace](https://huggingface.co/datasets/bolshyC/qwen3-0.6B-music):

```bash
# Download the model using huggingface_hub
huggingface-cli download bolshyC/qwen3-0.6B-music --local-dir ./qwen3-0.6B-music
```

Then modify `MODEL_PATH` in `train.sh` to point to the local path:
```bash
MODEL_PATH="./qwen3-0.6B-music"  # or absolute path
```

## Dataset Configuration

### `--dataset` Parameter

**Note:** The current script `train.sh` uses `train_demo.jsonl` (for demonstration purposes). For actual training, you need to use the full dataset.

### Actual Training Data

For actual training, please use the following two files from the [HuggingFace dataset](https://huggingface.co/datasets/bolshyC/Muse_train):

- **`train_cn.jsonl`** - Chinese training data
- **`train_en.jsonl`** - English training data

### Usage

1. Download the dataset from HuggingFace:
```bash
# Using huggingface_hub to download
huggingface-cli download bolshyC/Muse_train train_cn.jsonl --local-dir ./data
huggingface-cli download bolshyC/Muse_train train_en.jsonl --local-dir ./data
```

2. Modify the `--dataset` parameter in `train.sh`:
```bash
# If using Chinese data only
--dataset 'data/train_cn.jsonl'

# If using both Chinese and English data (comma-separated, no spaces)
--dataset 'data/train_cn.jsonl,data/train_en.jsonl'
```

**Note:** In ms-swift, multiple dataset files should be comma-separated without spaces.

## Building Custom Training Data

If you want to build your own training dataset, you need to encode audio files into discrete tokens using MuCodec.

### Audio Encoding

Use `train/encode_audio.py` to encode audio files into discrete tokens:

1. **Prepare input data file**: Create a JSONL file where each line contains a dictionary with an audio file path:
   ```json
   {"path": "path/to/audio1.wav"}
   {"path": "path/to/audio2.mp3"}
   ```

2. **Modify paths in `encode_audio.py`**:
   - Set `DATA_PATH` to your input JSONL file path
   - Set `SAVE_DIR` to the directory where encoded tokens will be saved

3. **Run encoding**:
   ```bash
   python train/encode_audio.py
   ```

The script will:
- Load audio files from the paths specified in the JSONL file
- Encode each audio file into discrete tokens using MuCodec
- Save the encoded tokens as `.pt` files in the `SAVE_DIR` directory
- Skip files that have already been encoded

**Note:** The audio files should be in WAV or MP3 format and will be automatically resampled to 48kHz if needed.

## Training Performance

### Training Time

On 8× H200 GPUs, training one epoch takes approximately **150 minutes**.
