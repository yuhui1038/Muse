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

## Training Performance

### Training Time

On 8× H200 GPUs, training one epoch takes approximately **150 minutes**.
