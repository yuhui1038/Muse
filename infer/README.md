# Inference Guide

This directory contains scripts for generating music using the Muse model and decoding audio tokens.

## Prerequisites

- Model: Download from [https://huggingface.co/bolshyC/Muse-0.6b](https://huggingface.co/bolshyC/Muse-0.6b)
- Input data: Use `infer/test.jsonl` as the input data file

## Step 1: Generate Audio Tokens

Use `batch_multi_generate.py` to generate audio tokens from the input data.

```bash
python infer/batch_multi_generate.py \
    --input_path infer/test.jsonl \
    --output_dir <output_directory> \
    --ckpt_dir <path_to_Muse-0.6b_model> \
    --repetition_penalty 1.1 \
    --batch_size 8 \
    --log_path error.log
```

**Parameters:**
- `--input_path`: Path to input JSONL file (e.g., `infer/test.jsonl`)
- `--output_dir`: Directory to save generated outputs
- `--ckpt_dir`: Path to the downloaded Muse-0.6b model checkpoint
- `--repetition_penalty`: Repetition penalty coefficient (default: 1.1)
- `--batch_size`: Batch size for generation (default: 8)
- `--log_path`: Path to error log file (default: error.log)

The script will generate a JSONL file containing audio tokens in the output directory.

## Step 2: Decode Audio Tokens to Audio

Use `decode_audio.py` to decode the generated audio tokens back to audio files.

```bash
python infer/decode_audio.py
```

**Note:** You may need to modify the paths in `decode_audio.py` to point to:
- The generated token files from Step 1
- The MuCodec checkpoint path (default: `infer/ckpt/mucodec.pt`)

The decoded audio files will be saved as WAV files.

## Example Workflow

1. Download the model from Hugging Face:
   ```bash
   # Using huggingface-cli
   huggingface-cli download bolshyC/Muse-0.6b --local-dir ./models/Muse-0.6b
   ```

2. Generate audio tokens:
   ```bash
   python infer/batch_multi_generate.py \
       --input_path infer/test.jsonl \
       --output_dir ./outputs \
       --ckpt_dir ./models/Muse-0.6b \
       --repetition_penalty 1.1 \
       --batch_size 8
   ```

3. Decode tokens to audio:
   ```bash
   # Modify paths in decode_audio.py as needed, then run:
   python infer/decode_audio.py
   ```

## Input Data Format

The input JSONL file should contain messages in the following format:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": ""},
    ...
  ]
}
```

See `infer/test.jsonl` for example format.

