#!/usr/bin/env python3
"""
Batch music generation model inference using vLLM (supports context dialogue + batch processing)
Command line arguments:
    --input_path <input JSONL>
    --output_dir <output directory>
    --ckpt_dir <checkpoint directory>
    --batch_size <batch size, default 8>
"""

import os
import json
import argparse
from tqdm import tqdm
from itertools import islice
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--repetition_penalty", type=float, required=True, help="Repetition penalty coefficient")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--log_path", type=str, default="error.log", help="Error log file path")
    return parser.parse_args()

def batched(iterator, n):
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            return
        yield batch

def batch_generate(llm, tokenizer, batch_histories, gen_kwargs):
    texts = [ 
        tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        for history in batch_histories
    ]
    sampling_params = SamplingParams(**gen_kwargs)
    outputs = llm.generate(texts, sampling_params)
    replies = [out.outputs[0].text.strip() for out in outputs]
    return replies

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_name = os.path.basename(args.ckpt_dir.rstrip("/"))
    output_path = os.path.join(args.output_dir, f"generate_multi_{ckpt_name}.jsonl")

    # Skip if output file already exists
    if os.path.exists(output_path):
        print(f"⚠️ Detected {output_path} already exists, skipping this checkpoint.")
        return

    GEN_KWARGS = dict(max_tokens=3000, temperature=0, top_p=0.9, repetition_penalty=args.repetition_penalty)

    print(f"🚀 Loading model with vLLM: {args.ckpt_dir}")
    llm = LLM(model=args.ckpt_dir, enforce_eager=True, dtype="float32", tensor_parallel_size=1, trust_remote_code=True, max_model_len=20000, gpu_memory_utilization=0.8)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)

    print(f"Reading data: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out, open(args.log_path, "a", encoding="utf-8") as f_log:
        for lines in tqdm(batched(f_in, args.batch_size), desc=f"Batch generating ({ckpt_name})"):
            samples = [json.loads(line) for line in lines]
            
            for sample in samples:
                try: 
                    messages = sample["messages"]
                    new_messages, history = [], []
                    for msg in messages:
                        if msg["role"] == "user":
                            history.append(msg)
                            new_messages.append(msg)
                        elif msg["role"] == "assistant":
                            replies = batch_generate(llm, tokenizer, [history], GEN_KWARGS)
                            reply = replies[0]
                            new_messages.append({"role": "assistant", "content": reply})
                            history.append({"role": "assistant", "content": reply})
                    sample["messages"] = new_messages
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    f_out.write("__VLLM_FAIL__" + json.dumps(sample, ensure_ascii=False) + "\n")

                    # Log error
                    f_log.write("\n========== ERROR DETECTED ==========\n")
                    f_log.write(f"Error sample: {sample}\n")
                    f_log.write(f"Error reason: {str(e)}\n")
                    f_log.write("====================================\n")
                    f_log.flush()
                    continue
                

    print(f"✅ Generated results: {output_path}")

if __name__ == "__main__":
    main()
