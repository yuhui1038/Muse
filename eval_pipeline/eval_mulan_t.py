#!/usr/bin/env python3
"""
Mulan-T评测: 计算音频与文本prompt的相似度
用法: python eval_mulan_t.py --input_dir <音频目录> --prompts <prompt文件> --model_name <模型名> --output <输出文件>
输出: 汇总结果 + _details.jsonl 详细结果
"""
import argparse, json, os, re, sys, glob
import librosa, torch
from tqdm import tqdm

sys.path.append("Music_eval")
from muq import MuQMuLan

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="MuQ-MuLan-large")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    with open(args.prompts) as f:
        prompts = json.load(f)
    
    model = MuQMuLan.from_pretrained(args.model).to(device).eval()
    
    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3"))
    scores = []
    details = []
    
    for f in tqdm(files, desc="Mulan-T"):
        idx = extract_idx(os.path.basename(f))
        if idx is None or idx >= len(prompts): continue
        
        try:
            wav, _ = librosa.load(f, sr=24000)
            wavs = torch.tensor(wav).unsqueeze(0).to(device)
            with torch.no_grad():
                audio_emb = model(wavs=wavs)
                text_emb = model(texts=[prompts[idx]])
                sim = model.calc_similarity(audio_emb, text_emb).item()
            scores.append(sim)
            
            details.append({
                "file": os.path.basename(f),
                "idx": idx,
                "prompt": prompts[idx],
                "similarity": sim
            })
        except Exception as e:
            print(f"Error {f}: {e}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    avg = sum(scores)/len(scores) if scores else 0
    with open(args.output, 'w') as f:
        json.dump({"model": args.model_name, "metrics": {"Mulan-T": avg}, "count": len(scores)}, f, indent=2)
    
    # 保存详细结果
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    print(f"Saved: {args.output}")
    print(f"Details: {details_file}")

if __name__ == "__main__":
    main()
