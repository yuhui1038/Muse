#!/usr/bin/env python3
"""
SongEval Evaluation: Evaluate audio quality in 5 dimensions
Usage: python eval_songeval.py --input_dir <audio_directory> --model_name <model_name> --output <output_file>
Output: Summary results + _details.jsonl detailed results
"""
import argparse, json, os, sys, glob, re
import librosa, torch
from tqdm import tqdm

sys.path.append("SongEval")
sys.path.append("xxx/MuQ/src")
from hydra.utils import instantiate
from muq import MuQ
from omegaconf import OmegaConf
from safetensors.torch import load_file

METRICS = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default="xxx/SongEval/ckpt/model.safetensors")
    parser.add_argument("--config", default="xxx/SongEval/config.yaml")
    parser.add_argument("--muq", default="xxx/MuQ-large-msd-iter")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # Load model
    config = OmegaConf.load(args.config)
    model = instantiate(config.generator).to(device).eval()
    model.load_state_dict(load_file(args.ckpt, device="cpu"), strict=False)
    muq = MuQ.from_pretrained(args.muq).to(device).eval()
    
    # Evaluate
    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3"))
    scores_all = {m: [] for m in METRICS}
    details = []
    
    for f in tqdm(files, desc="SongEval"):
        try:
            wav, _ = librosa.load(f, sr=24000)
            audio = torch.tensor(wav).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = muq(audio, output_hidden_states=True)["hidden_states"][6]
                scores = model(features).squeeze(0)
            
            file_scores = {}
            for i, m in enumerate(METRICS):
                val = scores[i].item()
                scores_all[m].append(val)
                file_scores[m] = val
            
            details.append({
                "file": os.path.basename(f),
                "idx": extract_idx(os.path.basename(f)),
                "scores": file_scores
            })
        except Exception as e:
            print(f"Error {f}: {e}")
        torch.cuda.empty_cache()
    
    # Save summary
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    avg = {m: sum(v)/len(v) if v else 0 for m, v in scores_all.items()}
    with open(args.output, 'w') as f:
        json.dump({"model": args.model_name, "metrics": avg, "count": len(files)}, f, indent=2)
    
    # Save detailed results
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    print(f"Saved: {args.output}")
    print(f"Details: {details_file}")

if __name__ == "__main__":
    main()
