#!/usr/bin/env python3
"""
Audiobox Evaluation: Evaluate audio aesthetic scores
Usage: python eval_audiobox.py --input_dir <audio_directory> --model_name <model_name> --output <output_file>
Output: Summary results + _details.jsonl detailed results
Requires sao environment
"""
import argparse, json, os, glob, subprocess, tempfile, re

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default="xxx/audiobox-aesthetics_ckpt/checkpoint.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    # Collect files
    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3"))
    
    # Write temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for p in files:
            f.write(json.dumps({"path": os.path.abspath(p)}) + '\n')
        paths_file = f.name
    
    # Run audio-aes
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        scores_file = f.name
    
    cmd = f'audio-aes "{paths_file}" --batch-size {args.batch_size}'
    if os.path.exists(args.ckpt):
        cmd += f' --ckpt "{args.ckpt}"'
    
    with open(scores_file, 'w') as out:
        subprocess.run(cmd, shell=True, stdout=out)
    
    # Parse results - match input files in order
    metrics = {"CE": [], "CU": [], "PC": [], "PQ": []}
    details = []
    
    # Read all valid score lines
    score_records = []
    with open(scores_file) as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                # Check if contains score fields
                if all(k in rec for k in ["CE", "CU", "PC", "PQ"]):
                    score_records.append(rec)
            except: pass
    
    # Match files and scores in order
    for i, file_path in enumerate(files):
        if i >= len(score_records):
            break
        
        rec = score_records[i]
        filename = os.path.basename(file_path)
        
        file_scores = {
            "CE": rec["CE"],
            "CU": rec["CU"],
            "PC": rec["PC"],
            "PQ": rec["PQ"]
        }
        file_scores["Score"] = sum(file_scores.values())
        
        for k in ["CE", "CU", "PC", "PQ"]:
            metrics[k].append(rec[k])
        
        details.append({
            "file": filename,
            "idx": extract_idx(filename),
            "scores": file_scores
        })
    
    # Calculate average
    avg = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
    avg["Score"] = sum(avg.values())
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({"model": args.model_name, "metrics": avg, "count": len(files)}, f, indent=2)
    
    # Save detailed results
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    # Cleanup
    os.unlink(paths_file)
    os.unlink(scores_file)
    print(f"Saved: {args.output}")
    print(f"Details: {details_file}")

if __name__ == "__main__":
    main()
