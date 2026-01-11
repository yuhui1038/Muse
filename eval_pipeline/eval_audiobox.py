#!/usr/bin/env python3
"""
Audiobox评测: 评测音频美学分数
用法: python eval_audiobox.py --input_dir <音频目录> --model_name <模型名> --output <输出文件>
输出: 汇总结果 + _details.jsonl 详细结果
需在sao环境运行
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
    
    # 收集文件
    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3"))
    
    # 写临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for p in files:
            f.write(json.dumps({"path": os.path.abspath(p)}) + '\n')
        paths_file = f.name
    
    # 运行audio-aes
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        scores_file = f.name
    
    cmd = f'audio-aes "{paths_file}" --batch-size {args.batch_size}'
    if os.path.exists(args.ckpt):
        cmd += f' --ckpt "{args.ckpt}"'
    
    with open(scores_file, 'w') as out:
        subprocess.run(cmd, shell=True, stdout=out)
    
    # 解析结果 - 按顺序匹配输入文件
    metrics = {"CE": [], "CU": [], "PC": [], "PQ": []}
    details = []
    
    # 读取所有有效的评分行
    score_records = []
    with open(scores_file) as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                # 检查是否包含评分字段
                if all(k in rec for k in ["CE", "CU", "PC", "PQ"]):
                    score_records.append(rec)
            except: pass
    
    # 按顺序匹配文件和评分
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
    
    # 计算平均
    avg = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
    avg["Score"] = sum(avg.values())
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({"model": args.model_name, "metrics": avg, "count": len(files)}, f, indent=2)
    
    # 保存详细结果
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    # 清理
    os.unlink(paths_file)
    os.unlink(scores_file)
    print(f"Saved: {args.output}")
    print(f"Details: {details_file}")

if __name__ == "__main__":
    main()
