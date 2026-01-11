#!/usr/bin/env python3
"""
PER计算: 根据转录结果和GT歌词计算音素错误率
用法: python calc_per.py --hyp_file <转录文件.jsonl> --gt_file <GT歌词.jsonl> --model_name <模型名> --output <输出文件>
"""
import argparse, json, os, re
from tqdm import tqdm
import phoneme_utils

def extract_idx(filename):
    """从文件名提取索引"""
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def signal_filter(text:str):
    """符号处理，全部变成空格"""
    pattern = r'[ ,。"，:;&—‘’\'.\]\[()?\n-]'
    text = re.sub(pattern, ' ', text)
    while text.find("  ") != -1:
        text = text.replace("  ", " ")
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", required=True, help="转录文件 (jsonl)")
    parser.add_argument("--gt_file", required=True, help="GT歌词文件 (jsonl)")
    parser.add_argument("--model_name", required=True, help="模型名称")
    parser.add_argument("--output", required=True, help="输出结果文件")
    parser.add_argument("--offset", type=int, default=0, help="索引偏移量")
    args = parser.parse_args()
    
    # 加载GT
    gt = {}
    with open(args.gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                idx = rec.get('file_index')
                if idx is not None:
                    gt[idx] = rec['lyrics']
            except:
                continue
    print(f"Loaded {len(gt)} ground truth lyrics")
    
    # 加载转录并计算PER
    per_scores = []
    details = []
    
    with open(args.hyp_file, 'r', encoding='utf-8') as f:
        for id, line in tqdm(enumerate(f), desc="Calculating PER"):
            try:
                rec = json.loads(line)
                hyp_text = rec.get('hyp_text', '')
                filename = rec.get('file_name', '')
                
                idx = rec.get('file_idx')
                if idx is None:
                    idx = extract_idx(filename)
                if idx is None:
                    continue
                
                gt_idx = idx + args.offset
                if gt_idx not in gt:
                    continue
                
                ref_text = gt[gt_idx]

                # 标点处理(yzx)
                ref_text = signal_filter(ref_text)
                hyp_text = signal_filter(hyp_text)

                # 截取公共长度(yzx)
                min_len = min(len(ref_text), len(hyp_text))
                ref_text = ref_text[:min_len]
                hyp_text = hyp_text[:min_len]
                
                # 转换为音素并计算PER
                ref_phonemes = phoneme_utils.get_phonemes(ref_text)
                hyp_phonemes = phoneme_utils.get_phonemes(hyp_text)
                per = phoneme_utils.calc_per(ref_phonemes, hyp_phonemes)
                
                # 阈值检查(yzx)
                # while per > 0.3 and min_len > 20:
                #     min_len //= 2
                #     ref_text = ref_text[:min_len]
                #     hyp_text = hyp_text[:min_len]
                #     ref_phonemes = phoneme_utils.get_phonemes(ref_text)
                #     hyp_phonemes = phoneme_utils.get_phonemes(hyp_text)
                #     per = phoneme_utils.calc_per(ref_phonemes, hyp_phonemes)
                # if min_len <= 20:
                #     continue
 
                # if per > 0.5:
                #     print(id, per)
                #     print(f"gt: {ref_text}\nge: {hyp_text}\n")

                per_scores.append(per)
                details.append({
                    "file": filename,
                    "idx": idx,
                    "per": per,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "ref_phonemes": " ".join(ref_phonemes),
                    "hyp_phonemes": " ".join(hyp_phonemes)
                })
            except:
                continue
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    avg_per = sum(per_scores) / len(per_scores) if per_scores else 1.0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model_name,
            "metrics": {"PER": avg_per},
            "count": len(per_scores)
        }, f, indent=2)
    
    # 保存详细结果
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    print(f"Average PER: {avg_per:.4f} ({len(per_scores)} samples)")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()

