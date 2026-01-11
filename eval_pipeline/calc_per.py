#!/usr/bin/env python3
"""
PER Calculation: Calculate Phoneme Error Rate based on transcription results and GT lyrics
Usage: python calc_per.py --hyp_file <transcription_file.jsonl> --gt_file <GT_lyrics_file.jsonl> --model_name <model_name> --output <output_file>
"""
import argparse, json, os, re
from tqdm import tqdm
import phoneme_utils

def extract_idx(filename):
    """Extract index from filename"""
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def signal_filter(text:str):
    """Symbol processing, convert all to spaces"""
    pattern = r'[ ,。"，:;&—''\'.\]\[()?\n-]'
    text = re.sub(pattern, ' ', text)
    while text.find("  ") != -1:
        text = text.replace("  ", " ")
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", required=True, help="Transcription file (jsonl)")
    parser.add_argument("--gt_file", required=True, help="GT lyrics file (jsonl)")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--output", required=True, help="Output result file")
    parser.add_argument("--offset", type=int, default=0, help="Index offset")
    args = parser.parse_args()
    
    # Load GT
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
    
    # Load transcription and calculate PER
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

                # Punctuation processing
                ref_text = signal_filter(ref_text)
                hyp_text = signal_filter(hyp_text)

                # Extract common length
                min_len = min(len(ref_text), len(hyp_text))
                ref_text = ref_text[:min_len]
                hyp_text = hyp_text[:min_len]
                
                # Convert to phonemes and calculate PER
                ref_phonemes = phoneme_utils.get_phonemes(ref_text)
                hyp_phonemes = phoneme_utils.get_phonemes(hyp_text)
                per = phoneme_utils.calc_per(ref_phonemes, hyp_phonemes)

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
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    avg_per = sum(per_scores) / len(per_scores) if per_scores else 1.0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model_name,
            "metrics": {"PER": avg_per},
            "count": len(per_scores)
        }, f, indent=2)
    
    # Save detailed results
    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    print(f"Average PER: {avg_per:.4f} ({len(per_scores)} samples)")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()

