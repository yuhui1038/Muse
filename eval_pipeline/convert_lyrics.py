#!/usr/bin/env python3
"""
Convert existing lyric files to transcription.jsonl format
Usage: python convert_lyrics.py --input_dir <directory_containing_txt> --output <output_file>

Input format (xxx.txt):
    First line: Chinese/English (optional, will be ignored)
    Second line and after: Lyric content

Output format (transcription.jsonl):
    {"file_path": "...", "file_name": "xxx.mp3", "file_idx": 1, "hyp_text": "lyrics"}
"""
import argparse, json, os, re, glob
from pathlib import Path

def extract_idx(filename):
    """Extract index from filename (last number sequence)"""
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def read_lyrics(txt_path):
    """Read txt file and extract lyrics"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip first line if it's a language identifier
    if lines and lines[0].strip().lower() in ['chinese', 'english', 'zh', 'en']:
        lines = lines[1:]
    
    # Merge remaining lines as lyrics
    lyrics = ' '.join(line.strip() for line in lines if line.strip())
    return lyrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing txt lyric files")
    parser.add_argument("--output", default="", help="Output file (default: input_dir/transcription.jsonl)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = args.output if args.output else input_dir / "transcription.jsonl"
    
    # Find all txt files
    txt_files = sorted(glob.glob(str(input_dir / "*.txt")))
    print(f"Found {len(txt_files)} txt files in {input_dir}")
    
    records = []
    for txt_path in txt_files:
        txt_name = os.path.basename(txt_path)
        idx = extract_idx(txt_name)
        
        # Infer corresponding audio filename
        base_name = os.path.splitext(txt_name)[0]
        # Try to find corresponding audio file
        audio_name = None
        for ext in ['.mp3', '.wav']:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                audio_name = f"{base_name}{ext}"
                break
        if not audio_name:
            audio_name = f"{base_name}.mp3"  # Default
        
        lyrics = read_lyrics(txt_path)
        
        rec = {
            "file_path": str(input_dir / audio_name),
            "file_name": audio_name,
            "file_idx": idx,
            "hyp_text": lyrics
        }
        records.append(rec)
    
    # Sort by index
    records.sort(key=lambda x: x["file_idx"] if x["file_idx"] is not None else 999999)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(records)} files -> {output_file}")

if __name__ == "__main__":
    main()

