#!/usr/bin/env python3
"""
Split audio: Split 100 songs into Chinese and English directories
Usage: python split_audio.py <input_directory> <output_directory>
Example: python split_audio.py /path/to/audio /path/to/output
      Input: 1-50 Chinese, 51-100 English
      Output: model_cn/ (renumbered to 0-49), model_en/ (renumbered to 0-49)
      Output numbering aligns with GT file_index
"""
import os, re, shutil, argparse
from pathlib import Path

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def split(src, dst):
    src, dst = Path(src), Path(dst)
    name = src.name
    cn_dir, en_dir = dst / f"{name}_cn", dst / f"{name}_en"
    cn_dir.mkdir(parents=True, exist_ok=True)
    en_dir.mkdir(parents=True, exist_ok=True)
    
    for f in sorted(src.glob("*.*")):
        if f.suffix.lower() not in ['.wav', '.mp3']: continue
        idx = extract_idx(f.name)
        if idx is None: continue
        
        if 1 <= idx <= 50:
            # Renumber to 0-49 to match GT file_index
            shutil.copy2(f, cn_dir / f"{idx-1:06d}{f.suffix}")
        elif 51 <= idx <= 100:
            # Renumber to 0-49 to match GT file_index
            shutil.copy2(f, en_dir / f"{idx-51:06d}{f.suffix}")
    
    print(f"Split {name} -> {cn_dir.name}, {en_dir.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    split(args.input_dir, args.output_dir)
