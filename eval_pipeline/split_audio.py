#!/usr/bin/env python3
"""
拆分音频: 将100首歌拆分为中英文两个目录
用法: python split_audio.py <输入目录> <输出目录>
示例: python split_audio.py /path/to/audio /path/to/output
      输入: 1-50 中文, 51-100 英文
      输出: model_cn/ (重编号为 0-49), model_en/ (重编号为 0-49)
      输出编号与 GT file_index 对齐
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
            # 重编号为 0-49 以匹配 GT 的 file_index
            shutil.copy2(f, cn_dir / f"{idx-1:06d}{f.suffix}")
        elif 51 <= idx <= 100:
            # 重编号为 0-49 以匹配 GT 的 file_index
            shutil.copy2(f, en_dir / f"{idx-51:06d}{f.suffix}")
    
    print(f"Split {name} -> {cn_dir.name}, {en_dir.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    split(args.input_dir, args.output_dir)
