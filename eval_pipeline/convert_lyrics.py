#!/usr/bin/env python3
"""
转换已有歌词文件为 transcription.jsonl 格式
用法: python convert_lyrics.py --input_dir <包含txt的目录> --output <输出文件>

输入格式 (xxx.txt):
    第一行: Chinese/English (可选，会被忽略)
    第二行及之后: 歌词内容

输出格式 (transcription.jsonl):
    {"file_path": "...", "file_name": "xxx.mp3", "file_idx": 1, "hyp_text": "歌词"}
"""
import argparse, json, os, re, glob
from pathlib import Path

def extract_idx(filename):
    """从文件名提取索引（最后一个数字序列）"""
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def read_lyrics(txt_path):
    """读取txt文件，提取歌词"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过第一行如果是语言标识
    if lines and lines[0].strip().lower() in ['chinese', 'english', 'zh', 'en']:
        lines = lines[1:]
    
    # 合并剩余行作为歌词
    lyrics = ' '.join(line.strip() for line in lines if line.strip())
    return lyrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="包含txt歌词文件的目录")
    parser.add_argument("--output", default="", help="输出文件 (默认: input_dir/transcription.jsonl)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = args.output if args.output else input_dir / "transcription.jsonl"
    
    # 查找所有txt文件
    txt_files = sorted(glob.glob(str(input_dir / "*.txt")))
    print(f"Found {len(txt_files)} txt files in {input_dir}")
    
    records = []
    for txt_path in txt_files:
        txt_name = os.path.basename(txt_path)
        idx = extract_idx(txt_name)
        
        # 推断对应的音频文件名
        base_name = os.path.splitext(txt_name)[0]
        # 尝试查找对应的音频文件
        audio_name = None
        for ext in ['.mp3', '.wav']:
            candidate = input_dir / f"{base_name}{ext}"
            if candidate.exists():
                audio_name = f"{base_name}{ext}"
                break
        if not audio_name:
            audio_name = f"{base_name}.mp3"  # 默认
        
        lyrics = read_lyrics(txt_path)
        
        rec = {
            "file_path": str(input_dir / audio_name),
            "file_name": audio_name,
            "file_idx": idx,
            "hyp_text": lyrics
        }
        records.append(rec)
    
    # 按索引排序
    records.sort(key=lambda x: x["file_idx"] if x["file_idx"] is not None else 999999)
    
    # 写入输出
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(records)} files -> {output_file}")

if __name__ == "__main__":
    main()

