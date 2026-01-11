#!/usr/bin/env python3
"""
拆分已有歌词: 将完整目录中的 txt 歌词按中英文拆分到对应目录
用法: python split_lyrics.py <源目录> <目标基础目录>
示例: python split_lyrics.py ./audio/sunov4_5 ./audio
      -> ./audio/sunov4_5_cn/transcription.jsonl (索引 1-50 -> 0-49)
      -> ./audio/sunov4_5_en/transcription.jsonl (索引 51-100 -> 0-49)

适用场景: 在拆分音频前已经转录好了歌词 txt 文件
"""
import os, re, argparse, json
from pathlib import Path

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def read_lyrics(txt_path):
    """读取txt文件，提取歌词"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 跳过第一行如果是语言标识
        if lines and lines[0].strip().lower() in ['chinese', 'english', 'zh', 'en']:
            lines = lines[1:]
        return ' '.join(line.strip() for line in lines if line.strip())
    except:
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="包含 txt 歌词的源目录")
    parser.add_argument("dst_dir", help="目标目录 (会生成 name_cn 和 name_en)")
    args = parser.parse_args()
    
    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    name = src.name
    
    cn_dir = dst / f"{name}_cn"
    en_dir = dst / f"{name}_en"
    
    cn_trans, en_trans = [], []
    
    # 遍历所有 txt 文件
    for txt_path in sorted(src.glob("*.txt")):
        idx = extract_idx(txt_path.name)
        if idx is None:
            continue
        
        lyrics = read_lyrics(txt_path)
        
        # 推断音频扩展名
        base = txt_path.stem
        audio_ext = ".mp3"
        for ext in ['.mp3', '.wav']:
            if (src / f"{base}{ext}").exists():
                audio_ext = ext
                break
        
        if 1 <= idx <= 50:
            new_idx = idx - 1
            new_name = f"{new_idx:06d}{audio_ext}"
            cn_trans.append({
                "file_path": str(cn_dir / new_name),
                "file_name": new_name,
                "file_idx": new_idx,
                "hyp_text": lyrics
            })
        elif 51 <= idx <= 100:
            new_idx = idx - 51
            new_name = f"{new_idx:06d}{audio_ext}"
            en_trans.append({
                "file_path": str(en_dir / new_name),
                "file_name": new_name,
                "file_idx": new_idx,
                "hyp_text": lyrics
            })
    
    # 写入 transcription.jsonl
    for trans_list, out_dir, lang in [(cn_trans, cn_dir, "cn"), (en_trans, en_dir, "en")]:
        if trans_list:
            out_dir.mkdir(parents=True, exist_ok=True)
            trans_list.sort(key=lambda x: x["file_idx"])
            out_file = out_dir / "transcription.jsonl"
            with open(out_file, 'w', encoding='utf-8') as f:
                for rec in trans_list:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f"{out_file} ({len(trans_list)} songs)")
    
    print("Done!")

if __name__ == "__main__":
    main()

