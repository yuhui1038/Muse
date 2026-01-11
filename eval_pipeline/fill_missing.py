#!/usr/bin/env python3
"""
补充缺失的转录: 检查拆分目录中缺失的转录，调用ASR补上
用法: python fill_missing.py <拆分后的目录> [--api_key KEY]
示例: python fill_missing.py ./audio/sunov4_5_cn
      检查 transcription.jsonl 中缺失的条目，对缺失的音频调用 ASR 并补充
"""
import argparse, json, os, re, glob, subprocess, sys
from pathlib import Path
from tqdm import tqdm

# 导入 API key
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_key import get_key

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def transcribe(audio_path, api_key):
    """调用qwen3-asr并过滤多余输出"""
    try:
        result = subprocess.run(
            ['qwen3-asr', '-i', audio_path, '-key', api_key],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout.strip()
        
        # 过滤多余日志
        lines = output.split('\n')
        transcription = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 过滤日志行
            if any(skip in line for skip in [
                "Loaded wav duration:", "DETECTED LANGUAGE", "Detected Language:",
                "FULL TRANSCRIPTION OF", "Wav duration is longer than",
                "Silero VAD model for segmenting", "saved to", "Retry", 
                "status_code", "Throttling.RateQuota"
            ]):
                continue
            # 处理 Full Transcription: 前缀
            if "Full Transcription:" in line:
                parts = line.split("Full Transcription:", 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    continue
            # 处理 Segmenting done 行
            if "Segmenting done, total segments" in line:
                if "segments:" in line:
                    parts = line.split("segments:", 1)
                    remaining = parts[1].strip()
                    match = re.match(r'^\d+\s*(.*)', remaining)
                    if match and match.group(1):
                        line = match.group(1)
                    else:
                        continue
            transcription += line + " "
        
        return transcription.strip()
    except Exception as e:
        print(f"ASR Error {audio_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="拆分后的目录 (包含音频和 transcription.jsonl)")
    parser.add_argument("--api_key", default="", help="API Key (默认从 api_key.py 读取)")
    args = parser.parse_args()
    
    # 获取 API key
    api_key = args.api_key if args.api_key else get_key()
    args.api_key = api_key
    
    input_dir = Path(args.input_dir)
    trans_file = input_dir / "transcription.jsonl"
    
    # 获取所有音频文件
    audio_files = sorted(glob.glob(str(input_dir / "*.mp3")) + glob.glob(str(input_dir / "*.wav")))
    audio_indices = {}
    for f in audio_files:
        idx = extract_idx(os.path.basename(f))
        if idx is not None:
            audio_indices[idx] = f
    
    print(f"Found {len(audio_indices)} audio files")
    
    # 读取已有的转录
    existing = set()
    records = []
    if trans_file.exists():
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    idx = rec.get('file_idx')
                    if idx is not None:
                        existing.add(idx)
                except:
                    continue
    
    print(f"Existing transcriptions: {len(existing)}")
    
    # 找出缺失的
    missing = [idx for idx in audio_indices if idx not in existing]
    missing.sort()
    
    if not missing:
        print("No missing transcriptions!")
        return
    
    print(f"Missing {len(missing)} transcriptions: {missing}")
    
    # 转录缺失的
    new_records = []
    for idx in tqdm(missing, desc="Transcribing missing"):
        audio_path = audio_indices[idx]
        hyp_text = transcribe(audio_path, args.api_key)
        
        rec = {
            "file_path": audio_path,
            "file_name": os.path.basename(audio_path),
            "file_idx": idx,
            "hyp_text": hyp_text
        }
        new_records.append(rec)
    
    # 合并并排序
    all_records = records + new_records
    all_records.sort(key=lambda x: x.get("file_idx", 999999))
    
    # 写回
    with open(trans_file, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Added {len(new_records)} transcriptions, total: {len(all_records)}")

if __name__ == "__main__":
    main()

