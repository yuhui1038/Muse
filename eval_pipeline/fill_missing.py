#!/usr/bin/env python3
"""
Fill missing transcriptions: Check for missing transcriptions in split directory and call ASR to fill them
Usage: python fill_missing.py <split_directory> [--api_key KEY]
Example: python fill_missing.py ./audio/sunov4_5_cn
      Check for missing entries in transcription.jsonl, call ASR on missing audio and fill them
"""
import argparse, json, os, re, glob, subprocess, sys
from pathlib import Path
from tqdm import tqdm

# Import API key
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_key import get_key

def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None

def transcribe(audio_path, api_key):
    """Call qwen3-asr and filter redundant output"""
    try:
        result = subprocess.run(
            ['qwen3-asr', '-i', audio_path, '-key', api_key],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout.strip()
        
        # Filter redundant logs
        lines = output.split('\n')
        transcription = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Filter log lines
            if any(skip in line for skip in [
                "Loaded wav duration:", "DETECTED LANGUAGE", "Detected Language:",
                "FULL TRANSCRIPTION OF", "Wav duration is longer than",
                "Silero VAD model for segmenting", "saved to", "Retry", 
                "status_code", "Throttling.RateQuota"
            ]):
                continue
            # Handle Full Transcription: prefix
            if "Full Transcription:" in line:
                parts = line.split("Full Transcription:", 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    continue
            # Handle Segmenting done line
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
    parser.add_argument("input_dir", help="Split directory (contains audio and transcription.jsonl)")
    parser.add_argument("--api_key", default="", help="API Key (default: read from api_key.py)")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key if args.api_key else get_key()
    args.api_key = api_key
    
    input_dir = Path(args.input_dir)
    trans_file = input_dir / "transcription.jsonl"
    
    # Get all audio files
    audio_files = sorted(glob.glob(str(input_dir / "*.mp3")) + glob.glob(str(input_dir / "*.wav")))
    audio_indices = {}
    for f in audio_files:
        idx = extract_idx(os.path.basename(f))
        if idx is not None:
            audio_indices[idx] = f
    
    print(f"Found {len(audio_indices)} audio files")
    
    # Read existing transcriptions
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
    
    # Find missing ones
    missing = [idx for idx in audio_indices if idx not in existing]
    missing.sort()
    
    if not missing:
        print("No missing transcriptions!")
        return
    
    print(f"Missing {len(missing)} transcriptions: {missing}")
    
    # Transcribe missing ones
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
    
    # Merge and sort
    all_records = records + new_records
    all_records.sort(key=lambda x: x.get("file_idx", 999999))
    
    # Write back
    with open(trans_file, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Added {len(new_records)} transcriptions, total: {len(all_records)}")

if __name__ == "__main__":
    main()

