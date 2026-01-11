#!/usr/bin/env python3
"""
ASR Transcription: Call qwen3-asr for speech recognition on audio files
Usage: python transcribe.py --input_dir <audio_directory> --output <output_file.jsonl> --api_key <API_KEY>
Requires qwenasr environment
"""
import argparse, json, os, re, glob, subprocess, sys
from tqdm import tqdm

# Import API key
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_key import get_key

def extract_idx(filename):
    """Extract index from filename (last number sequence)"""
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
    parser.add_argument("--input_dir", required=True, help="Audio directory")
    parser.add_argument("--output", required=True, help="Output transcription file (jsonl)")
    parser.add_argument("--api_key", default="", help="API Key (default: read from api_key.py)")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key if args.api_key else get_key()
    args.api_key = api_key
    
    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3"))
    print(f"Found {len(files)} audio files")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for audio_path in tqdm(files, desc="Transcribing"):
            filename = os.path.basename(audio_path)
            idx = extract_idx(filename)
            hyp_text = transcribe(audio_path, args.api_key)
            
            rec = {
                "file_path": audio_path,
                "file_name": filename,
                "file_idx": idx,
                "hyp_text": hyp_text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()

