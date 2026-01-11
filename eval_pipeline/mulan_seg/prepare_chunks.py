#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare chunks for decode.py
1. Reads raw conversation jsonl
2. Extracts each section's Audio Tokens and dsec Prompt
3. Writes a flat jsonl file where each line is just "<AUDIO_xxx>..." (for decode.py to consume)
4. Saves a metadata map so we know which wav corresponds to which section/prompt.
"""

import os
import re
import json
import argparse
from tqdm import tqdm

def extract_audio_codes_str(text: str):
    """Find all <AUDIO_xxx> and return them as a single string string"""
    # decode.py needs the string to contain <AUDIO_...> tags
    # It parses them using re.findall(r"<AUDIO_(\d+)>", text)
    # So we just need to keep the audio tags.
    codes = re.findall(r"<AUDIO_\d+>", text)
    return "".join(codes)

def extract_dsec(text: str):
    # Match [dsec:...] OR [desc:...]
    # Using re.DOTALL to allow newlines in the description
    match = re.search(r"\[(?:dsec|desc):(.*?)(?:\]|$)", text, re.DOTALL)
    
    # If using regex like this, we need to be careful not to capture too much if there are multiple brackets.
    # The original regex was r"\[dsec:(.*?)\]" which is non-greedy.
    # Let's support both variants.
    
    match = re.search(r"\[(dsec|desc):(.*?)]", text, re.DOTALL)
    return match.group(2).strip() if match else None

def extract_section_name(text: str):
    # Match [Section][dsec:...] or [Section][desc:...]
    match = re.search(r"\[(.*?)\]\[(?:dsec|desc):", text)
    if match:
        return match.group(1).strip()
    match = re.match(r"^\[(.*?)\]", text)
    if match:
        name = match.group(1).strip()
        if name not in ["dsec", "desc"]: return name
    return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Dir containing raw .jsonl files")
    parser.add_argument("--output_dir", required=True, help="Where to save processed jsonl for decoding")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter out files starting with "mulan_" to avoid processing result files
    jsonl_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl") and not f.startswith("mulan_")]
    
    for jsonl_file in jsonl_files:
        input_path = os.path.join(args.input_dir, jsonl_file)
        # Create a subdir in output_dir for this jsonl task
        # decode.py reads a directory of jsonl files.
        # So we might want to put the generated jsonl file directly in args.output_dir
        # But wait, decode.py does: for jsonl in input_dir...
        
        print(f"Processing {input_path}...")
        
        flat_lines = []
        metadata = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for song_idx, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                except:
                    continue
                
                messages = data.get("messages", [])
                current_prompt = None
                current_section = "Start"
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        dsec = extract_dsec(content)
                        if dsec:
                            current_prompt = dsec
                            current_section = extract_section_name(content)
                    
                    elif role == "assistant":
                        audio_str = extract_audio_codes_str(content)
                        if not audio_str:
                            continue
                        
                        if not current_prompt:
                            continue
                            
                        # This line will be fed to decode.py
                        # It just needs to contain the audio tokens.
                        flat_lines.append(audio_str)
                        
                        # Metadata to track what this line is
                        metadata.append({
                            "song_idx": song_idx,
                            "section_name": current_section,
                            "text_prompt": current_prompt,
                            "flat_index": len(flat_lines) - 1, # The line number (0-based) in the new jsonl
                            "original_jsonl": jsonl_file
                        })
                        
                        # Reset for next section
                        current_prompt = None
                        current_section = "Unknown"

        # Write the "flat" jsonl file
        # Name it specifically so we can identify it later
        flat_jsonl_name = f"flat_{jsonl_file}"
        flat_jsonl_path = os.path.join(args.output_dir, flat_jsonl_name)
        
        with open(flat_jsonl_path, 'w', encoding='utf-8') as f_out:
            for l in flat_lines:
                f_out.write(l + "\n")
                
        # Write metadata mapping
        meta_name = f"meta_{jsonl_file}"
        meta_path = os.path.join(args.output_dir, meta_name)
        with open(meta_path, 'w', encoding='utf-8') as f_meta:
            for m in metadata:
                f_meta.write(json.dumps(m, ensure_ascii=False) + "\n")
                
        print(f"  -> Generated {len(flat_lines)} chunks. Saved to {flat_jsonl_path}")
        print(f"  -> Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()

