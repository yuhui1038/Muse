#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split full audio files into chunks based on token counts from jsonl files.
Configuration is similar to run_chunk_decode_v2_2.sh tasks.

Usage:
    python split_audio_by_tokens.py "TASK_STRING" "TASK_STRING" ...

Task String Format:
    GPU_ID:JSONL_DIR
    (GPU_ID is ignored here but kept for compatibility with the config format)

Logic:
    1. Read the jsonl file to get token counts for each segment.
    2. Calculate start and end time for each segment (25 tokens = 1 second).
    3. Load the corresponding FULL audio file.
       - It assumes the full audio is located at: JSONL_DIR/subdir_name/000000.wav
       - Or adjacent to the jsonl file if structure is different.
       - Based on user input: /.../output_xxx/generate_multi.../000000.wav
    4. Slice the audio and save as 000000.wav, 000001.wav, etc. in a new 'chunks' subdirectory.
"""

import os
import sys
import re
import json
import argparse
import torchaudio
import torch
from tqdm import tqdm

SAMPLE_RATE = 48000
TOKEN_RATE = 25.0

def extract_audio_codes(text: str):
    """Extract <AUDIO_XXXX> -> [int, ...]"""
    return [int(x) for x in re.findall(r"<AUDIO_(\d+)>", text)]

def process_task(task_str):
    parts = task_str.split(":", 1)
    if len(parts) != 2:
        print(f"[WARN] Invalid task format: {task_str}")
        return
    
    # input_dir contains the jsonl files
    input_dir = parts[1]
    
    if not os.path.isdir(input_dir):
        print(f"[WARN] Directory not found: {input_dir}")
        return

    print(f"\n[INFO] Processing directory: {input_dir}")
    
    # Find all jsonl files
    jsonl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jsonl")])
    
    if not jsonl_files:
        print(f"[WARN] No jsonl files found in {input_dir}")
        return

    for jsonl_name in jsonl_files:
        jsonl_path = os.path.join(input_dir, jsonl_name)
        # Assuming folder structure:
        # input_dir/
        #   task_name.jsonl
        #   task_name/ (contains 000000.wav - the full audio)
        
        subdir_name = os.path.splitext(jsonl_name)[0]
        full_audio_dir = os.path.join(input_dir, subdir_name)
        
        # Where to save the chunks? 
        # Let's create a 'chunks' folder inside the full_audio_dir
        output_dir = os.path.join(full_audio_dir, "chunks")
        os.makedirs(output_dir, exist_ok=True)
            
        print(f"  -> Processing {jsonl_name}")
        print(f"     Full Audio Dir: {full_audio_dir}")
        print(f"     Output Dir: {output_dir}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Filter empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        for idx, line in enumerate(tqdm(lines, desc="Splitting")):
            # Construct expected full audio filename for this line
            # Assuming standard naming: {idx:06d}.wav
            full_audio_filename = f"{idx:06d}.wav"
            full_audio_path = os.path.join(full_audio_dir, full_audio_filename)

            if not os.path.isfile(full_audio_path):
                # Optionally warn, but might span stdout too much
                # print(f"[SKIP] Audio not found: {full_audio_path}")
                continue

            # Load full audio for this specific line/song
            try:
                waveform, sr = torchaudio.load(full_audio_path)
            except Exception as e:
                print(f"[ERROR] Failed to load audio {full_audio_path}: {e}")
                continue
                
            if sr != SAMPLE_RATE:
                # Resample if necessary
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = SAMPLE_RATE

            # Extract tokens to calculate duration
            # ... (rest of processing logic)
            
            # If it's a JSON line (raw format), we need to extract assistant content
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    # Extract text from assistant messages
                    text_content = ""
                    for msg in data.get("messages", []):
                        if msg.get("role") == "assistant":
                            text_content += msg.get("content", "")
                    
                    # If multiple assistant messages, this might be tricky. 
                    # But usually it's conversation turns.
                    # Wait, if we are splitting a FULL audio generated from a FULL jsonl,
                    # we need to know which part of the audio corresponds to which line.
                    # Usually 'generate.py' generates one audio per LINE in the input jsonl.
                    # If the input jsonl has multiple lines, generate.py outputs 000000.wav, 000001.wav...
                    
                    # CASE A: The "full audio" 000000.wav corresponds to ONE line in the jsonl.
                    # But that line contains multiple <AUDIO_...> chunks (multi-turn or long generation).
                    # If so, we split that single wav file into sub-segments based on tokens IN THAT LINE.
                    
                    # CASE B: The jsonl has multiple lines, and we have multiple wavs.
                    # But the user query says "audio decoded after sounds different".
                    # The user wants to split the "Full Song" into "Chunks".
                    # Usually a song is ONE line in the inference jsonl.
                    
                    tokens = extract_audio_codes(text_content)
                    
                except json.JSONDecodeError:
                    # Maybe it's not json, try direct extraction
                    tokens = extract_audio_codes(line)
            else:
                tokens = extract_audio_codes(line)
            
            if not tokens:
                print(f"[WARN] No tokens found in line {idx}")
                continue
                
            # If we are splitting ONE file (000000.wav) based on sub-parts, 
            # we need to know if the jsonl represents the structure of that one file.
            # Assuming the standard use case:
            # The jsonl has 1 line (the song). That line has multiple <AUDIO> segments?
            # OR the jsonl has multiple lines (segments) that were somehow merged?
            
            # Re-reading user context: 
            # "Files are decoded in batches... audio length is not 40.96s... there may be subtle differences compared to decoding a complete song"
            # It implies we want to take the "Golden Reference" (Full Song) 
            # and chop it up to match the "Chunks" so we can evaluate them.
            
            # The "Chunks" are defined by `prepare_chunks.py`.
            # That script took a raw jsonl and split it into many small lines (flat_xxx.jsonl).
            # So `flat_xxx.jsonl` has N lines.
            # The original raw jsonl had M lines (songs).
            
            # We need to map the "Full Wav" (from original jsonl) to the "Chunks" (in flat jsonl).
            # But here we are given the directory of the jsonl files.
            # If we look at `prepare_chunks.py`, it calculates duration based on tokens.
            
            # Strategy:
            # 1. We iterate through the segments in the line (if it's a multi-segment line).
            #    Wait, standard MuCodec generation usually outputs one wav per line.
            #    If the line has multiple <AUDIO> tags, they are usually concatenated in the output wav?
            #    Yes, usually.
            
            # Let's assume standard behavior:
            # One line in JSONL -> One 000000.wav file.
            # We want to split this 000000.wav into smaller pieces.
            # What defines the pieces?
            # The pieces are defined by the `<AUDIO_...>` blocks if they are separate?
            # Or does the user want to split it into arbitrary 40.96s chunks?
            # The user says "according to token quantity".
            
            # If the line looks like: "User: ... Assistant: <AUDIO_1>...<AUDIO_N> [User: ...] Assistant: <AUDIO_M>..."
            # Then we have multiple audio segments.
            # We should parse the line, find all continuous <AUDIO> blocks.
            # For each block, calculate duration, and slice the wav.
            
            # Refined Loop for a single line (single song):
            
            # Find all blocks of audio tokens
            # We need to preserve order.
            # Regex to find <AUDIO> tags.
            
            # Actually, `extract_audio_codes` returns a flat list of ALL tokens in the line.
            # If the audio is generated from them in sequence, we can just take the total length?
            # No, if there are multiple turns, there might be silence or it's one continuous file?
            # Usually `generate.py` concatenates everything for one sample.
            
            # But wait, `prepare_chunks.py` creates a "flat" jsonl where EACH LINE is a chunk.
            # The user wants to match these chunks.
            # So we need to look at `prepare_chunks.py` logic again?
            # No, the user provided a directory path that seems to be the OUTPUT of the MAIN generation (Full).
            # "output_5e-4_1.3_main"
            
            # If we simply want to replicate the "Chunking" logic but apply it to the "Full Audio":
            # We need the definition of the chunks.
            # The definition is in the original JSONL.
            
            # Let's assume the jsonl in the directory is the ORIGINAL one used to generate the full audio.
            # We parse it, identify the segments (e.g. by turns or explicit structure), 
            # calculate their token lengths, and slice the full audio accordingly.
            
            # Implementation Detail:
            # Parse the line into "Assistant Responses" EXACTLY as prepare_chunks.py does.
            # prepare_chunks.py iterates through messages and picks assistant turns that have a preceding user prompt with [dsec:...].
            # HOWEVER, if we are splitting a FULL audio, that full audio was generated from the FULL CONVERSATION.
            # Does the full audio contain silence for user turns? Or just concatenated assistant turns?
            # Standard "generate.py" usually concatenates ONLY the generated audio tokens.
            # So if there are 3 assistant turns, the full audio is Turn1 + Turn2 + Turn3.
            # We must match that structure.
            
            full_tokens_list = [] # List of lists of tokens
            
            if line.startswith("{"):
                 try:
                     data = json.loads(line)
                     messages = data.get("messages", [])
                     
                     # Logic from prepare_chunks.py (simplified for splitting)
                     # In prepare_chunks.py, it skips assistant turns if no prompt or audio.
                     # But here we assume the full audio was generated from ALL assistant tokens found in the file?
                     # Wait, if `generate.py` was used, it generates whatever is in the text.
                     # If the text has <AUDIO_...>, it generates.
                     # So we should collect ALL assistant audio tokens in order.
                     
                     for msg in messages:
                         if msg.get("role") == "assistant":
                             content = msg.get("content", "")
                             
                             # In prepare_chunks.py:
                             # audio_str = extract_audio_codes_str(content) -> keeps <AUDIO_...> tags
                             # decode.py -> extracts ints from that string
                             
                             # Here we extract ints directly to count them
                             turn_tokens = extract_audio_codes(content)
                             
                             # NOTE: prepare_chunks.py has a filter:
                             # if not current_prompt: continue
                             # This means some assistant turns might be SKIPPED in the "Chunks" evaluation
                             # if they don't have a valid dsec prompt.
                             # BUT, the "Full Audio" (000000.wav) likely includes ALL of them if it's a standard generation?
                             # Or was the full audio ALSO generated with similar filtering?
                             # Typically, "generate_main" generates everything.
                             # IF we skip some chunks here but they exist in the audio, our time slicing will be WRONG (misaligned).
                             
                             # CRITICAL: We must account for EVERY token present in the full audio to maintain sync.
                             # Even if `prepare_chunks.py` skipped it for evaluation, we still need to "consume" 
                             # that duration in the audio file to get to the next valid chunk.
                             #
                             # So: we collect ALL chunks.
                             # Later, we can decide which ones to save or how to name them.
                             # But to keep it simple and safe: We split EVERYTHING.
                             # The user can then evaluate the ones they have metadata for.
                             
                             if turn_tokens:
                                 full_tokens_list.append(turn_tokens)
                 except json.JSONDecodeError:
                     # Fallback
                     full_tokens_list.append(extract_audio_codes(line))
            else:
                # Raw text line
                full_tokens_list.append(extract_audio_codes(line))

            # Now we have a list of token-chunks.
            # We slice the audio sequentially.
            
            # Reset cursor for each new file
            current_time_samples = 0
            
            chunk_idx_counter = 0
            
            for t_list in full_tokens_list:
                if not t_list: continue
                
                num_tokens = len(t_list)
                duration_sec = num_tokens / TOKEN_RATE
                duration_samples = int(duration_sec * SAMPLE_RATE)
                
                # Check bounds
                start_sample = current_time_samples
                end_sample = start_sample + duration_samples
                
                if end_sample > waveform.shape[1]:
                    # Pad or just clamp? Clamp usually.
                    end_sample = waveform.shape[1]
                
                # Slice
                chunk_wave = waveform[:, start_sample:end_sample]
                
                # Save
                # Naming: {original_wav_idx}_{chunk_sub_idx}.wav
                # e.g. 000000_0000.wav, 000000_0001.wav, 000001_0000.wav...
                chunk_filename = f"{idx:06d}_{chunk_idx_counter:04d}.wav"
                chunk_path = os.path.join(output_dir, chunk_filename)
                
                torchaudio.save(chunk_path, chunk_wave, SAMPLE_RATE)
                
                # Advance cursor
                current_time_samples = end_sample
                chunk_idx_counter += 1
                
                if current_time_samples >= waveform.shape[1]:
                    break

def main():
    parser = argparse.ArgumentParser(description="Split full audio into chunks based on tokens.")
    parser.add_argument("tasks", nargs="+", help="List of tasks in format GPU_ID:PATH")
    args = parser.parse_args()
    
    for task in args.tasks:
        process_task(task)

if __name__ == "__main__":
    main()

