"""
Convert data to ACE-STEP acceptable format
"""

import re
import json
import random
from tqdm import tqdm

random.seed(42)

def load_jsonl(path:str) -> list[dict]:
    data = []
    with open(path, 'r') as file:
        for line in tqdm(file, desc=f"Loading {path}"):
            data.append(json.loads(line))
    return data

def save_jsonl(data:list, path:str):
    with open(path, 'w', encoding='utf-8') as file:
        for ele in tqdm(data, desc=f"Saving {path}"):
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")

START_STR = "Please generate a song in the following style:"
END_STR = "\nNext, I will tell you the requirements and lyrics"

def process_tag(content:str) -> str:
    """Process segment label"""
    # Extract label
    end = content.find("[desc:")
    tag = content[1:end-1]
    # Lowercase & remove numbers & remove parentheses
    tag = tag.lower()
    tag = re.sub(r'\d+', '', tag)
    tag = re.sub(r'\([^)]*\)', '', tag).strip()
    if tag == "pre-chorus":
        tag = "chorus"
    return f"[{tag}]"

def process_lyrics(content:str) -> str:
    """Process segment lyrics"""
    # Extract lyrics
    start = content.find("[lyrics:\n")
    if start == -1:
        return ""
    end = content.find("][phoneme:")
    lyric = content[start+len("[lyrics:\n"):end]
    
    # Punctuation conversion
    pattern = r'[,。"，:;&—‘\'.\]\[()?\n-]'
    lyric = re.sub(pattern, '\n', lyric)
    while lyric.find("\n\n") != -1:
        lyric = lyric.replace("\n\n", "\n")
    if lyric.endswith('\n'):
        lyric = lyric[:-1]
    return lyric

def has_chinese(text) -> bool:
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Basic Chinese characters
            return True
    return False

def process_duration(lyrics:str):
    if has_chinese(lyrics):
        lyrics = lyrics.replace("\n", "")
        length = len(lyrics)
    else:
        lyrics = lyrics.replace("\n", " ")
        length = len(lyrics.split())
    duration = random.randint(int(length * 0.4), int(length * 0.7))
    return duration

def process_one(messages:list[dict]):
    """Process a conversation messages into input format, return gt_lyric and descriptions"""
    # Overall style
    style:str = messages[0]['content']
    start = style.find(START_STR)
    end = style.find(END_STR)
    descriptions = style[start+len(START_STR):end]

    # Line-by-line lyrics
    all_lyrics = "[intro]\n\n"
    pure_lyrics = ""
    for message in messages[1:]:
        if message['role'] == "assistant":
            continue
        content = message['content']
        # Segment label
        tag = process_tag(content)
        # Segment lyrics
        lyric = process_lyrics(content)
        all_lyrics += f"{tag}\n{lyric}\n\n"
        pure_lyrics += lyric
    all_lyrics = all_lyrics[:-2]

    # Duration
    duration = process_duration(pure_lyrics)

    obj = {
        "prompt": descriptions,
        "lyrics": all_lyrics,
        "audio_duration": duration,
        "infer_step": 60,
        "guidance_scale": 15,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 10,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0,
        "min_guidance_scale": 3,
        "use_erg_tag": True,
        "use_erg_lyric": True,
        "use_erg_diffusion": True,
        "oss_steps": [],
        "actual_seeds": [
            3299954530
        ]
    }
    return obj

def main():
    path = "xxx/ACE-Step/data/inputs/messages.jsonl"
    dataset = load_jsonl(path)

    for id, ele in tqdm(enumerate(dataset), desc="Processing"):
        messages = ele['messages']
        data = process_one(messages)
        path = f"./data/inputs/test_{id}.jsonl"
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

def load_jsonl(path):
    dataset = []
    with open(path, 'r') as file:
        for line in file:
            dataset.append(json.loads(line))
    return dataset

def save_jsonl(dataset, path):
    with open(path, 'w', encoding='utf-8') as file:
        for ele in dataset:
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")

if __name__ == "__main__":
    # main()
    dataset = load_jsonl("./data/outputs/lyrics_params.jsonl")
    for ele in dataset:
        path = ele['audio_path']
        ele['extra'] = int(path[len("./data/outputs/test_"):-len(".wav")])
    sorted_data = sorted(dataset, key=lambda x: x['extra'])
    
    save_path = "./data/outputs/lyrics_params_.jsonl"
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in sorted_data:
            del ele['extra']
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")
