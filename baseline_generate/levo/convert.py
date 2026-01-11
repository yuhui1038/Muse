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
    lyric = re.sub(pattern, '.', lyric)
    while lyric.find("..") != -1:
        lyric = lyric.replace("..", ".")
    if lyric.endswith('.'):
        lyric = lyric[:-1]
    return lyric

def random_size() -> str:
    # Intro/outro length
    sizes = ['short', 'medium', 'long']
    return random.choice(sizes)

def process_one(messages:list[dict]):
    """Process a conversation messages into input format, return gt_lyric and descriptions"""
    # Overall style
    style:str = messages[0]['content']
    start = style.find(START_STR)
    end = style.find(END_STR)
    descriptions = style[start+len(START_STR):end]

    # Line-by-line lyrics
    start_tag = "intro-" + random_size()
    end_tag = "outro-" + random_size()
    gt_lyric = f"[{start_tag}] ;"
    for message in messages[1:]:
        if message['role'] == "assistant":
            continue
        content = message['content']
        # Segment label
        tag = process_tag(content)
        # Segment lyrics
        lyric = process_lyrics(content)
        if lyric == "" or tag.startswith("[outro"):
            gt_lyric += f" [{end_tag}]"
            break
        gt_lyric += f" {tag} {lyric} ;"
    if not gt_lyric.endswith(f" [{end_tag}]"):
        gt_lyric += f" [{end_tag}]"
    return descriptions, gt_lyric

def main():
    path = "xxx/SongGeneration/data/inputs/test_messages.jsonl"
    dataset = load_jsonl(path)
    save_path = "xxx/SongGeneration/data/inputs/lyrics.jsonl"

    with open(save_path, 'w', encoding='utf-8') as file:
        for id, ele in tqdm(enumerate(dataset), desc="Processing"):
            messages = ele['messages']
            descriptions, gt_lyric = process_one(messages)
            data = {
                "idx": f"test_{id}",
                "descriptions": descriptions,
                "gt_lyric": gt_lyric
            }
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")

if __name__ == "__main__":
    main()