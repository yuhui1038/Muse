import os
import re
import json
import copy
from tqdm import tqdm
from my_tool import dict_sort_print
from collections import defaultdict
from convert_convs import _parse_lyric_with_timestamps

# ===== Lyric Parsing =====

def _parse_lyrics(text:str) -> dict:
    """Parse metadata, lyrics and timestamps from lyric information"""
    segs = text.split("\n")
    metadata = {
        "lyrics_meta": {},
        "lyrics": [],
        "lyrics_time": [],
    }
    for seg in segs:
        # Format: [time] metadata / lyrics
        results = _parse_lyric_with_timestamps(seg)
        for time, content in results:
            if ":" in content or "：" in content:
                # Metadata
                pos1 = content.find(":")
                pos2 = content.find("：")
                pos = pos1 if pos1 != -1 else pos2
                key = content[:pos].strip()
                value = content[pos+1:].strip()
                metadata["lyrics_meta"][key] = value
            elif time == "00:00.00":
                # Unstructured metadata at the beginning
                continue
            elif len(metadata['lyrics']) == 0 and "/" in content:
                # Unstructured metadata at the beginning
                continue
            else:
                # Only keep English and space punctuation
                if len(content) == 0:
                    # Middle gap/end
                    if len(metadata['lyrics']) != 0 and metadata['lyrics'][-1] != "<nop>":
                        # If there's no previous segment (beginning), or previous segment is empty, don't record (merge)
                        metadata['lyrics'].append("<nop>")
                        metadata['lyrics_time'].append(time)
                else:
                    if len(metadata['lyrics_time']) != 0 and metadata['lyrics_time'][-1] == time and time != "<nop>":
                        # Same timestamp means it's a translation (don't record)
                        continue
                    # Actual lyrics
                    metadata['lyrics'].append(content)
                    metadata['lyrics_time'].append(time)
    return metadata

# ===== Language Detection =====

def _count_ch_nan(text:str):
    """Count the number of Chinese and other non-English characters in a string"""
    ch_num = 0
    nan_num = 0
    nan = ""
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            ch_num += 1
        elif ('a' <= c <= 'z') or ('A' <= c <= 'Z') or len(c.strip()) == 0:
            continue
        else:
            nan_num += 1
            nan += c
    # if len(nan) > 0:
    #     print(nan)
    return ch_num, nan_num

def _lang_decide(lyrics:list[str], val_limit:int=5, word_limit=3) -> str:
    """
    Determine the language type of lyrics (en/zh/ez/instrument/nan)
    - val_limit: Only count if there are at least this many sentences
    - word_limit: Only count if a sentence has at least this many words
    """
    ch_lyrics = 0
    en_lyrics = 0
    nan_lyrics = 0
    for lyric in lyrics:
        lyric = copy.deepcopy(lyric)
        if lyric.strip() == "<nop>":
            continue
        lyric = re.sub(r"[''￥·′´（），。？""!@#$%^&*()?.'/,=+_—— ！…《》<>0-9～※~;－・\"、☆｜△【】＃「」‖{}\[\]-]", " ", lyric)
        ch_num, nan_num = _count_ch_nan(lyric)
        
        if nan_num > word_limit:
            nan_lyrics += 1
            continue
        elif ch_num > word_limit:
            ch_lyrics += 1
        
        lyric = re.sub(r'[\u4e00-\u9fff]+', '', lyric)
        # Count English words by space separation
        en_num = len(lyric.split(" "))
        if en_num > word_limit:
            en_lyrics += 1

    if nan_lyrics > val_limit:
        return "nan"
    if ch_lyrics > val_limit and en_lyrics > val_limit:
        return "ez"
    if ch_lyrics > val_limit:
        return "zh"
    if en_lyrics > val_limit:
        return "en"
    return "instrument"

# ===== External Interface =====

def get_convert_lyrics(dataset:list[dict], save_path:str, dir:str, src_subfix:str=""):
    """Convert lyrics and annotate language type (need to locate corresponding song)"""
    new_dataset = []
    lang_count = defaultdict(int)
    unmatch = []
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in tqdm(dataset, desc="Converting Lyrics"):
            ele = copy.deepcopy(ele)
            # Skip if no lyrics
            if not ele['has_lyric']:
                # Don't add to final result
                continue
            # Get lyrics
            lyric = ele['lyric']
            if lyric == "":
                lyric = ele['tlyric']

            # Parse lyrics
            new_data = _parse_lyrics(lyric)

            # Language detection
            lang = _lang_decide(new_data['lyrics'])
            lang_count[lang] += 1

            # Remove redundant fields
            del ele['artists']
            del ele['lyric']
            del ele['tlyric']
            del ele['has_lyric']

            # Add new fields
            ele['lyric_lang'] = lang
            ele['source'] += src_subfix
            for key, value in new_data.items():
                ele[key] = value
            
            new_dataset.append(ele)
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")

    dict_sort_print(lang_count)
    return new_dataset, unmatch

def get_match_music(music_data:list[dict], lyric_data:list[dict]):
    """Get songs that match or don't match with lyrics"""
    # 1. Build lookup set from songs
    name_map = {}
    for ele in tqdm(lyric_data, desc="Existing Lyrics"):
        name = ele['name']
        name = re.sub(" ", "", name)
        artist = ele['artist']
        complete_name = f"{name} - {artist}.mp3"
        name_map[complete_name] = ele
    
    # 2. Iterate through songs to find remaining ones
    matches = []
    unmatches = []
    for ele in tqdm(music_data, desc="Check Matching"):
        path = ele['path']
        name = os.path.basename(path)
        if name not in name_map:
            unmatches.append(ele)
        else:
            meta = name_map[name]
            meta['path'] = path
            matches.append(meta)
    return matches, unmatches
