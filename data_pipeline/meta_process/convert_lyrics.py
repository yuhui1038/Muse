import os
import re
import json
import copy
from tqdm import tqdm
from my_tool import dict_sort_print
from collections import defaultdict
from convert_convs import _parse_lyric_with_timestamps

# ===== 歌词解析 =====

def _parse_lyrics(text:str) -> dict:
    """解析歌词信息中的元信息，歌词和时间戳"""
    segs = text.split("\n")
    metadata = {
        "lyrics_meta": {},
        "lyrics": [],
        "lyrics_time": [],
    }
    for seg in segs:
        # 形如[time] metadata / lyrics
        results = _parse_lyric_with_timestamps(seg)
        for time, content in results:
            if ":" in content or "：" in content:
                # 元信息
                pos1 = content.find(":")
                pos2 = content.find("：")
                pos = pos1 if pos1 != -1 else pos2
                key = content[:pos].strip()
                value = content[pos+1:].strip()
                metadata["lyrics_meta"][key] = value
            elif time == "00:00.00":
                # 开头非结构化元信息
                continue
            elif len(metadata['lyrics']) == 0 and "/" in content:
                # 开头非结构化元信息
                continue
            else:
                # 只保留英文和空格标点
                if len(content) == 0:
                    # 中间空隙/结束
                    if len(metadata['lyrics']) != 0 and metadata['lyrics'][-1] != "<nop>":
                        # 如果没有前一段(开头)，或者前一段为空也不记(合并)
                        metadata['lyrics'].append("<nop>")
                        metadata['lyrics_time'].append(time)
                else:
                    if len(metadata['lyrics_time']) != 0 and metadata['lyrics_time'][-1] == time and time != "<nop>":
                        # 时间戳相同说明是在翻译(不记录)
                        continue
                    # 实际歌词
                    metadata['lyrics'].append(content)
                    metadata['lyrics_time'].append(time)
    return metadata

# ===== 语言检测 =====

def _count_ch_nan(text:str):
    """计算一个字符串内中文和其它非英文字符的数量"""
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
    判断歌词的语言类型(en/zh/ez/instrument/nan)
    - val_limit: 不少于这么多句才计入
    - word_limit: 一句不少于这么多字才计入
    """
    ch_lyrics = 0
    en_lyrics = 0
    nan_lyrics = 0
    for lyric in lyrics:
        lyric = copy.deepcopy(lyric)
        if lyric.strip() == "<nop>":
            continue
        lyric = re.sub(r"[‘’￥·′´（），。？“”!@#$%^&*()?.'/,=+_—— ！…《》<>0-9～※~;－・\"、☆｜△【】＃「」‖{}\[\]-]", " ", lyric)
        ch_num, nan_num = _count_ch_nan(lyric)
        
        if nan_num > word_limit:
            nan_lyrics += 1
            continue
        elif ch_num > word_limit:
            ch_lyrics += 1
        
        lyric = re.sub(r'[\u4e00-\u9fff]+', '', lyric)
        # 空格分隔看英文数量
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

# ===== 对外接口 =====

def get_convert_lyrics(dataset:list[dict], save_path:str, dir:str, src_subfix:str=""):
    """对歌词进行转换，并标注歌词语言类型(需定位对应歌曲)"""
    new_dataset = []
    lang_count = defaultdict(int)
    unmatch = []
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in tqdm(dataset, desc="Converting Lyrics"):
            ele = copy.deepcopy(ele)
            # 没有歌词则跳过
            if not ele['has_lyric']:
                # 不添加到最后结果
                continue
            # 获取歌词
            lyric = ele['lyric']
            if lyric == "":
                lyric = ele['tlyric']

            # 歌词解析
            new_data = _parse_lyrics(lyric)

            # 语言解析
            lang = _lang_decide(new_data['lyrics'])
            lang_count[lang] += 1

            # 除去多余字段
            del ele['artists']
            del ele['lyric']
            del ele['tlyric']
            del ele['has_lyric']

            # 添加新字段
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
    """获取与歌词匹配或不匹配的歌曲"""
    # 1. 用歌曲构建查找集
    name_map = {}
    for ele in tqdm(lyric_data, desc="Existing Lyrics"):
        name = ele['name']
        name = re.sub(" ", "", name)
        artist = ele['artist']
        complete_name = f"{name} - {artist}.mp3"
        name_map[complete_name] = ele
    
    # 2. 遍历歌曲找出剩下的
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
