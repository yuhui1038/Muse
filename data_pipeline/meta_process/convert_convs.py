"""
生成多轮对话数据，每轮包含歌词文本和对应的音频 token 切片
"""

import json
import re
import os
import torch
from tqdm import tqdm
from my_tool import load_jsonl

TOKEN_PER_SECOND = 25  # 每秒音频对应的 token 数
NUM_ITEMS = 100000  # 先处理 10 条数据

timestamp_pattern = re.compile(r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]")

def _parse_lyric_with_timestamps(lyric: str):
    """
    返回 [(start_time_s, text), ...] 按时间戳顺序
    """
    result = []
    for match in timestamp_pattern.finditer(lyric):
        start_idx = match.end()
        end_idx = lyric.find("[", start_idx)
        text = lyric[start_idx:end_idx].strip() if end_idx != -1 else lyric[start_idx:].strip()
        if not text:
            continue
        minute = int(match.group(1))
        second = int(match.group(2))
        ms = int(match.group(3)) if match.group(3) else 0
        total_seconds = minute * 60 + second + ms / 1000
        result.append((total_seconds, text))
    return result

def _load_audio_tokens(pt_file):
    """
    加载音频的Mucodec编码
    """
    audio_ids = torch.load(pt_file, map_location="cpu").squeeze().long()
    return audio_ids

def _get_token_slice(audio_tokens, start_s, end_s):
    """按时间段对编码进行切分"""
    start_idx = int(start_s * TOKEN_PER_SECOND)
    end_idx = int(end_s * TOKEN_PER_SECOND)
    sliced = audio_tokens[start_idx:end_idx]
    return "[SOA]" + "".join([f"<AUDIO_{i.item()}>" for i in sliced]) + "[EOA]"

def _process_item(item, pt_dir:str):
    song_name = item.get("song") or item.get("name")
    song_name = song_name.split('.mp3')[0]              # 针对mucodec，去掉后缀
    pt_file = os.path.join(pt_dir, f"{song_name}.pt")
    if not os.path.exists(pt_file):
        return None

    audio_tokens = _load_audio_tokens(pt_file)
    tlyric_ = item.get('tlyric', "")
    lyric_ = item.get('lyric', "")
    lyric = tlyric_ if len(tlyric_) > len(lyric_) else lyric_
    lyrics_ts = _parse_lyric_with_timestamps(lyric)

    if not lyrics_ts:
        # 如果没有歌词，跳过
        return None

    rounds = []

    # 先生成一条包含歌曲信息的系统消息
    intro_text = (
        f"请生成一首歌曲，歌名为《{item.get('name', '')}》，风格是{item.get('style','')}"
        f"，情绪为{item.get('emotion','')}，节奏：{item.get('rhythm','')}，"
        f"{item.get('description','')}，由{item.get('singer','')}演唱，语言：{item.get('lang','')}。"
        f"歌词如下：" + " ".join([text for _, text in lyrics_ts]) + "接下来我会逐句告诉你需要生成歌曲片段的歌词，\n请先生成前奏"
    )
    rounds.append({"role": "user", "content": intro_text})
    rounds.append({"role": "assistant", "content": _get_token_slice(audio_tokens, 0, lyrics_ts[0][0])})  # 前奏 token

    # 每句歌词对应一轮  
    for idx, (start_s, text) in enumerate(lyrics_ts[:-1]):  ## 最后一行单独处理
        end_s = lyrics_ts[idx + 1][0] if idx + 1 < len(lyrics_ts) else len(audio_tokens)/TOKEN_PER_SECOND  # 最后一行到音频结尾
        rounds.append({"role": "user", "content": text})
        rounds.append({"role": "assistant", "content": _get_token_slice(audio_tokens, start_s, end_s)})

    # 尾部处理逻辑
    rounds.append({"role": "user", "content": f"请生成歌词{lyrics_ts[-1][1]}以及歌曲结尾"})
    rounds.append({"role": "assistant", "content": _get_token_slice(audio_tokens, lyrics_ts[-1][0], len(audio_tokens)/TOKEN_PER_SECOND)})

    return rounds

# ===== 对外接口 =====

def get_convert_convs(dataset:list[dict], pt_dir:str, save_path:str):
    with open(save_path, "w", encoding="utf-8") as fout:
        for item in tqdm(dataset, desc="Converting convs"):
            rounds = _process_item(item, pt_dir)
            if not rounds:
                continue
            fout.write(json.dumps({"messages": rounds}, ensure_ascii=False) + "\n")