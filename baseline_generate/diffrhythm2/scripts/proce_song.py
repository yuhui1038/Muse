#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
处理 songs.jsonl，生成对应的 lrc 文件和 jsonl 文件。
"""

import json
import os
import re
from pathlib import Path
from typing import List

INPUT_JSONL = Path("xxx/diffrhythm2/example/final_zh_test.jsonl")
OUTPUT_SONG_DIR = Path("xxx/diffrhythm2/example/zh_songs")
OUTPUT_LRC_DIR = Path("xxx/diffrhythm2/example/zh_lrc")

TIMESTAMP_PATTERN = re.compile(r"\[\d{2}:\d{2}(?:\.\d+)?\]")
STRUCTURE_PATTERN = re.compile(r"^\[[^\]]+\]$")


def normalize_structure(tag: str) -> str:
    """将结构标签转换成目标格式。"""
    tag_lower = tag.lower()
    if tag_lower.startswith("verse"):
        return "[verse]"
    if "chorus" in tag_lower:
        return "[chorus]"
    if "bridge" in tag_lower:
        return "[bridge]"
    return f"[{tag_lower}]"


def transform_lyrics(raw_lyrics: str) -> List[str]:
    """根据需求转换歌词为 LRC 行列表。"""
    lines = ["[start]", "[intro]"]
    for raw_line in raw_lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 结构标签单独处理
        if STRUCTURE_PATTERN.match(line) and not TIMESTAMP_PATTERN.match(line):
            tag_content = line[1:-1].strip()
            lines.append(normalize_structure(tag_content))
            continue

        # 去掉时间戳
        text = TIMESTAMP_PATTERN.sub("", line).strip()
        if not text:
            continue
        lines.append(text)

    lines.append("[end]")
    return lines


def ensure_dirs() -> None:
    OUTPUT_SONG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LRC_DIR.mkdir(parents=True, exist_ok=True)


def process_songs() -> None:
    ensure_dirs()
    with INPUT_JSONL.open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            description = data.get("description", "")
            lyrics_raw = data.get("lyrics", "")

            lrc_lines = transform_lyrics(lyrics_raw)
            lrc_filename = f"song_{idx}.lrc"
            lrc_path = OUTPUT_LRC_DIR / lrc_filename
            lrc_path.write_text("\n".join(lrc_lines), encoding="utf-8")

            song_base = f"song_{idx}"
            song_filename = f"{song_base}.jsonl"
            song_json_path = OUTPUT_SONG_DIR / song_filename
            song_entry = {
                "song_name": song_base,
                "style_prompt": description,
                "lyrics": f"example/zh_lrc/{lrc_filename}",
            }
            song_json_path.write_text(json.dumps(song_entry, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Processed song {idx}: {song_filename}")


if __name__ == "__main__":
    process_songs()
