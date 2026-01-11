"""
Generate multi-turn dialogue data based on segment descriptions (segment-by-segment generation).

Input:
- MuseData/data/meta_suno_cn.jsonl      # Base meta (contains src_path, lyrics, lang, tag, etc.)
- 3_block_80000_cn_desc.jsonl           # Segment descriptions, contains startS/endS/text/desc for each section
- mucodec pt directory: same as PT_DIR_CN in multi_data_suno.py

Output:
- MuseData/sft_dataset_suno_cn.jsonl    # Multi-turn dialogues generated segment by segment

Dialogue format example (refer to temp1.jsonl):
- First user: Summary prompt (Chinese), explains "segment-by-segment" + provides [Intro dsec...] description
- First assistant: Intro tokens (0 ~ first section's startS)
- Subsequent segments:
  user.content = "[{Section} dsec]{desc}\\n{text}"
  assistant.content = corresponding time segment tokens
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from my_tool import load_jsonl, clean_newlines

# Language configuration
LANG = "en"
# Path configuration
META_FILE = f"meta_suno_{LANG}.jsonl"
# desc folder, read all *.jsonl files in it
DESC_DIR = "desc"
PT_DIR = f"suno_mucodec_{LANG}"
# Output directory (different from meta directory), each desc file generates a set of three files
OUTPUT_DIR = "outputs"
OUTPUT_BASENAME = "minus_phonemes"

TOKEN_PER_SECOND = 25


LOG_FILE = os.path.join(OUTPUT_DIR, "section_mismatch_cn.log")   # Place in output directory

def _log_warning(msg: str):
    """Print and save to file"""
    print(msg, end='\n')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# Timestamp parsing regex
timestamp_pattern = re.compile(
    r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]"
)


def load_pt(pt_file: str) -> torch.Tensor:
    return torch.load(pt_file, map_location="cpu").squeeze().long()


def get_token_slice(audio: torch.Tensor, start_s: float, end_s: float) -> str:
    if start_s < 0:
        start_s = 0
    if end_s < 0:
        end_s = 0
    s_idx = int(start_s * TOKEN_PER_SECOND)
    e_idx = int(end_s * TOKEN_PER_SECOND)
    s_idx = max(0, min(s_idx, audio.shape[0]))
    e_idx = max(0, min(e_idx, audio.shape[0]))
    if e_idx <= s_idx:
        sliced = []
    else:
        sliced = audio[s_idx:e_idx]
    return "[SOA]" + "".join(f"<AUDIO_{int(i)}>" for i in sliced) + "[EOA]"


def infer_pt_path(src_path: str) -> Optional[str]:
    if not src_path:
        return None
    stem = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(PT_DIR, f"{stem}.pt")


def parse_lyric_with_timestamps(lyric: str) -> List[Tuple[float, str]]:
    """
    Parse [(start_time_s, text), ...] from lyrics with timestamps, sorted by time ascending.
    The returned timestamps come from the lyrics field in the meta file.
    """
    result: List[Tuple[float, str]] = []
    matches = list(timestamp_pattern.finditer(lyric))
    
    for i, match in enumerate(matches):
        start_idx = match.end()
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(lyric)
        
        text = lyric[start_idx:end_idx].strip()
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        ms_str = match.group(3) if match.group(3) else "0"
        
        if len(ms_str) == 2:
            fractional_seconds = int(ms_str) / 100.0
        elif len(ms_str) == 3:
            fractional_seconds = int(ms_str) / 1000.0
        else:
            fractional_seconds = int(ms_str) / 1000.0 if ms_str else 0.0
        
        total_seconds = minutes * 60 + seconds + fractional_seconds
        result.append((total_seconds, text))
    return result


def extract_section_from_text(text: str) -> Optional[str]:
    """
    Relaxed rule: As long as [] contains English words (≥2 letters), return the entire bracket content as-is.
    """
    # Match if English words appear in the first []
    m = re.search(r'\[([A-Za-z][A-Za-z0-9\s\-\(\)]*)\]', text)
    if m:
        return m.group(1).strip()   # Remove leading and trailing spaces
    return None

def format_section_label(sec_name: str) -> str:
    """Keep original spaces, only trim leading and trailing whitespace."""
    return sec_name.strip()


def normalize_section_name(sec_name: str) -> str:
    """
    Normalize section name for matching:
    - Remove all spaces
    - Convert to lowercase
    - Remove trailing digits (if any)
    """
    # Remove all spaces
    normalized = sec_name.replace(" ", "").lower()
    # Remove trailing digits (e.g., "verse1" -> "verse", "chorus1" -> "chorus")
    normalized = re.sub(r"\d+$", "", normalized)
    return normalized


def clean_desc(desc: str) -> str:
    """
    Clean desc field:
    1. If starts with [desc], remove it
    2. If both ends are brackets, remove brackets
    """
    if not desc:
        return desc
    
    desc = desc.strip()
    
    # If starts with [desc], remove it
    if desc.startswith("[desc]"):
        desc = desc[6:].strip()
    
    # If both ends are brackets, remove brackets
    if desc.startswith("[") and desc.endswith("]"):
        desc = desc[1:-1].strip()
    
    return desc


def build_desc_map(desc_path_or_dir: str) -> Dict[Tuple[str, int], List[dict]]:
    """
    Support passing a single jsonl file or a directory containing multiple jsonl files.
    In directory scenario, files are sorted by name and read sequentially, later records with the same key will overwrite earlier ones.
    """
    mapping: Dict[Tuple[str, int], List[dict]] = {}

    paths: List[str] = []
    if os.path.isdir(desc_path_or_dir):
        for name in sorted(os.listdir(desc_path_or_dir)):
            if name.endswith(".jsonl"):
                paths.append(os.path.join(desc_path_or_dir, name))
    else:
        paths.append(desc_path_or_dir)

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    with open("error.txt", 'a', encoding='utf-8') as error_file:
                        error_file.write(line + "\n")
                    continue

                song_id = obj.get("song_id")
                track_idx = obj.get("track_index", 0)

                # Change: Put the entire object
                # sections = obj.get("sections", [])
                # mapping[(song_id, track_idx)] = sections
                mapping[(song_id, track_idx)] = obj
    return mapping


def extract_suffix_from_desc_path(desc_path: str, fallback_idx: int) -> str:
    """
    Extract suffix from desc filename for output file naming.
    Rule: Try to extract <suffix> from filename pattern "3_block_<suffix>_(cn|en)_desc.jsonl".
    If no match, use fallback_idx (starting from 0) converted to string.
    """
    fname = os.path.basename(desc_path)
    m = re.search(r"3_block_([^_]+)_(?:cn|en)_desc\.jsonl", fname, re.IGNORECASE)
    if m:
        return m.group(1)
    return str(fallback_idx)


def extract_suffix_num(desc_path: str, fallback_idx: int) -> int:
    """
    Extract sortable numeric suffix for processing desc files in numeric order.
    Use fallback_idx if parsing fails.
    """
    suffix = extract_suffix_from_desc_path(desc_path, fallback_idx)
    try:
        return int(suffix)
    except ValueError:
        return fallback_idx


def build_messages(item: dict, obj: dict, audio: torch.Tensor) -> Optional[dict]:
    """
    Use timestamps from meta to split audio, desc_sections provide desc and text.
    """
    if not obj:
        return None
    desc_sections = obj.get("sections", [])

    # Parse timestamps from meta's lyrics
    lyrics_raw = item.get("lyrics", "") or ""
    meta_ts_list = parse_lyric_with_timestamps(lyrics_raw)
    if not meta_ts_list:
        return None

    total_seconds = audio.shape[0] / float(TOKEN_PER_SECOND)
    
    # Sort desc_sections (by startS, for matching)
    desc_sections = sorted(desc_sections, key=lambda x: x.get("startS", 0.0))
    
    # Get middle sections from desc_sections (skip first Intro and last Outro)
    # Intro and Outro are not in meta's lyrics, need separate handling
    # Middle sections are matched in order
    middle_desc_sections = desc_sections[1:-1] if len(desc_sections) > 2 else desc_sections[1:] if len(desc_sections) > 1 else []
    
    # Identify sections from meta timestamps and build mapping
    # One section may contain multiple lyric lines, need to merge
    section_timestamps: List[Tuple[str, float, float, str, str]] = []  # (section_name, start_s, end_s, text, desc)
    current_section: Optional[Tuple[str, float, str]] = None  # (section_name, start_s, accumulated_text)
    desc_idx = 0  # For matching desc in order (from middle_desc_sections)
    
    for idx, (start_s, text) in enumerate(meta_ts_list):
        # Extract section name
        section_name = extract_section_from_text(text)
        
        if section_name:
            # Encountered new section label
            # First save previous section (if any)
            if current_section:
                # Determine end time of previous section (current timestamp)
                prev_sec_name, prev_start_s, prev_text = current_section
                # Only remove timestamp, keep all other content (including line breaks, section labels, etc.)
                clean_prev_text = re.sub(r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]", "", prev_text)
                
                # Get desc in order (match from middle sections in order)
                prev_desc = ""
                if desc_idx < len(middle_desc_sections):
                    prev_desc = clean_desc(middle_desc_sections[desc_idx].get("desc", ""))
                    desc_idx += 1
                
                section_timestamps.append((prev_sec_name, prev_start_s, start_s, clean_prev_text, prev_desc))
            
            # Start new section
            current_section = (section_name, start_s, text)
        else:
            # No section label, belongs to subsequent lines of current section
            if current_section:
                sec_name, sec_start, sec_text = current_section
                # Preserve line breaks, connect with line breaks
                current_section = (sec_name, sec_start, sec_text + "\n" + text)
            # If no current_section, skip (might be empty line before Intro)
    
    # Process last section
    # Check if last timestamp is empty text (indicates end marker)
    outro_start_s: Optional[float] = None
    if meta_ts_list and not meta_ts_list[-1][1].strip():
        # Last timestamp is empty text, indicates end marker
        outro_start_s = meta_ts_list[-1][0]
    
    if current_section:
        sec_name, sec_start, sec_text = current_section
        # If last timestamp is empty text, last section's end time should be this timestamp
        # Otherwise use total duration
        if outro_start_s is not None:
            end_s = outro_start_s
        else:
            end_s = total_seconds
        
        # Only remove timestamp, keep all other content
        clean_text = re.sub(r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]", "", sec_text)
        
        # Get desc in order (match from middle sections in order)
        desc = ""
        if desc_idx < len(middle_desc_sections):
            desc = clean_desc(middle_desc_sections[desc_idx].get("desc", ""))
            desc_idx += 1
        
        section_timestamps.append((sec_name, sec_start, end_s, clean_text, desc))
    
    # Check if counts match (only check middle sections, excluding Intro and Outro)
    if desc_idx != len(middle_desc_sections):
        _log_warning(f"⚠️ Warning: Section count mismatch! meta has {len(section_timestamps)} sections, desc has {len(middle_desc_sections)} middle sections (excluding Intro and Outro) (song_id: {item.get('song_id')}, track_index: {item.get('track_index')})")
    
    if not section_timestamps:
        return None

    # Intro segment: from 0 to first section's start time
    # Intro's desc should have been obtained in sequential matching, but Intro itself is not in meta's lyrics
    # So need to get from desc_sections' first section (usually Intro)
    first_section_start = section_timestamps[0][1] if section_timestamps else total_seconds
    intro_desc = ""
    if desc_sections and desc_sections[0].get("section", "").lower() == "intro":
        intro_desc = clean_desc(desc_sections[0].get("desc", ""))

    # Change: Use desc tag
    # tag = item.get("tag", "")
    song_id:str = obj.get("song_id", "")
    omni_tag = obj.get("omni", "")
    style_tag = obj.get("style", "")

    if song_id.find("cn") != -1:
        # Chinese songs use omni directly
        tag = omni_tag
    else:
        # English songs compare omni / style
        style_sim = obj.get("style_sim", 0)
        omni_sim = obj.get("omni_sim", 0)
        try:
            tag = omni_tag if omni_sim > style_sim else style_tag
        except Exception as e:
            # If sim score is invalid, default to omni
            tag = omni_tag
            print(f"Error: {song_id}, {e}")

    # Change: English
    intro_prompt = (
        f"Please generate a song in the following style:{tag}.\n"
        "Next, I will tell you the requirements and lyrics for the song fragment to be generated, section by section.\n"
        f"[Intro][desc:{intro_desc}]"
    )

    messages: List[dict] = []
    messages.append({"role": "user", "content": intro_prompt})
    messages.append(
        {
            "role": "assistant",
            "content": get_token_slice(audio, 0.0, first_section_start),
        }
    )

    # Process segment by segment (using timestamps from meta)
    for idx, (sec_name, start_s, end_s, text, desc) in enumerate(section_timestamps):
        # user content: [Section dsec : desc][Section lyrics : ...] (preserve original spaces)
        label = format_section_label(sec_name)
        content = f"[{label}]"
        content += f"[desc:{desc}]"
        lyrics_text = re.sub(r'^\[.*?\]\s*\n?', '', text.strip())
        if lyrics_text:
            content += f"[lyrics:\n{lyrics_text}]"
        messages.append({"role": "user", "content": content})
        messages.append(
            {
                "role": "assistant",
                "content": get_token_slice(audio, start_s, end_s),
            }
        )
    
    # If last timestamp is empty text, add Outro segment
    # Outro's desc should be obtained from desc_sections' last section (usually Outro)
    if outro_start_s is not None and outro_start_s < total_seconds:
        outro_desc = ""
        if desc_sections and desc_sections[-1].get("section", "").lower() == "outro":
            outro_desc = clean_desc(desc_sections[-1].get("desc", ""))
        
        messages.append({"role": "user", "content": f"[Outro][desc:{outro_desc}]"})
        messages.append(
            {
                "role": "assistant",
                "content": get_token_slice(audio, outro_start_s, total_seconds),
            }
        )

    sample = {
        "song_id": item.get("song_id"),
        "track_index": item.get("track_index"),
        "src_path": item.get("src_path"),
        "tag": item.get("tag"),
        "lang": item.get("lang"),
        "duration": item.get("duration"),
        "messages": messages,
    }
    return sample


def process_with_desc(desc_path: str, suffix: str) -> None:
    """
    Generate output files using a single desc file (messages-only / meta-only).
    Does not generate main file.
    """
    desc_map = build_desc_map(desc_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # messages-only naming: remove suffix, only keep block number (e.g., ..._8000.jsonl)
    out_msg = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_{suffix}.jsonl")
    out_meta = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_{suffix}_meta_only.jsonl")

    dataset = load_jsonl(META_FILE)
    total = len(dataset)
    kept = 0
    skipped = 0

    for path in [out_msg, out_meta]:
        if os.path.exists(path):
            # Change: use past
            already_msg = load_jsonl(out_msg)
            already_meta = load_jsonl(out_meta)
            assert len(already_meta) == len(already_msg)
            dataset = dataset[len(already_msg):]
    
    with open(out_msg, "a", encoding="utf-8") as fout_msg, \
         open(out_meta, "a", encoding="utf-8") as fout_meta:
        
        for item in tqdm(dataset, desc=f"Processing meta_suno_{LANG}.jsonl (desc: {suffix})"):
            key = (item.get("song_id"), item.get("track_index", 0))

            obj = desc_map.get(key)

            if not obj:
                skipped += 1
                continue

            pt_path = infer_pt_path(item.get("src_path", ""))
            if not pt_path or not os.path.exists(pt_path):
                skipped += 1
                continue

            audio = load_pt(pt_path)
            sample = build_messages(item, obj, audio)
            if not sample:
                skipped += 1
                continue

            # Write messages-only (remove _messages_only suffix)
            messages_only = {"messages": sample.get("messages", [])}
            fout_msg.write(json.dumps(messages_only, ensure_ascii=False) + "\n")
            # Write meta-only
            meta_only = {k: v for k, v in sample.items() if k != "messages"}
            fout_meta.write(json.dumps(meta_only, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ messages-only: {out_msg}")
    print(f"✅ meta-only: {out_meta}")
    print(f"Total {total}, kept {kept}, skipped {skipped}")


def convert_train_valid():
    # Collect desc files
    if not os.path.isdir(DESC_DIR):
        print(f"⚠️ DESC_DIR does not exist: {DESC_DIR}")
        return

    unsorted_files = [
        os.path.join(DESC_DIR, name)
        for name in os.listdir(DESC_DIR)
        if name.endswith(".jsonl")
    ]

    # Sort by extracted numeric suffix, ensure 8000 comes before 16000
    desc_files = sorted(
        unsorted_files,
        key=lambda p: extract_suffix_num(p, 0),
    )
    if not desc_files:
        print(f"⚠️ No desc files found: {DESC_DIR}")
        return

    # Change
    # for idx, desc_path in enumerate(desc_files):
    #     suffix = extract_suffix_from_desc_path(desc_path, idx)
    #     process_with_desc(desc_path, suffix)

    for desc_path in desc_files:
        name = os.path.splitext(os.path.basename(desc_path))[0]
        if name.endswith(LANG):
            process_with_desc(desc_path, name)

# assistant needs, but content is empty
# Then each section inside has 3 descs, you take the one corresponding to the maximum value
# omni also has two, take the larger one (style is not used)

from meta_phonemes import _get_lyrics, _trans_sentences

def _form_section(section:dict, en:bool) -> str:
    """Process inside a section, determine which desc to select"""
    # Segment label
    section_tag = f"[{section['section']}]"
    # Segment description
    descs = [section['desc1'], section['desc2'], section['desc3']]
    sims = [section['desc1_sim'], section['desc2_sim'], section['desc3_sim']]
    max_sim = max(sims)
    max_index = sims.index(max_sim)
    desc:str = descs[max_index]
    
    if desc == "音频过短":  # "Audio too short"
        desc = "[desc:]"
    else:
        DESC_START = "[desc] "
        if desc.startswith(DESC_START):
            desc = desc[len(DESC_START):] + "]"
        desc = "[desc:" + desc[1:]
    
    # Lyrics & phonemes
    text:str = section['text']
    if text.find(']') != -1:
        # Remove preceding segment label
        start = text.rfind(']')
        text = text[start+1:]
    if len(text.strip()) == 0:
        # Opening segment has no lyrics/phonemes
        lyrics = ""
        phonemes = ""
    else:
        if en:
            lyrics = "[lyrics:\n" + clean_newlines(text) + "]"
        else:
            lyrics = "[lyrics:" + text + "]"

        sentences, lyrics = _get_lyrics(lyrics)
        phonemes = _trans_sentences(sentences)
    return section_tag + desc + lyrics + phonemes

def _form_intro(ele:dict) -> str:
    """Process to get multi-turn dialogue opening"""
    omni1 = ele['omni1']
    omni2 = ele['omni2']
    omni_sim1 = ele['omni1_sim']
    omni_sim2 = ele['omni2_sim']
    tag = omni1 if omni_sim1 > omni_sim2 else omni2
    return (
        f"Please generate a song in the following style:{tag}.\n"
        "Next, I will tell you the requirements and lyrics for the song fragment to be generated, section by section.\n"
    )

def convert_test():
    path = "filter.jsonl"
    dataset = load_jsonl(path)
    save_path = "messages.jsonl"
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in tqdm(dataset, desc=f"Converting {path}"):
            messages = []
            # Segment processing
            sections = ele['sections']
            id:str = ele['song_id']
            english = id.startswith("suno_test_en")
            for section in sections:
                content = _form_section(section, english)
                messages += [
                    {
                        "role": "user",
                        "content": content
                    },
                    {
                        "role": "assistant",
                        "content": ""
                    }
                ]
            # Initial addition
            first_content = messages[0]['content']
            intro = _form_intro(ele)
            messages[0]['content'] = intro + first_content

            data = {"messages": messages}
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")

if __name__ == "__main__":
    convert_test()