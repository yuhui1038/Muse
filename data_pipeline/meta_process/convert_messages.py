"""
基于分段描述生成多轮对话数据（逐段生成）。

输入：
- MuseData/data/meta_suno_cn.jsonl      # 基础 meta（包含 src_path、lyrics、lang、tag 等）
- 3_block_80000_cn_desc.jsonl           # 分段描述，含各 section 的 startS/endS/text/desc
- mucodec pt 目录：同 multi_data_suno.py 的 PT_DIR_CN

输出：
- MuseData/sft_dataset_suno_cn.jsonl    # 逐段生成的多轮对话

对话格式示例（参考 temp1.jsonl）：
- 第一条 user：汇总提示（中文），说明“逐段”+ 提供 [Intro dsec...] 描述
- 第一条 assistant：前奏 token（0 ~ 第一个 section 的 startS）
- 后续每段：
  user.content = "[{Section} dsec]{desc}\\n{text}"
  assistant.content = 对应时间片段 token
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from my_tool import load_jsonl, clean_newlines

# 语言配置
LANG = "en"
# 路径配置
META_FILE = f"meta_suno_{LANG}.jsonl"
# desc 文件夹，读取其中所有 *.jsonl
DESC_DIR = "desc"
PT_DIR = f"suno_mucodec_{LANG}"
# 输出目录（与 meta 不同目录），每个 desc 文件生成一组三个文件（中文）
OUTPUT_DIR = "outputs"
OUTPUT_BASENAME = "minus_phonemes"

TOKEN_PER_SECOND = 25


LOG_FILE = os.path.join(OUTPUT_DIR, "section_mismatch_cn.log")   # 放在输出目录里

def _log_warning(msg: str):
    """打印并落盘"""
    print(msg, end='\n')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# 时间戳解析正则
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
    从带时间戳的歌词中解析出 [(start_time_s, text), ...]，按时间升序。
    返回的时间戳来自 meta 文件中的 lyrics 字段。
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
    放宽规则：只要 [] 里包含英文单词（≥2 字母），整个括号内容原样返回。
    """
    # 第一个 [] 内只要出现英文单词就匹配
    m = re.search(r'\[([A-Za-z][A-Za-z0-9\s\-\(\)]*)\]', text)
    if m:
        return m.group(1).strip()   # 去掉首尾空格
    return None

def format_section_label(sec_name: str) -> str:
    """保持原有空格，只做首尾空白裁剪。"""
    return sec_name.strip()


def normalize_section_name(sec_name: str) -> str:
    """
    标准化 section 名称用于匹配：
    - 去掉所有空格
    - 转小写
    - 去掉末尾数字（如果有）
    """
    # 去掉所有空格
    normalized = sec_name.replace(" ", "").lower()
    # 去掉末尾的数字（如 "verse1" -> "verse", "chorus1" -> "chorus"）
    normalized = re.sub(r"\d+$", "", normalized)
    return normalized


def clean_desc(desc: str) -> str:
    """
    清理 desc 字段：
    1. 如果开头有 [desc]，删去
    2. 如果首尾都是中括号，把中括号删去
    """
    if not desc:
        return desc
    
    desc = desc.strip()
    
    # 如果开头有 [desc]，删去
    if desc.startswith("[desc]"):
        desc = desc[6:].strip()
    
    # 如果首尾都是中括号，把中括号删去
    if desc.startswith("[") and desc.endswith("]"):
        desc = desc[1:-1].strip()
    
    return desc


def build_desc_map(desc_path_or_dir: str) -> Dict[Tuple[str, int], List[dict]]:
    """
    支持传入单个 jsonl 文件或包含多个 jsonl 的目录。
    目录场景下，会按文件名排序后逐个读取，后读到的同 key 会覆盖前面的记录。
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

                # Change: 将整个放入
                # sections = obj.get("sections", [])
                # mapping[(song_id, track_idx)] = sections
                mapping[(song_id, track_idx)] = obj
    return mapping


def extract_suffix_from_desc_path(desc_path: str, fallback_idx: int) -> str:
    """
    根据 desc 文件名提取后缀，用于输出文件命名。
    规则：尝试从文件名中截取 "3_block_<suffix>_(cn|en)_desc.jsonl" 的 <suffix>。
    如果无法匹配，使用 fallback_idx（从 0 开始）转为字符串。
    """
    fname = os.path.basename(desc_path)
    m = re.search(r"3_block_([^_]+)_(?:cn|en)_desc\.jsonl", fname, re.IGNORECASE)
    if m:
        return m.group(1)
    return str(fallback_idx)


def extract_suffix_num(desc_path: str, fallback_idx: int) -> int:
    """
    提取可排序的数字后缀，用于按数字顺序处理 desc 文件。
    无法解析时使用 fallback_idx。
    """
    suffix = extract_suffix_from_desc_path(desc_path, fallback_idx)
    try:
        return int(suffix)
    except ValueError:
        return fallback_idx


def build_messages(item: dict, obj: dict, audio: torch.Tensor) -> Optional[dict]:
    """
    使用 meta 中的时间戳来切分音频，desc_sections 提供 desc 和 text。
    """
    if not obj:
        return None
    desc_sections = obj.get("sections", [])

    # 从 meta 的 lyrics 中解析时间戳
    lyrics_raw = item.get("lyrics", "") or ""
    meta_ts_list = parse_lyric_with_timestamps(lyrics_raw)
    if not meta_ts_list:
        return None

    total_seconds = audio.shape[0] / float(TOKEN_PER_SECOND)
    
    # 排序 desc_sections（按 startS，用于匹配）
    desc_sections = sorted(desc_sections, key=lambda x: x.get("startS", 0.0))
    
    # 从 desc_sections 中获取中间的 sections（跳过第一个 Intro 和最后一个 Outro）
    # Intro 和 Outro 不在 meta 的 lyrics 中，需要单独处理
    # 中间的 sections 按顺序匹配
    middle_desc_sections = desc_sections[1:-1] if len(desc_sections) > 2 else desc_sections[1:] if len(desc_sections) > 1 else []
    
    # 从 meta 时间戳中识别 section 并建立映射
    # 一个 section 可能包含多行歌词，需要合并
    section_timestamps: List[Tuple[str, float, float, str, str]] = []  # (section_name, start_s, end_s, text, desc)
    current_section: Optional[Tuple[str, float, str]] = None  # (section_name, start_s, accumulated_text)
    desc_idx = 0  # 用于按顺序匹配 desc（从 middle_desc_sections 中匹配）
    
    for idx, (start_s, text) in enumerate(meta_ts_list):
        # 提取 section 名称
        section_name = extract_section_from_text(text)
        
        if section_name:
            # 遇到新的 section 标签
            # 先保存上一个 section（如果有）
            if current_section:
                # 确定上一个 section 的结束时间（当前时间戳）
                prev_sec_name, prev_start_s, prev_text = current_section
                # 只删除时间戳，保留其他所有内容（包括换行符、section标签等）
                clean_prev_text = re.sub(r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]", "", prev_text)
                
                # 按顺序获取 desc（从中间的 sections 中按顺序匹配）
                prev_desc = ""
                if desc_idx < len(middle_desc_sections):
                    prev_desc = clean_desc(middle_desc_sections[desc_idx].get("desc", ""))
                    desc_idx += 1
                
                section_timestamps.append((prev_sec_name, prev_start_s, start_s, clean_prev_text, prev_desc))
            
            # 开始新的 section
            current_section = (section_name, start_s, text)
        else:
            # 没有 section 标签，属于当前 section 的后续行
            if current_section:
                sec_name, sec_start, sec_text = current_section
                # 保留换行符，用换行符连接
                current_section = (sec_name, sec_start, sec_text + "\n" + text)
            # 如果没有 current_section，跳过（可能是 Intro 前的空行）
    
    # 处理最后一个 section
    # 检查最后一个时间戳是否是空文本（表示结尾标记）
    outro_start_s: Optional[float] = None
    if meta_ts_list and not meta_ts_list[-1][1].strip():
        # 最后一个时间戳是空文本，表示结尾标记
        outro_start_s = meta_ts_list[-1][0]
    
    if current_section:
        sec_name, sec_start, sec_text = current_section
        # 如果最后一个时间戳是空文本，最后一个 section 的结束时间应该是这个时间戳
        # 否则使用总时长
        if outro_start_s is not None:
            end_s = outro_start_s
        else:
            end_s = total_seconds
        
        # 只删除时间戳，保留其他所有内容
        clean_text = re.sub(r"\[([0-9]{1,2}):([0-9]{1,2})(?:[.:]([0-9]{1,3}))?\]", "", sec_text)
        
        # 按顺序获取 desc（从中间的 sections 中按顺序匹配）
        desc = ""
        if desc_idx < len(middle_desc_sections):
            desc = clean_desc(middle_desc_sections[desc_idx].get("desc", ""))
            desc_idx += 1
        
        section_timestamps.append((sec_name, sec_start, end_s, clean_text, desc))
    
    # 检查个数是否匹配（只检查中间的 sections，不包括 Intro 和 Outro）
    if desc_idx != len(middle_desc_sections):
        _log_warning(f"⚠️ 警告：section 个数不匹配！meta 中有 {len(section_timestamps)} 个 section，desc 中有 {len(middle_desc_sections)} 个中间 section（不包括 Intro 和 Outro）(song_id: {item.get('song_id')}, track_index: {item.get('track_index')})")
    
    if not section_timestamps:
        return None

    # Intro 段：从 0 到第一个 section 的开始时间
    # Intro 的 desc 应该已经在顺序匹配中获取了，但 Intro 本身不在 meta 的 lyrics 中
    # 所以需要从 desc_sections 的第一个 section（通常是 Intro）获取
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
        # 中文歌直接用omni
        tag = omni_tag
    else:
        # 英文歌比较omni / style
        style_sim = obj.get("style_sim", 0)
        omni_sim = obj.get("omni_sim", 0)
        try:
            tag = omni_tag if omni_sim > style_sim else style_tag
        except Exception as e:
            # sim分数有误默认omni
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

    # 逐段处理（使用 meta 中的时间戳）
    for idx, (sec_name, start_s, end_s, text, desc) in enumerate(section_timestamps):
        # user content: [Section dsec : desc][Section lyrics : ...]（保留原有空格）
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
    
    # 如果最后一个时间戳是空文本，添加 Outro 段
    # Outro 的 desc 应该从 desc_sections 的最后一个 section 获取（通常是 Outro）
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
    使用单个 desc 文件生成输出文件（messages-only / meta-only）。
    不生成主文件。
    """
    desc_map = build_desc_map(desc_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # messages-only 去掉后缀命名，仅保留 block 序号（如 ..._8000.jsonl）
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
        
        for item in tqdm(dataset, desc=f"处理 meta_suno_{LANG}.jsonl (desc: {suffix})"):
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

            # 写 messages-only（去除 _messages_only 后缀）
            messages_only = {"messages": sample.get("messages", [])}
            fout_msg.write(json.dumps(messages_only, ensure_ascii=False) + "\n")
            # 写 meta-only
            meta_only = {k: v for k, v in sample.items() if k != "messages"}
            fout_meta.write(json.dumps(meta_only, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ messages-only: {out_msg}")
    print(f"✅ meta-only: {out_meta}")
    print(f"总计 {total}，保留 {kept}，跳过 {skipped}")


def convert_train_valid():
    # 收集 desc 文件
    if not os.path.isdir(DESC_DIR):
        print(f"⚠️ DESC_DIR 不存在: {DESC_DIR}")
        return

    unsorted_files = [
        os.path.join(DESC_DIR, name)
        for name in os.listdir(DESC_DIR)
        if name.endswith(".jsonl")
    ]

    # 按提取的数字后缀排序，保证 8000 排在 16000 前
    desc_files = sorted(
        unsorted_files,
        key=lambda p: extract_suffix_num(p, 0),
    )
    if not desc_files:
        print(f"⚠️ 未找到 desc 文件: {DESC_DIR}")
        return

    # Change
    # for idx, desc_path in enumerate(desc_files):
    #     suffix = extract_suffix_from_desc_path(desc_path, idx)
    #     process_with_desc(desc_path, suffix)

    for desc_path in desc_files:
        name = os.path.splitext(os.path.basename(desc_path))[0]
        if name.endswith(LANG):
            process_with_desc(desc_path, name)

# assistant需要，只是内容为空
# 然后里面每个section有3个desc，你取最大值对应的
# omni也有两个，取最大的那个（style就不用了）

from meta_phonemes import _get_lyrics, _trans_sentences

def _form_section(section:dict, en:bool) -> str:
    """处理一个section内部，确定选择对应desc"""
    # 段落标签
    section_tag = f"[{section['section']}]"
    # 段落描述
    descs = [section['desc1'], section['desc2'], section['desc3']]
    sims = [section['desc1_sim'], section['desc2_sim'], section['desc3_sim']]
    max_sim = max(sims)
    max_index = sims.index(max_sim)
    desc:str = descs[max_index]
    
    if desc == "音频过短":
        desc = "[desc:]"
    else:
        DESC_START = "[desc] "
        if desc.startswith(DESC_START):
            desc = desc[len(DESC_START):] + "]"
        desc = "[desc:" + desc[1:]
    
    # 歌词 & 音素
    text:str = section['text']
    if text.find(']') != -1:
        # 去除前面的段落标签
        start = text.rfind(']')
        text = text[start+1:]
    if len(text.strip()) == 0:
        # 开头段没有歌词/音素
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
    """处理得到一个多轮对话开头"""
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
            # 段落处理
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
            # 初始添加
            first_content = messages[0]['content']
            intro = _form_intro(ele)
            messages[0]['content'] = intro + first_content

            data = {"messages": messages}
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")

if __name__ == "__main__":
    convert_test()