import os
import re
import json
import copy
import jieba
import string
from tqdm import tqdm
from g2p_en import G2p
from my_tool import BASE_DIR, load_json
from pypinyin import pinyin, Style, load_phrases_dict
from pypinyin_dict.phrase_pinyin_data import cc_cedict

cc_cedict.load()
re_special_pinyin = re.compile(r'^(n|ng|m)$')
reference = load_json("poly_correct.json")
load_phrases_dict(reference)

# ===== Chinese Conversion =====

def _split_py(py):
    """Split pinyin with tone number into initial (sm) and final (ym) parts"""
    tone = py[-1]
    py = py[:-1]
    sm = ""
    ym = ""
    suf_r = ""
    if re_special_pinyin.match(py):
        py = 'e' + py
    if py[-1] == 'r':
        suf_r = 'r'
        py = py[:-1]

    if len(py) == 0:
        # rx
        return "", suf_r + tone

    if py == 'zi' or py == 'ci' or py == 'si' or py == 'ri':
        sm = py[:1]
        ym = "ii"
    elif py == 'zhi' or py == 'chi' or py == 'shi':
        sm = py[:2]
        ym = "iii"
    elif py == 'ya' or py == 'yan' or py == 'yang' or py == 'yao' or py == 'ye' or py == 'yong' or py == 'you':
        sm = ""
        ym = 'i' + py[1:]
    elif py == 'yi' or py == 'yin' or py == 'ying':
        sm = ""
        ym = py[1:]
    elif py == 'yu' or py == 'yv' or py == 'yuan' or py == 'yvan' or py == 'yue ' or py == 'yve' or py == 'yun' or py == 'yvn':
        sm = ""
        ym = 'v' + py[2:]
    elif py == 'wu':
        sm = ""
        ym = "u"
    elif py[0] == 'w':
        sm = ""
        ym = "u" + py[1:]
    elif len(py) >= 2 and (py[0] == 'j' or py[0] == 'q' or py[0] == 'x') and py[1] == 'u':
        sm = py[0]
        ym = 'v' + py[2:]
    else:
        seg_pos = re.search('a|e|i|o|u|v', py)
        try:
            sm = py[:seg_pos.start()]
            ym = py[seg_pos.start():]
            if ym == 'ui':
                ym = 'uei'
            elif ym == 'iu':
                ym = 'iou'
            elif ym == 'un':
                ym = 'uen'
            elif ym == 'ue':
                ym = 've'
        except Exception:
            sm = ym = ""
            return sm, ym
    ym += suf_r + tone
    return sm, ym

# All Chinese punctuation
chinese_punctuation_pattern = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09]'

def _has_ch_punc(text):
    match = re.search(chinese_punctuation_pattern, text)
    return match is not None

def _has_en_punc(text):
    return text in string.punctuation

def _trans_cn(text:str, with_sp=True):
    """Convert Chinese to phonemes"""
    phonemes = []
    # Word segmentation
    seg_list = jieba.cut(text)
    # Process word by word
    for seg in seg_list:
        # String validity
        if seg.strip() == "": continue
        # seg_tn = tn_chinese(seg)
        # Convert to pinyin (without tone)
        py =[_py[0] for _py in pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)]
        # Punctuation detection (skip if present)
        if any([_has_ch_punc(_py) for _py in py])  or any([_has_en_punc(_py) for _py in py]):
            continue
        # Split pinyin
        # phonemes += _split_py(_py)
        for _py in py:
            sm, ym = _split_py(_py)
            if sm != "":
                phonemes.append(sm)
            if ym != "":
                phonemes.append(ym)
        if with_sp:
            phonemes += ["sp"]
    return phonemes

# ===== English Conversion =====

def _read_lexicon(lex_path):
    """Read English lexicon"""
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

LEX_PATH = BASE_DIR / f"data/ref/lexion.txt"
lexicon = _read_lexicon(LEX_PATH)

g2p = G2p()

def _trans_en(word:str, with_sp=True):
    """Convert English (word) to phonemes"""
    w_lower = word.lower()
    phonemes = []
    if w_lower in lexicon:
        # Use lexicon if available (cannot directly get reference)
        phonemes += lexicon[w_lower]
    else:
        # Use G2P if not in lexicon
        phonemes = g2p(w_lower)
        if not phonemes:
            phonemes = []
        # Add to lexicon
        lexicon[w_lower] = phonemes
    if len(phonemes) > 0 and with_sp:
        phonemes.append("sp")
    return phonemes

# ===== Single Sentence Processing =====

def _char_lang(c:str) -> int:
    """
    Check if a character is Chinese, English, or other
    0 - Chinese
    1 - English
    2 - Number
    3 - Other
    """
    if '\u4e00' <= c <= '\u9fff':
        return 0
    elif ('a' <= c <= 'z') or ('A' <= c <= 'Z'):
        return 1
    elif c.isdigit():
        return 2
    else:
        return 3

NUMBER_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

def _lang_seperate(text:str) -> list[str]:
    """Split string by language"""
    lang_segs = []      # Set of split strings
    lang_tags = []      # Tags for each string segment
    lang_seg = ""       # Previous continuous language string
    lang_tag = -1       # Language type of previous character
    en_count = 0
    for c in text:
        lang = _char_lang(c)
        if lang_tag != lang:
            # Different from previous character type
            if lang_seg != "":
                lang_segs.append(lang_seg)
                lang_tags.append(lang_tag)
                if lang_tag == 1:
                    en_count += 1
                lang_seg = ""
            if lang == 2 and en_count >= 4:
                # Number conversion in English
                lang_segs.append(NUMBER_MAP[c])
                lang_tags.append(1)
            lang_tag = lang
        if lang < 2:
            lang_seg += c
    if lang_seg != "":
        # Last valid segment
        lang_segs.append(lang_seg)
        lang_tags.append(lang_tag)
    return lang_segs, lang_tags

def _phoneme_trans(text:str, with_sp=True):
    """Convert a lyric segment to phonemes"""
    # Split by language
    lang_segs, lang_tags = _lang_seperate(text)
    # Convert segment by segment
    phonemes = []
    for lang_seg, lang_tag in zip(lang_segs, lang_tags):
        if lang_tag == 0:
            # Chinese
            phonemes += _trans_cn(lang_seg, with_sp)
        else:
            # English
            phonemes += _trans_en(lang_seg, with_sp)
    return phonemes

# ===== Dynamic Adaptation =====

def _get_lyrics(raw_content:str) -> list[str]:
    """Extract lyric content from dialogue, format like '[stage][dsec:xxx][lyrics:xxx\nxxx]'"""
    START_FORMAT = "[lyrics:"
    start = raw_content.find(START_FORMAT)
    if start == -1:
        return None, None
    content = raw_content[start+len(START_FORMAT):-1]
    # Filter brackets
    content = re.sub(r'\[.*?\]', '', content)   # Complete brackets
    content = re.sub(r'[\[\]]', '', content)    # Unclosed brackets
    # Split sentences
    sentences = content.split("\n")
    # Reconstruct
    new_content = raw_content[:start] + START_FORMAT + content + "]"
    return sentences, new_content

def _trans_sentences(sentences:list[str], with_sp:bool=True) -> str:
    """Convert sentence list to wrapped phoneme string"""
    phonemes_lis = []
    for sentence in sentences:
        phonemes = _phoneme_trans(sentence, with_sp)
        phonemes_lis.append(" ".join(phonemes))
    # Wrap
    phonemes_str = '\n'.join(phonemes_lis)
    envelope = f"[phoneme:{phonemes_str}]"
    envelope = re.sub(r'\d+', '', envelope)     # Remove tones
    return envelope

# ===== External Interface =====

def get_phonemes_meta(dataset:list[dict], save_path:str, with_sp:bool=True):
    """Add phonemes to lyrics in dataset"""
    new_dataset = []
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in tqdm(dataset, desc="Phoneme trans"):
            ele = copy.deepcopy(ele)
            messages = ele['messages']
            # Skip first message, process subsequent ones sentence by sentence
            for message in messages[1:]:
                if message['role'] == "assistant":
                    continue
                content = message['content']
                sentences, new_content = _get_lyrics(content)
                if sentences is None:
                    continue
                phonemes = _trans_sentences(sentences, with_sp)
                message['content'] = new_content + phonemes
            new_dataset.append(ele)
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")
    return new_dataset
