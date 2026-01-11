import re
import jieba
import string
from pypinyin import pinyin, Style, load_phrases_dict
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from g2p_en import G2p
import os
import json

# Initialize G2P
g2p = G2p()

# Load phrase pinyin dict for better polyphone disambiguation
cc_cedict.load()

# Load custom polyphone corrections if available
poly_correct_path = "eval_pipeline/poly_correct.json"
if os.path.exists(poly_correct_path):
    try:
        with open(poly_correct_path, 'r', encoding='utf-8') as f:
            poly_correct = json.load(f)
            # Apply corrections: poly_correct should be {phrase: [[pinyin], ...]}
            # Evaluate script saves: correct_dic[seg] = pred_phones (list of lists)
            load_phrases_dict(poly_correct)
        print(f"Loaded polyphone corrections from {poly_correct_path}")
    except Exception as e:
        print(f"Warning: Failed to load polyphone corrections: {e}")

re_special_pinyin = re.compile(r'^(n|ng|m)$')

# Number mapping for English numbers
NUMBER_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

chinese_punctuation_pattern = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09]'

def _has_ch_punc(text):
    match = re.search(chinese_punctuation_pattern, text)
    return match is not None

def _has_en_punc(text):
    return text in string.punctuation

def _split_py(py):
    """Split pinyin with tone into initial (sm) and final (ym)."""
    if not py:
        return "", ""
    
    tone = ""
    if py[-1].isdigit():
        tone = py[-1]
        py = py[:-1]
        
    sm = ""
    ym = ""
    suf_r = ""
    
    if re_special_pinyin.match(py):
        py = 'e' + py
    if py.endswith('r'):
        suf_r = 'r'
        py = py[:-1]
        
    if py in ['zi', 'ci', 'si', 'ri']:
        sm = py[:1]
        ym = "ii"
    elif py in ['zhi', 'chi', 'shi']:
        sm = py[:2]
        ym = "iii"
    elif py in ['ya', 'yan', 'yang', 'yao', 'ye', 'yong', 'you']:
        sm = ""
        ym = 'i' + py[1:]
    elif py in ['yi', 'yin', 'ying']:
        sm = ""
        ym = py[1:]
    elif py in ['yu', 'yv', 'yuan', 'yvan', 'yue', 'yve', 'yun', 'yvn']:
        sm = ""
        ym = 'v' + py[2:]
    elif py == 'wu':
        sm = ""
        ym = "u"
    elif py.startswith('w'):
        sm = ""
        ym = "u" + py[1:]
    elif len(py) >= 2 and py[0] in ['j', 'q', 'x'] and py[1] == 'u':
        sm = py[0]
        ym = 'v' + py[2:]
    else:
        seg_pos = re.search('a|e|i|o|u|v', py)
        if seg_pos:
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
        else:
            sm = ym = ""
            return sm, ym
            
    ym += suf_r + tone
    return sm, ym

def _trans_cn(text: str, with_sp=False):
    """Convert Chinese text to phonemes."""
    phonemes = []
    # Use jieba for segmentation
    seg_list = jieba.cut(text)
    
    for seg in seg_list:
        if seg.strip() == "":
            continue
            
        # Convert to pinyin
        py_list = [_py[0] for _py in pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)]
        
        # Check for punctuation
        if any(_has_ch_punc(_py) for _py in py_list) or any(_has_en_punc(_py) for _py in py_list):
            continue
            
        for _py in py_list:
            sm, ym = _split_py(_py)
            if sm != "":
                phonemes.append(sm)
            if ym != "":
                phonemes.append(ym)
                
    if len(phonemes) > 0 and with_sp:
        phonemes.append("sp")
        
    return phonemes

def _trans_en(word: str, with_sp=False):
    """Convert English word to phonemes using g2p_en."""
    # Note: meta_phonemes.py uses a local lexicon cache. 
    # We simplified this to direct g2p call for portability, 
    # but the core logic (g2p_en) is the same.
    
    # We strip punctuation from the word before processing
    word_clean = word.strip(string.punctuation)
    if not word_clean:
        return []
        
    phonemes = g2p(word_clean)
    # g2p might return punctuation as phonemes, we might want to filter them
    phonemes = [p for p in phonemes if p.strip() and p not in string.punctuation]
    
    if len(phonemes) > 0 and with_sp:
        phonemes.append("sp")
    return phonemes

def _char_lang(c: str) -> int:
    """
    0 - Chinese
    1 - English
    2 - Digit
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

def _lang_seperate(text: str):
    """
    Separate text by language.
    Replicating logic from meta_phonemes.py more closely.
    """
    lang_segs = []      # Split string segments
    lang_tags = []      # Language tags
    lang_seg = ""       # Current continuous segment
    lang_tag = -1       # Previous char language tag
    en_count = 0
    
    for c in text:
        lang = _char_lang(c)
        if lang_tag != lang:
            # Language switch
            if lang_seg != "":
                lang_segs.append(lang_seg)
                lang_tags.append(lang_tag)
                if lang_tag == 1:
                    en_count += 1
                lang_seg = ""
            
            # Special handling for numbers in English context (meta_phonemes logic)
            if lang == 2 and en_count >= 4:
                # Number conversion in English context
                if c in NUMBER_MAP:
                    lang_segs.append(NUMBER_MAP[c])
                    lang_tags.append(1)
                # Reset to process next char if needed, but meta_phonemes does this:
                # It appends the number as English word immediately.
                # And effectively consumes this char.
                lang_tag = lang 
                # Note: meta_phonemes doesn't continue here, it proceeds to check if lang < 2
                # But since lang is 2, it won't append to lang_seg in the block below.
            else:
                lang_tag = lang
            
        if lang < 2:
            lang_seg += c
        # Note: meta_phonemes ignores lang >= 2 for appending to lang_seg
        # unless it was handled in the switch block above.
            
    if lang_seg != "":
        lang_segs.append(lang_seg)
        lang_tags.append(lang_tag)
        
    return lang_segs, lang_tags

def _phoneme_trans(text: str, with_sp=False):
    """Convert text to phonemes (replicating meta_phonemes logic)."""
    # Split by language
    lang_segs, lang_tags = _lang_seperate(text)
    
    phonemes = []
    for lang_seg, lang_tag in zip(lang_segs, lang_tags):
        if lang_tag == 0:
            # Chinese
            phonemes.extend(_trans_cn(lang_seg, with_sp))
        else:
            # English (or numbers treated as English words via g2p)
            # Note: meta_phonemes calls _trans_en for lang_tag != 0
            # If lang_tag is 1 (English) or potentially others passed to this block
            # For pure numbers that weren't converted in _lang_seperate, they might be skipped here
            # because _trans_en expects words.
            # But let's follow the structure.
            
            # If it's a list of words (space separated), we should handle it
            words = lang_seg.split()
            for word in words:
                phonemes.extend(_trans_en(word, with_sp))
                
    return phonemes

def get_phonemes(text: str, with_sp=False, remove_tones=True) -> list:
    """
    Main entry point for phoneme conversion.
    """
    phonemes = _phoneme_trans(text, with_sp)
    
    if remove_tones:
        # Remove digits from phonemes
        phonemes = [re.sub(r'\d+', '', p) for p in phonemes]
        
    return phonemes

def calc_per(ref_phonemes, hyp_phonemes):
    """
    Calculate Phoneme Error Rate (PER).
    PER = (S + D + I) / N
    Using Levenshtein distance.
    """
    # Levenshtein.distance calculates insertions + deletions + substitutions
    # This is exactly S + D + I
    
    if not ref_phonemes:
        return 1.0 if hyp_phonemes else 0.0
        
    return levenshtein_distance(ref_phonemes, hyp_phonemes) / len(ref_phonemes)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
