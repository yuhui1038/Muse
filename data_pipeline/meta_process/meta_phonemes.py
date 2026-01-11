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

# ===== 中文转换 =====

def _split_py(py):
    """将带声调数字的拼音拆成声母sm、韵母ym两部分"""
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

# 所有中文符号
chinese_punctuation_pattern = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09]'

def _has_ch_punc(text):
    match = re.search(chinese_punctuation_pattern, text)
    return match is not None

def _has_en_punc(text):
    return text in string.punctuation

def _trans_cn(text:str, with_sp=True):
    """中文转音素"""
    phonemes = []
    # 分词
    seg_list = jieba.cut(text)
    # 逐词处理
    for seg in seg_list:
        # 字符串有效性
        if seg.strip() == "": continue
        # seg_tn = tn_chinese(seg)
        # 转成拼音(不要声调)
        py =[_py[0] for _py in pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)]
        # 标点检测(有的话跳过)
        if any([_has_ch_punc(_py) for _py in py])  or any([_has_en_punc(_py) for _py in py]):
            continue
        # 拼音拆分
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

# ===== 英文转换 =====

def _read_lexicon(lex_path):
    """读取英文词典"""
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
    """英文(单词)转音素"""
    w_lower = word.lower()
    phonemes = []
    if w_lower in lexicon:
        # 词典有就用词典(不能直接获取引用)
        phonemes += lexicon[w_lower]
    else:
        # 词典没有就用G2P
        phonemes = g2p(w_lower)
        if not phonemes:
            phonemes = []
        # 添加进词典
        lexicon[w_lower] = phonemes
    if len(phonemes) > 0 and with_sp:
        phonemes.append("sp")
    return phonemes

# ===== 单句处理 =====

def _char_lang(c:str) -> int:
    """
    检查一个字符是中文英文还是其它
    0 - 中文
    1 - 英文
    2 - 数字
    3 - 其它
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
    """根据语言切分字符串"""
    lang_segs = []      # 拆分出的字符串集合
    lang_tags = []      # 各字符串段的标签
    lang_seg = ""       # 上一个连续语言字符串
    lang_tag = -1       # 上一个字符的语言类型
    en_count = 0
    for c in text:
        lang = _char_lang(c)
        if lang_tag != lang:
            # 和上一个字符类型不同
            if lang_seg != "":
                lang_segs.append(lang_seg)
                lang_tags.append(lang_tag)
                if lang_tag == 1:
                    en_count += 1
                lang_seg = ""
            if lang == 2 and en_count >= 4:
                # 英文中的数字转换
                lang_segs.append(NUMBER_MAP[c])
                lang_tags.append(1)
            lang_tag = lang
        if lang < 2:
            lang_seg += c
    if lang_seg != "":
        # 最后一段有效的
        lang_segs.append(lang_seg)
        lang_tags.append(lang_tag)
    return lang_segs, lang_tags

def _phoneme_trans(text:str, with_sp=True):
    """将一段歌词转成音素"""
    # 按语言切分
    lang_segs, lang_tags = _lang_seperate(text)
    # 逐段转换
    phonemes = []
    for lang_seg, lang_tag in zip(lang_segs, lang_tags):
        if lang_tag == 0:
            # 中文
            phonemes += _trans_cn(lang_seg, with_sp)
        else:
            # 英文
            phonemes += _trans_en(lang_seg, with_sp)
    return phonemes

# ===== 动态适配 =====

def _get_lyrics(raw_content:str) -> list[str]:
    """从对话中获取歌词内容, 形如'[stage][dsec:xxx][lyrics:xxx\nxxx]'"""
    START_FORMAT = "[lyrics:"
    start = raw_content.find(START_FORMAT)
    if start == -1:
        return None, None
    content = raw_content[start+len(START_FORMAT):-1]
    # 过滤中括号
    content = re.sub(r'\[.*?\]', '', content)   # 前后完整的
    content = re.sub(r'[\[\]]', '', content)    # 不闭合的
    # 句子拆分
    sentences = content.split("\n")
    # 拼接还原
    new_content = raw_content[:start] + START_FORMAT + content + "]"
    return sentences, new_content

def _trans_sentences(sentences:list[str], with_sp:bool=True) -> str:
    """将句子列表装换成包装好的音素字串"""
    phonemes_lis = []
    for sentence in sentences:
        phonemes = _phoneme_trans(sentence, with_sp)
        phonemes_lis.append(" ".join(phonemes))
    # 封装
    phonemes_str = '\n'.join(phonemes_lis)
    envelope = f"[phoneme:{phonemes_str}]"
    envelope = re.sub(r'\d+', '', envelope)     # 去除音调
    return envelope

# ===== 对外接口 =====

def get_phonemes_meta(dataset:list[dict], save_path:str, with_sp:bool=True):
    """为数据集中的歌词加上音素"""
    new_dataset = []
    with open(save_path, 'w', encoding='utf-8') as file:
        for ele in tqdm(dataset, desc="Phoneme trans"):
            ele = copy.deepcopy(ele)
            messages = ele['messages']
            # 第一句不要，只对后面的逐句处理
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
