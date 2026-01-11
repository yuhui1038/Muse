import os
import json
from tqdm import tqdm
from funasr import AutoModel
from typing import List, Tuple
from collections import defaultdict
from my_tool import get_free_gpu, dup_remove

# ===== ASR模型(对外) =====

def load_asr_model(bs:int):
    """加载歌词识别模型"""
    device = f"cuda:{get_free_gpu()}"
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        batch_size=bs,
        max_batch_size=bs * 2,
    )
    print(f"Using {device}")
    return model

# ===== ASR解析 =====

def _struct2lang(text: str) -> str:
    """从结构化表示中提取语言标识"""
    start = text.find("|")
    text = text[start+1:]
    end = text.find("|")
    return text[:end]

def _struct2lyrics(text:str) -> str:
    start = text.rfind(">")
    lyric = text[start+1:]
    return lyric

def _struct_parse(text: str) -> Tuple[List[str], List[str]]:
    """将结构化信息逐句切分再做解析"""
    texts = text.split(" <")
    langs, lyrics = [], []
    for ele in texts:
        langs.append(_struct2lang(ele))
        lyrics.append(_struct2lyrics(ele))
    return langs, lyrics

# ===== ASR处理 =====

def _batch_asr(model, paths:List[str]) -> List[Tuple[List[str], List[str]]]:
    """批量语音识别"""
    outputs = model.generate(
        input=paths,
        cache=None,
        language="auto",
        use_itn=True,
        batch_size_s=240,
        merge_vad=True,
        merge_length_s=15,
    )
    return [_struct_parse(output['text']) for output in outputs]

# ===== 整体语言判别 =====

def _lang_decide(lang_lyrics:list[tuple[str, str]], val_limit:int=5, word_limit=5) -> str:
    """
    根据句识别信息做语言判断
    - val_limit: 不少于这么多句才计入
    - word_limit: 一句不少于这么多字才计入
    """
    lang_count = defaultdict(int)
    seg_langs, seg_lyrics = lang_lyrics
    for lang, lyric in zip(seg_langs, seg_lyrics):
        lyric = lyric.strip()
        if lang == "en":
            words_num = len(lyric.split())
        else:
            words_num = len(lyric)
        if words_num >= word_limit:
            lang_count[lang] += 1
    langs = []
    for lang, count in lang_count.items():
        if count >= val_limit:
            langs.append(lang)
    if len(langs) == 0:
        return "pure"
    elif len(langs) == 1:
        return langs[0]
    else:
        return "multi: " + " ".join(langs)

# ===== 对外接口 =====

def get_lang_meta(model, dataset:list[dict], bs:int, save_path:str, save_middle:bool=True) -> list[dict]:
    """
    对一个JSONL数据集做语言识别
    - 最后语言标签保存到lang字段，有zh, en, ja, ko, yue, pure, multi等类型
    - save_middle决定是否将识别的中间结果——各句的语言和歌词保存到save.langs, save.lyrics
    """
    data_num = len(dataset)
    dataset = dup_remove(dataset, save_path, 'path', 'lang')
    new_dataset = []
    with open(save_path, 'a', encoding='utf-8') as file:
        for i in tqdm(range(0, data_num, bs), desc="Lang detecting"):
            batch = []
            paths = []
            for ele in dataset[i:i+bs]:
                path = ele['path']
                if os.path.exists(path):
                    batch.append(ele)
                    paths.append(path)
            lang_lyrics_lis = _batch_asr(model, paths)
            langs = [_lang_decide(lang_lyrics) for lang_lyrics in lang_lyrics_lis]
            for ele, (seg_langs, seg_lyrics), lang in zip(batch, lang_lyrics_lis, langs):
                ele['lang'] = lang
                if save_middle:
                    if 'save' not in ele:
                        ele['save'] = {}
                    ele['save']['langs'] = seg_langs
                    ele['save']['lyrics'] = seg_lyrics
                new_dataset.append(ele)
                json.dump(ele, file, ensure_ascii=False)
                file.write("\n")
    return new_dataset