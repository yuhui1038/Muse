import os
import json
from tqdm import tqdm
from funasr import AutoModel
from typing import List, Tuple
from collections import defaultdict
from my_tool import get_free_gpu, dup_remove

# ===== ASR Model (External) =====

def load_asr_model(bs:int):
    """Load lyric recognition model"""
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

# ===== ASR Parsing =====

def _struct2lang(text: str) -> str:
    """Extract language identifier from structured representation"""
    start = text.find("|")
    text = text[start+1:]
    end = text.find("|")
    return text[:end]

def _struct2lyrics(text:str) -> str:
    start = text.rfind(">")
    lyric = text[start+1:]
    return lyric

def _struct_parse(text: str) -> Tuple[List[str], List[str]]:
    """Split structured information sentence by sentence and then parse"""
    texts = text.split(" <")
    langs, lyrics = [], []
    for ele in texts:
        langs.append(_struct2lang(ele))
        lyrics.append(_struct2lyrics(ele))
    return langs, lyrics

# ===== ASR Processing =====

def _batch_asr(model, paths:List[str]) -> List[Tuple[List[str], List[str]]]:
    """Batch speech recognition"""
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

# ===== Overall Language Detection =====

def _lang_decide(lang_lyrics:list[tuple[str, str]], val_limit:int=5, word_limit=5) -> str:
    """
    Determine language based on sentence recognition information
    - val_limit: Only count if there are at least this many sentences
    - word_limit: Only count if a sentence has at least this many words
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

# ===== External Interface =====

def get_lang_meta(model, dataset:list[dict], bs:int, save_path:str, save_middle:bool=True) -> list[dict]:
    """
    Perform language recognition on a JSONL dataset
    - Final language tag is saved to lang field, types include zh, en, ja, ko, yue, pure, multi, etc.
    - save_middle determines whether to save intermediate recognition results (sentence languages and lyrics) to save.langs, save.lyrics
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