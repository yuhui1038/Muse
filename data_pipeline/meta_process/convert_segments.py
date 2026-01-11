import os
import json
from tqdm import tqdm
from my_tool import path_join, load_json
from concurrent.futures import ProcessPoolExecutor, as_completed

def _check_label(label:str, max_length:int=30) -> bool:
    """检查标签是否合格(非空，非时间戳，非长歌词)"""
    length = len(label.strip())
    if length == 0:
        # print("Error Label: Empty")
        return False
    if length > max_length:
        # print(f"Error Label: Words - {label}")
        return False
    if label.find(":") != -1 and label.find(".") != -1:
        # 认为是时间戳
        # print(f"Error Label: Timestamp - {label}")
        return False
    return True

def _convert_one(path:str):
    """对一条歌曲元数据进行分段，剔除多余的内容"""
    data = load_json(path)
    dir = os.path.dirname(path)
    name = f"{data['song_id']}_{data['track_index']}.mp3"
    path = path_join(dir, name)
    new_data = {
        "path": path,
        "song_id": data['song_id'],
        "segments": []
    }
    words_info = data['timestamped_lyrics']['alignedWords']  # 逐句信息
    seg_info = None

    empty_head = False
    for id, word_info in enumerate(words_info):
        if not word_info['success']:
            continue
        word:str = word_info['word']
        
        label = ""
        if word.startswith('['):
            if seg_info is not None:
                new_data['segments'].append(seg_info)
            label_end = word.find(']')
            label = word[1:label_end]
            if not _check_label(label):
                label = ""

        if label != "":
            seg_info = {
                "start": word_info['startS'],
                "end": 0,
                "label": label,
                "word": word[label_end+2:]
            }
        elif seg_info is not None:
            seg_info['end'] = word_info['endS']
            seg_info['word'] += word
        else:
            empty_head = True
    if seg_info is not None:
        seg_info['end'] = word_info['endS']
        seg_info['word'] += word
    else:
        empty_head = True
    if empty_head:
        # print(f"Empty Head, segment: {len(new_data['segments'])}, path: {path}")
        pass
    return new_data

# ===== 对外接口 =====

def get_convert_segments(data_dir:str, save_path:str, max_workers:int=10):
    paths = []
    for name in tqdm(os.listdir(data_dir), desc="Getting the JSON Paths"):
        if name.endswith(".json"):
            path = path_join(data_dir, name)
            paths.append(path)

    dataset = []
    with open(save_path, 'w', encoding='utf-8') as file:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_convert_one, path) for path in paths]
            with tqdm(total=len(futures), desc="Converting Segments") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    dataset.append(result)
                    json.dump(result, file, ensure_ascii=False)
                    file.write("\n")
                    pbar.update(1)
    return dataset