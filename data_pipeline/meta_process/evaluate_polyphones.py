import re
import jieba
from tqdm import tqdm
from my_tool import load_jsonl, save_json, load_json
from pypinyin import pinyin, Style, load_phrases_dict
from pypinyin_dict.phrase_pinyin_data import cc_cedict

cc_cedict.load()
re_special_pinyin = re.compile(r'^(n|ng|m)$')

# Add
reference = load_json("poly_correct.json")
load_phrases_dict(reference)

def _filter(dataset:list[dict]):
    """Filter non-polyphone characters in test set"""
    new_dataset = []
    for ele in tqdm(dataset, desc="Filtering"):
        pos = ele['pos']
        sentence = ele['sentence']
        word = sentence[pos]
        phones = pinyin(word, style=Style.NORMAL, heteronym=True)[0]
        if len(phones) > 1:
            new_dataset.append(ele)
    print(f"Filter non polyphone, {len(dataset)} -> {len(new_dataset)}")
    return new_dataset

def evaluate_polyphones(dataset:list[dict], save_fail:str):
    """Check pinyin processing accuracy for polyphones"""
    dataset = _filter(dataset)
    total = len(dataset)
    right = 0
    correct_dic = {}
    for ele in tqdm(dataset):
        pos = ele['pos']
        phone = ele['phone']
        sentence = ele['sentence']
        seg_list = jieba.cut(sentence)
        length = 0
        for seg in seg_list:
            if length <= pos and length + len(seg) > pos:
                delta = pos - length    # Position in segment
                break
            length += len(seg)
        pred_phones = pinyin(seg, style=Style.NORMAL)
        pred_phone = pred_phones[delta][0]
        if pred_phone == phone or pred_phone.endswith("v"):
            right += 1
        elif len(pred_phones) > 1:
            # Corrected pronunciation (only meaningful for phrases)
            pred_phones[delta] = [phone]
            correct_dic[seg] = pred_phones
    print(f"Acc: {(right / total):.2f}")
    
    origin_dic = load_json(save_fail)
    merge_dic = origin_dic | correct_dic
    save_json(merge_dic, save_fail)
        
if __name__ == "__main__":
    path = "polyphones.jsonl"
    dataset = load_jsonl(path)
    evaluate_polyphones(dataset, "poly_correct.json")