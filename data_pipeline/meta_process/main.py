from my_tool import (
    load_json,
    load_jsonl, 
    load_txt, 
    save_jsonl, 
    format_meta, 
    pure_name, 
    BASE_DIR,
    compose_analyze,
    get_sample,
    get_field_suno,
    tags_analyze,
    find_json,
    show_dir,
    convert_mp3,
    tar_dir,
    tar_size_check,
    clean_newlines,
    dict_sort_print,
)
from meta_lang import load_asr_model, get_lang_meta
from meta_tags import load_tag_model, get_tags_meta
from meta_endpoints import get_endpoints_meta
from meta_phonemes import get_phonemes_meta
from filter import filter_lang, filter_length
from convert_convs import get_convert_convs
from convert_segments import get_convert_segments
from convert_lyrics import get_convert_lyrics, get_match_music

def pipeline():
    import os
    dir = "suno_batch"
    name = pure_name(dir)
    save_dir = BASE_DIR / f"data/{name}"

    # Initialize paths (only once)
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, "raw.jsonl")
    if os.path.exists(raw_path):
        dataset = load_jsonl(raw_path)
    else:
        dataset = format_meta(dir)
    save_jsonl(dataset, raw_path)

    # Length filtering
    dataset = dataset[:1000]
    max_workers = 10
    dataset = filter_length(dataset, 120, 360, max_workers)

    # Language tagging
    # lang_bs = 8
    # model = load_asr_model(lang_bs)
    # lang_path = os.path.join(save_dir, "meta_lang.jsonl")
    # dataset = get_lang_meta(model, dataset, lang_bs, lang_path)

    # Language filtering
    # dataset = filter_lang(dataset, ['zh', 'en'])

    # Style tagging
    tag_bs = 4
    tag_path = os.path.join(save_dir, "meta_tags.jsonl")
    model, processor = load_tag_model()
    prompt_path = BASE_DIR / "prompts/new_tags.md"
    prompt = load_txt(prompt_path)
    get_tags_meta(model, processor, dataset, prompt, tag_bs, tag_path)

def repeat(func):
    while True:
        try:
            func()
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    repeat(pipeline)