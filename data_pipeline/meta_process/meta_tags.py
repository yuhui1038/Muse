import os
import json
import torch
from tqdm import tqdm
from transformers import (
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeForConditionalGeneration
)
from qwen_omni_utils import process_mm_info
from my_tool import get_free_gpu, audio_cut, extract_json, dup_remove, BASE_DIR

# ===== 标签模型和处理器(对外) =====

def load_tag_model():
    """加载标签模型"""
    device = f"cuda:{get_free_gpu()}"
    print(f"Using {device}")
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        # local_files_only=True
    ).to(device)
    model.disable_talker()
    model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name,
        # local_files_only=True
    )
    return model, processor

# ===== Tag标注 =====

def _format_messages(prompt:str, path:str) -> list[dict]:
    """构造传入omni的messages"""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": path},
            ]
        }
    ]
    return messages

def _batch_tagging(model, processor, paths:list[str], prompt:str, mode="random"):
    """对一批歌曲进行标注"""
    convs = []
    middle_paths = []
    output_dir = BASE_DIR / "data/temp"
    for path in paths:
        seg_path = audio_cut(path, mode, output_dir)
        middle_paths.append(seg_path)
        messages = _format_messages(prompt, seg_path)
        convs.append(messages)

    USE_AUDIO_IN_VIDEO = False

    with torch.no_grad():
        text = processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(convs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(
            text=text,
            audio=audios,
            padding=True, 
            images=images,
            videos=videos,
            return_tensors="pt",
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        gene_texts = processor.batch_decode(
            text_ids[0].sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    torch.cuda.empty_cache()
    # 删除音频切段
    for path in middle_paths:
        if os.path.exists(path):
            os.remove(path)
    return gene_texts

# ===== 对外接口 =====

def get_tags_meta(model, processor, dataset:list[dict], prompt:str, bs:int, save_path:str):
    data_num = len(dataset)
    dataset = dup_remove(dataset, save_path, 'path', 'tags')
    new_dataset = []
    with open(save_path, 'a', encoding="utf-8") as file:
        for i in tqdm(range(0, data_num, bs)):
            batch = []
            paths = []
            for ele in dataset[i:i+bs]:
                path = ele['path']
                if os.path.exists(path):
                    batch.append(ele)
                    paths.append(path)
            contents = _batch_tagging(model, processor, paths, prompt)
            for ele, content in zip(batch, contents):
                ckeck, json_data = extract_json(content)
                if not ckeck:
                    continue
                ele['tags'] = json_data['tags']
                new_dataset.append(ele)
                json.dump(ele, file, ensure_ascii=False)
                file.write('\n')
    return new_dataset