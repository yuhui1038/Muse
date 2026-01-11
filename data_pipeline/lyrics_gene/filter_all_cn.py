import json
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 使用 HuggingFace 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置 HuggingFace 缓存目录，让 SentenceTransformer 能够识别已下载的模型
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")


def clean_lyrics(lyrics):
    """
    清理歌词，去除段落标签、时间戳标签和换行符，只保留纯歌词文本
    
    Args:
        lyrics: 原始歌词文本（包含段落标签如[Verse 1]、时间戳如[00:07.00]和换行符）
        
    Returns:
        清理后的歌词文本（纯文本，无标签和换行符）
    """
    # 使用正则表达式移除所有 [标签] 格式的内容（包括段落标签和时间戳）
    # 匹配 [任意内容] 的模式
    cleaned = re.sub(r'\[.*?\]', '', lyrics)
    
    # 去除所有换行符，替换为空格
    cleaned = cleaned.replace('\n', ' ')
    
    # 去除多余的空格（将多个连续空格替换为单个空格）
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 去除首尾空格
    cleaned = cleaned.strip()
    
    return cleaned


def load_music_data(input_file, max_count=None):
    """
    从jsonl文件加载音乐数据
    
    Args:
        input_file: 输入的jsonl文件路径
        max_count: 最大读取数量，None表示读取全部
        
    Returns:
        音乐数据列表
    """
    music_list = []
    print(f"正在加载音乐数据: {input_file}")
    if max_count:
        print(f"限制读取前 {max_count} 首歌曲")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载数据"):
            try:
                data = json.loads(line.strip())
                # 确保包含必需字段
                if 'description' in data and 'lyrics' in data:
                    music_list.append(data)
                    # 如果达到最大数量，停止读取
                    if max_count and len(music_list) >= max_count:
                        break
            except json.JSONDecodeError:
                continue
    print(f"成功加载 {len(music_list)} 首歌曲")
    return music_list


def deduplicate_music(music_list, texts, model, threshold=0.90, output_file=None, save_interval=10000, matrix_save_dir=None):
    """
    基于文本相似度对音乐数据进行去重
    
    Args:
        music_list: 音乐数据列表
        texts: 用于比较的文本列表
        model: SentenceTransformer模型
        threshold: 相似度阈值
        output_file: 输出文件路径，如果提供则支持增量保存
        save_interval: 每处理多少首有效歌曲保存一次
        matrix_save_dir: 矩阵保存目录
        
    Returns:
        去重后的音乐数据列表
    """
    print(f"正在计算 {len(texts)} 个文本的embeddings...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("正在计算相似度矩阵...")
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    
    # 保存相似度矩阵和embeddings
    if matrix_save_dir:
        os.makedirs(matrix_save_dir, exist_ok=True)
        embeddings_path = os.path.join(matrix_save_dir, 'embeddings.pt')
        cos_scores_path = os.path.join(matrix_save_dir, 'cos_scores.pt')
        print(f"正在保存embeddings到: {embeddings_path}")
        torch.save(embeddings.cpu(), embeddings_path)
        print(f"正在保存相似度矩阵到: {cos_scores_path}")
        torch.save(cos_scores.cpu(), cos_scores_path)
        print("矩阵保存完成！")
    
    print(f"正在去重（阈值: {threshold}）...")
    keep_idx = []
    removed = set()
    
    # 如果提供了输出文件，以追加模式打开
    f = None
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        f = open(output_file, 'w', encoding='utf-8')
    
    saved_count = 0
    
    for i in tqdm(range(len(music_list)), desc="去重进度"):
        if i in removed:
            continue
        keep_idx.append(i)
        
        # 如果启用增量保存，每累积save_interval首就保存一次
        if f and len(keep_idx) - saved_count >= save_interval:
            # 保存从saved_count到当前的所有有效歌曲
            for idx in range(saved_count, len(keep_idx)):
                music = music_list[keep_idx[idx]]
                f.write(json.dumps(music, ensure_ascii=False) + '\n')
            f.flush()  # 确保写入磁盘
            saved_count = len(keep_idx)
            print(f"已保存 {saved_count} 首有效歌曲到文件")
        
        for j in range(i+1, len(music_list)):
            if cos_scores[i][j] > threshold:
                removed.add(j)
    
    # 保存剩余的有效歌曲
    if f:
        for idx in range(saved_count, len(keep_idx)):
            music = music_list[keep_idx[idx]]
            f.write(json.dumps(music, ensure_ascii=False) + '\n')
        f.close()
        print(f"已保存所有 {len(keep_idx)} 首有效歌曲到文件")
    
    deduped_music_list = [music_list[i] for i in keep_idx]
    print(f"去重完成: {len(music_list)} -> {len(deduped_music_list)} (移除了 {len(removed)} 首)")
    
    return deduped_music_list


def dedup_by_description_and_lyrics(input_file, output_file, threshold=0.95, max_count=None, device='cuda:1', save_interval=10000, matrix_save_dir=None):
    """
    方式2：基于description + lyrics进行去重
    
    Args:
        input_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
        threshold: 相似度阈值
        max_count: 最大读取数量，None表示读取全部
        device: 使用的设备，默认cuda:1（GPU1）
        save_interval: 每处理多少首有效歌曲保存一次，默认10000
        matrix_save_dir: 矩阵保存目录，如果提供则保存embeddings和相似度矩阵
    """
    print("\n========== 方式2：基于description + lyrics去重 ==========")
    print(f"使用设备: {device}")
    
    # 加载数据
    music_list = load_music_data(input_file, max_count=max_count)
    
    # 提取description + lyrics的组合文本
    combined_texts = []
    for music in music_list:
        description = music.get('description', '')
        lyrics = music.get('lyrics', '')
        # 清理lyrics，去除结构标签
        cleaned_lyrics = clean_lyrics(lyrics)
        # 将description和清理后的lyrics拼接（用分隔符隔开）
        combined_text = f"{description} [SEP] {cleaned_lyrics}"
        combined_texts.append(combined_text)
    
    # 加载中文模型并指定设备
    # 检查本地是否已有模型，如果有则直接使用本地路径，避免重新下载
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_cache_dir = os.path.join(hf_home, "hub", "models--shibing624--text2vec-bge-large-chinese", "snapshots")
    
    # 查找模型快照目录
    local_model_path = None
    if os.path.exists(model_cache_dir):
        snapshots = [d for d in os.listdir(model_cache_dir) if os.path.isdir(os.path.join(model_cache_dir, d))]
        if snapshots:
            # 使用最新的快照（通常是唯一的）
            local_model_path = os.path.join(model_cache_dir, snapshots[0])
            if os.path.exists(os.path.join(local_model_path, "config.json")):
                print(f"检测到本地模型，使用路径: {local_model_path}")
                model = SentenceTransformer(local_model_path, device=device)
            else:
                local_model_path = None
    
    if local_model_path is None:
        print(f"正在加载模型到 {device}...")
        model = SentenceTransformer('shibing624/text2vec-bge-large-chinese', device=device)
    
    # 去重（支持增量保存）
    deduped_music_list = deduplicate_music(
        music_list, 
        combined_texts, 
        model, 
        threshold, 
        output_file=output_file,
        save_interval=save_interval,
        matrix_save_dir=matrix_save_dir
    )
    
    # 去重函数已经处理了保存，这里只需要打印信息
    if output_file:
        print(f"✓ 保存完成！去重后剩余 {len(deduped_music_list)} 首歌曲\n")
    else:
        # 如果没有提供输出文件，这里保存一次（兼容旧代码）
        print(f"正在保存结果到: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for music in deduped_music_list:
                f.write(json.dumps(music, ensure_ascii=False) + '\n')
        print(f"✓ 保存完成！去重后剩余 {len(deduped_music_list)} 首歌曲\n")
    
    return deduped_music_list


if __name__ == '__main__':
    # 输入文件路径
    input_file = 'lrc_4w_single_pro_des.jsonl'
    
    # 输出文件路径
    output_file = 'filter_all_4w.jsonl'
    
    # 矩阵保存目录
    matrix_save_dir = 'generate_lrc'
    
    # # 设置最大读取数量（用于测试，None表示读取全部）
    max_count = None  # 先测试前5首
        
    #基于description + lyrics去重（暂时注释）
    print("\n基于description + lyrics去重")
    dedup_by_description_and_lyrics(
        input_file, 
        output_file, 
        threshold=0.90, 
        max_count=max_count, 
        device='cuda:7',
        save_interval=10000,  # 每10000首有效歌曲保存一次
        matrix_save_dir=matrix_save_dir  # 保存相似度矩阵
    )
    print(f"\n完成！结果保存在: {output_file}")
    print(f"相似度矩阵保存在: {matrix_save_dir}")
    # 测试清理歌词效果
    # print("\n========== 测试歌词效果 ==========")
    # music_list = load_music_data(input_file, max_count=max_count)
    
    # print("\n" + "="*80)
    # for i, music in enumerate(music_list, 1):
    #     print(f"\n【第 {i} 首歌曲】")
    #     print(f"Description: {music.get('description', '')}")
    #     print("\n--- 原始歌词 ---")
    #     original_lyrics = music.get('lyrics', '')
    #     print(original_lyrics[:500] + "..." if len(original_lyrics) > 500 else original_lyrics)
    #     print("\n--- 清理后的歌词 ---")
    #     cleaned_lyrics = clean_lyrics(original_lyrics)
    #     print(cleaned_lyrics)
    #     print("\n" + "-"*80)

