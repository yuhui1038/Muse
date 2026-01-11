import os
import re
import json
import random
import tarfile
import subprocess
import json_repair
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ===== 宏 =====

BASE_DIR = Path(__file__).parent

# ===== 辅助处理 =====

def pure_name(path:str):
    """获取一个路径文件的原名(不带后缀)"""
    basename = os.path.basename(path)
    dot_pos = basename.rfind('.')
    if dot_pos == -1:
        return basename
    return basename[:dot_pos]

def extract_json(text: str) -> tuple[bool, dict]:
    """从文本中提取并修复JSON数据（增强容错版）
    
    功能：
    1. 自动识别代码块标记(```json```或```)
    2. 修复常见JSON错误（引号不匹配、尾随逗号等）
    3. 支持宽松模式解析
    
    返回：(是否成功, 解析后的字典)
    """
    # 预处理：提取可能的JSON内容区域
    content = text
    
    # 情况1：检查```json```代码块
    if '```json' in text:
        start = text.find('```json')
        end = text.find('```', start + 6)
        content = text[start + 6:end].strip()
    # 情况2：检查普通```代码块
    elif '```' in text:
        start = text.find('```')
        end = text.find('```', start + 3)
        content = text[start + 3:end].strip()
    
    # 清理内容中的常见干扰项
    content = re.sub(r'^[^{[]*', '', content)  # 去除JSON前的非结构内容
    content = re.sub(r'[^}\]]*$', '', content)  # 去除JSON后的非结构内容
    
    # 尝试标准解析
    try:
        json_data = json.loads(content)
        return True, json_data
    except json.JSONDecodeError as e:
        standard_error = e
    
    # 尝试用json_repair修复
    try:
        repaired = json_repair.repair_json(content)
        json_data = json.loads(repaired)
        return True, json_data
    except Exception as e:
        repair_error = e
        return False, {
            "standard_error": standard_error,
            "repair_error": repair_error
        }

def path_join(dir, name):
    return os.path.join(dir, name)

def dict_sort_print(dic:dict, value:bool=True, reverse=True):
    """让一个字典按照值的大小排序后输出"""
    idx = 1 if value else 0
    sorted_lis = sorted(dic.items(), key=lambda x: x[idx], reverse=reverse)
    sorted_dic = {}
    for key, value in sorted_lis:
        sorted_dic[key] = value
    print(json.dumps(sorted_dic, indent=4, ensure_ascii=False))

def clean_newlines(text: str) -> str:
    """
    清理歌词换行：
    1. 标点符号后的换行保留
    2. 非标点符号后的换行 → 空格
    3. 修正英文撇号后的多余空格
    4. 多余空格合并
    5. 保留段落结构，保证标点后换行
    """
    if not text:
        return ""

    text = text.strip()

    # 先把原来的换行统一为 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 把非空行合并为一句（先去掉原来的换行）
    lines = [line.strip() for line in text.split('\n')]
    text = ' '.join(line for line in lines if line)

    # 在句子结尾标点后加换行（中文和英文标点）
    text = re.sub(r'([.,!?:;，。！？；])\s*', r'\1\n', text)

    # 英文撇号后的空格修正
    text = re.sub(r"’\s+", "’", text)

    # 合并多余空格
    text = re.sub(r'[ \t]+', ' ', text)

    # 去掉行首尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()

# ===== 检测工作 =====
def is_ch_char(char:str):
    """判断单个字符是否为中文汉字"""
    if len(char) != 1:
        return False
    
    # 中文汉字的Unicode范围
    # 1. 基本汉字：0x4E00-0x9FFF
    # 2. 扩展A区：0x3400-0x4DBF
    # 3. 扩展B区：0x20000-0x2A6DF
    # 4. 扩展C区：0x2A700-0x2B73F
    # 5. 扩展D区：0x2B740-0x2B81F
    # 6. 扩展E区：0x2B820-0x2CEAF
    
    code = ord(char)
    
    # 常用判断（覆盖绝大部分情况）
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # 扩展A区
    if 0x3400 <= code <= 0x4DBF:
        return True
    # 其他扩展区暂不考虑
    
    return False

# ===== 文件操作 =====

def load_txt(path:str) -> str:
    """以纯文本形式加载一个文件"""
    with open(path, 'r') as file:
        content = file.read()
    return content

def load_json(path:str):
    """加载一个JSON文件"""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def load_jsonl(path:str, limit=-1) -> list[dict]:
    """加载一个JSONL文件"""
    data = []
    with open(path, 'r') as file:
        for id, line in tqdm(enumerate(file), desc=f"Loading {path}"):
            if limit != -1 and id == limit:
                break
            data.append(json.loads(line))
    return data

def save_json(data, path:str):
    """保存一个JSON文件"""
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_jsonl(data:list[dict], path:str, mode='w'):
    """保存一个JSONL文件"""
    with open(path, mode, encoding='utf-8') as file:
        for ele in tqdm(data, desc=f"Saving to {path}"):
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")

def audio_cut(input_path, mode:str, output_dir:str, segment_length:int=30000):
    """
    从音频文件中截取指定长度的片段
    - mode: 切割类型(random / middle)
    - output_dir: 输出文件夹
    - segment_length: 分段长度(毫秒)
    """
    assert mode in ['random', 'middle']

    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    # 加载音频文件
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(44100).set_channels(1) # 设置采样率和声道
    audio_duration = len(audio) # 长度控制
    
    # 如果音频长度小于目标片段长度，使用整个音频
    if audio_duration <= segment_length:
        print(f"Warning: Audio too short ({audio_duration}ms), using full audio: {input_path}")
        segment = audio
    else:
        # 根据模式计算切片位置
        if mode == "random":
            # 随机切
            max_start = max(0, audio_duration - segment_length)
            start = random.randint(0, max_start)
            end = start + segment_length
        else:
            # 切中间
            middle_point = audio_duration // 2
            start = max(0, middle_point - (segment_length // 2))
            end = min(audio_duration, start + segment_length)
            
            # 如果从中间切会导致超出边界，调整起始位置
            if end > audio_duration:
                end = audio_duration
                start = end - segment_length
            elif start < 0:
                start = 0
                end = segment_length
        
        # 确保切片范围有效
        start = max(0, min(start, audio_duration))
        end = max(0, min(end, audio_duration))
        
        if start >= end:
            raise ValueError(f"Invalid slice range: start={start}, end={end}, duration={audio_duration}")
        
        # 执行切片
        segment = audio[start:end]
    
    # 生成输出路径
    basename = pure_name(input_path)
    output_path = os.path.join(output_dir, f"seg_{basename}.wav")
    
    # 保存片段
    segment.export(
        output_path, 
        format="wav",
        codec="pcm_s16le",  # 16位小端编码
        parameters=["-acodec", "pcm_s16le"]  # ffmpeg参数
    )
    return output_path

def format_meta(dir:str, show:bool=True) -> list[dict]:
    """递归获取一个文件夹下所有音频路径(wav / mp3)构建JSONL"""
    if not os.path.isdir(dir):
        return []
    dataset = []
    if show:
        for name in tqdm(os.listdir(dir), desc=f"Formating {dir}"):
            path = os.path.join(dir, name)
            if os.path.isdir(path):
                dataset += format_meta(path, False)
            elif name.endswith('.mp3') or name.endswith('.wav'):
                dataset.append({"path": path})
    else:
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            if os.path.isdir(path):
                dataset += format_meta(path, False)
            elif name.endswith('.mp3') or name.endswith('.wav'):
                dataset.append({"path": path})
    return dataset

def dup_remove(raw_data:list[dict], save_path:str, key:str, seg:str):
    """
    从数据集中去除已经生成的
    - key为raw数据集中的主键，save中的外键
    - seg为目标字段
    """
    if not os.path.exists(save_path):
        print(f"Dup num: 0")
        return raw_data
    save_data = load_jsonl(save_path)
    keys = set()
    for ele in tqdm(save_data, desc="Constructing Dup Set"):
        if seg in ele:
            keys.add(ele[key])
    rest_data = []
    dup_count = 0
    for ele in tqdm(raw_data, desc="Checking Dup"):
        if ele[key] not in keys:
            rest_data.append(ele)
        else:
            dup_count += 1
    print(f"Dup num: {dup_count}")
    return rest_data

def tar_size_check(data_dir:str, subfixes:list[str], per:int, max_size:int):
    """
    压缩前确定一块里面可容纳的文件数量(这里假设文件大小均匀)
    - data_dir: 待压缩文件夹
    - subfixes: 待压缩文件后缀(如.mp3)
    - per: 平均多少文件检查一遍
    - max_size: 最大的限制GB
    """
    names = sorted(list(os.listdir(data_dir)))
    count = 0
    size_sum = 0
    for name in tqdm(names, desc="Size Checking"):
        path = os.path.join(data_dir, name)
        subfix = os.path.splitext(name)[1]
        if subfix not in subfixes:
            continue
        count += 1
        size_sum += os.path.getsize(path)
        if count % per == 0:
            gb_size = size_sum / 1024 / 1024 / 1024
            if gb_size > max_size:
                break
            print(f"Count: {count}, Size: {gb_size:.2f}GB")

def tar_dir(
        data_dir:str,
        subfixes:list[str],
        save_dir:str,
        group_size:int,
        tmp_dir:str,
        mark:str,
        max_workers:int=10,
        ):
    """分块压缩一个目录下的文件(非递归)"""
    names = sorted(list(os.listdir(data_dir)))
    file_num = len(names)
    for i in range(0, file_num, group_size):
        names_subset = names[i:i+group_size]
        size_sum = 0
        name_path = os.path.join(tmp_dir, f"name_{i}_{mark}")
        with open(name_path, 'w', encoding='utf-8') as file:
            for name in tqdm(names_subset, desc=f"Counting Block {i}"):
                path = os.path.join(data_dir, name)
                subfix = os.path.splitext(path)[1]
                if subfix not in subfixes:
                    continue
                file.write("./" + name + "\n")
                size_sum += os.path.getsize(path)
        gb_size = size_sum / 1024 / 1024 / 1024
        print(f"Zipping block {i+1}, size: {gb_size:.2f}GB")
        
        tar_cmd = [
            'tar', 
            '--no-recursion',
            '--files-from', str(name_path),
            '-cf', '-'
        ]
        pigz_cmd = ['pigz', '-p', str(max_workers), '-c']

        tar_process = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, cwd=data_dir)
        pigz_process = subprocess.Popen(pigz_cmd, stdin=tar_process.stdout, stdout=subprocess.PIPE, cwd=data_dir)

        save_path = os.path.join(save_dir, f"block_{i}_{mark}.tar.gz")
        with open(save_path, 'wb') as out_file:
            while True:
                data = pigz_process.stdout.read(4096)
                if not data:
                    break
                out_file.write(data)
        
        tar_process.wait()
        pigz_process.wait()

        if tar_process.returncode == 0 and pigz_process.returncode == 0:
            print(f"压缩完成: {save_path}")
        else:
            print(f"压缩失败: tar返回码={tar_process.returncode}, pigz返回码={pigz_process.returncode}")

def music_avg_size(dir:str):
    """音乐平均大小(MB), 长度(s)"""
    dataset = format_meta(dir)
    dataset = dataset[:50]
    size_sum = 0
    length_sum = 0
    for ele in tqdm(dataset, desc=f"Counting Music Size in {dir}"):
        path = ele['path']
        audio = AudioSegment.from_file(path)
        length_sum += len(audio) / 1000.0
        size_sum += os.path.getsize(path)
    size_avg = size_sum / len(dataset) / 1024 / 1024
    length_avg = length_sum / len(dataset)
    return size_avg, length_avg

def get_sample(path:str, save_path:str="tmp.jsonl", num:int=100):
    """获取一个JSONL文件的100条数据"""
    if not os.path.exists(path):
        return
    if path.endswith(".jsonl"):
        dataset = load_jsonl(path)
    elif path.endswith(".json"):
        dataset = load_json(path)
    else:
        print(f"Unsupport file: {path}")
        return
    sub_dataset = random.sample(dataset, num)
    save_jsonl(sub_dataset, save_path)

def _get_field_one(path:str, field:str):
    """处理一个path的数据"""
    with open(path, 'r') as file:
        data = json.load(file)
    new_data = {
        "id": f"{data['song_id']}_{data['track_index']}",
        field: data[field]
    }
    return new_data

def get_field_suno(dir:str, save_path:str, field:str, max_workers:int=8):
    """从suno的散装json中集中提取出某个字段"""
    paths = []
    for name in tqdm(os.listdir(dir), desc="Getting names"):
        if not name.endswith(".json"):
            continue
        paths.append(os.path.join(dir, name))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_get_field_one, path, field) for path in paths]
        with open(save_path, 'w', encoding='utf-8') as file:
            with tqdm(total=len(paths), desc="Processing the JSONs") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    json.dump(result, file, ensure_ascii=False)
                    file.write("\n")
                    pbar.update(1)

def find_json(dir:str) -> list[str]:
    """找出一个文件夹下的JSONL / JSON文件"""
    names = []
    for name in tqdm(os.listdir(dir), desc="Finding JSON/JSONL"):
        if name.endswith(".json") or name.endswith(".jsonl"):
            names.append(name)
    return names

def show_dir(dir:str):
    """展示一个DIR中的所有内容"""
    if not os.path.isdir(dir):
        return
    for name in os.listdir(dir):
        print(name)

def _convert_mp3(path:str, dir:str):
    """对单个音频做处理"""
    purename = pure_name(path)
    output_path = os.path.join(dir, purename + ".mp3")
    if os.path.exists(output_path):
        # 已经完成
        return "pass"
    try:
        audio = AudioSegment.from_file(path)
    except Exception:
        # 读取文件失败
        print(f"fail to load {path}")
        return "fail"
    audio.export(output_path, format='mp3')
    return "finish"

def convert_mp3(meta_path:str, dir:str, max_workers:int=10):
    """将指定的所有音频转换成mp3并保存在指定目录"""
    os.makedirs(dir, exist_ok=True)
    dataset = load_jsonl(meta_path)
    pass_num = 0
    finish_num = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_convert_mp3, ele['path'], dir) for ele in dataset]
        with tqdm(total=len(dataset), desc=f"Converting {meta_path}") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res == "pass":
                    pass_num += 1
                else:
                    finish_num += 1
                pbar.update(1)
    print(f"Finish {finish_num}, Pass {pass_num}")

# ===== GPU与模型 =====

def get_free_gpu() -> int:
    """返回显存占用最少GPU的id"""
    cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    result = subprocess.check_output(cmd.split()).decode().strip().split("\n")

    free_list = []
    for line in result:
        idx, free_mem = line.split(",")
        free_list.append((int(idx), int(free_mem)))  # (GPU id, free memory MiB)
    
    # 按显存剩余排序
    free_list.sort(key=lambda x: x[1], reverse=True)
    return free_list[0][0]

# ===== 数据分析 =====

def compose_analyze(dataset:list[dict]):
    """音乐结构组成统计分析"""
    # 标签数量
    labels = defaultdict(int)
    for ele in tqdm(dataset):
        segments = ele['segments']
        for segment in segments:
            label = segment['label']
            labels[label] += 1
    print(f"标签数量: {len(labels)}")
    print(dict_sort_print(labels))

    # 不同组合
    label_combs = defaultdict(int)
    for ele in tqdm(dataset):
        segments = ele['segments']
        labels = []
        for segment in segments:
            label = segment['label']
            labels.append(label)
        if len(labels) == 0:
            continue
        label_comb = " | ".join(labels)
        label_combs[label_comb] += 1
    print(f"组合数量: {len(label_combs)}")
    print(dict_sort_print(label_combs))

def _filter_tag(content:str) -> list[str]:
    """对标签字段进行切分与格式整理"""
    tags = []
    raws = re.split(r'[,，.]', content)
    for raw in raws:
        raw = raw.strip().lower()   # 去空格转小写
        if raw == "":
            continue
        seg_pos = raw.find(":")
        if seg_pos != -1:
            # 有冒号只取后面的部分
            tag = raw[seg_pos+1:].strip()
        else:
            tag = raw
        tags.append(tag)
    return tags

def tags_analyze(dataset:list[dict]):
    """歌曲标签分析"""
    tag_count = defaultdict(int)
    for ele in tqdm(dataset, desc="Tag analyzing"):
        tags = _filter_tag(ele['style'])
        for tag in tags:
            tag_count[tag] += 1
    print(f"标签数量: {len(tag_count.keys())}")
    print(dict_sort_print(tag_count))