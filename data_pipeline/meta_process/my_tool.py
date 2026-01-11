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

# ===== Macros =====

BASE_DIR = Path(__file__).parent

# ===== Helper Functions =====

def pure_name(path:str):
    """Get the original name of a file path (without extension)"""
    basename = os.path.basename(path)
    dot_pos = basename.rfind('.')
    if dot_pos == -1:
        return basename
    return basename[:dot_pos]

def extract_json(text: str) -> tuple[bool, dict]:
    """Extract and repair JSON data from text (enhanced error-tolerant version)
    
    Features:
    1. Automatically identify code block markers (```json``` or ```)
    2. Fix common JSON errors (mismatched quotes, trailing commas, etc.)
    3. Support lenient parsing mode
    
    Returns: (success, parsed dictionary)
    """
    # Preprocessing: extract possible JSON content area
    content = text
    
    # Case 1: Check ```json``` code block
    if '```json' in text:
        start = text.find('```json')
        end = text.find('```', start + 6)
        content = text[start + 6:end].strip()
    # Case 2: Check regular ``` code block
    elif '```' in text:
        start = text.find('```')
        end = text.find('```', start + 3)
        content = text[start + 3:end].strip()
    
    # Clean common interference items in content
    content = re.sub(r'^[^{[]*', '', content)  # Remove unstructured content before JSON
    content = re.sub(r'[^}\]]*$', '', content)  # Remove unstructured content after JSON
    
    # Try standard parsing
    try:
        json_data = json.loads(content)
        return True, json_data
    except json.JSONDecodeError as e:
        standard_error = e
    
    # Try to repair with json_repair
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
    """Sort a dictionary by value size and output"""
    idx = 1 if value else 0
    sorted_lis = sorted(dic.items(), key=lambda x: x[idx], reverse=reverse)
    sorted_dic = {}
    for key, value in sorted_lis:
        sorted_dic[key] = value
    print(json.dumps(sorted_dic, indent=4, ensure_ascii=False))

def clean_newlines(text: str) -> str:
    """
    Clean lyric line breaks:
    1. Keep line breaks after punctuation
    2. Convert line breaks after non-punctuation → space
    3. Fix extra spaces after English apostrophes
    4. Merge redundant spaces
    5. Preserve paragraph structure, ensure line breaks after punctuation
    """
    if not text:
        return ""

    text = text.strip()

    # First unify line breaks to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Merge non-empty lines into one sentence (remove original line breaks first)
    lines = [line.strip() for line in text.split('\n')]
    text = ' '.join(line for line in lines if line)

    # Add line break after sentence-ending punctuation (Chinese and English punctuation)
    text = re.sub(r'([.,!?:;，。！？；])\s*', r'\1\n', text)

    # Fix spaces after English apostrophes
    text = re.sub(r"'\s+", "'", text)

    # Merge redundant spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove leading and trailing spaces from lines
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()

# ===== Detection Functions =====
def is_ch_char(char:str):
    """Determine if a single character is a Chinese character"""
    if len(char) != 1:
        return False
    
    # Unicode ranges for Chinese characters
    # 1. Basic Chinese: 0x4E00-0x9FFF
    # 2. Extension A: 0x3400-0x4DBF
    # 3. Extension B: 0x20000-0x2A6DF
    # 4. Extension C: 0x2A700-0x2B73F
    # 5. Extension D: 0x2B740-0x2B81F
    # 6. Extension E: 0x2B820-0x2CEAF
    
    code = ord(char)
    
    # Common check (covers most cases)
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # Extension A
    if 0x3400 <= code <= 0x4DBF:
        return True
    # Other extensions not considered for now
    
    return False

# ===== File Operations =====

def load_txt(path:str) -> str:
    """Load a file as plain text"""
    with open(path, 'r') as file:
        content = file.read()
    return content

def load_json(path:str):
    """Load a JSON file"""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def load_jsonl(path:str, limit=-1) -> list[dict]:
    """Load a JSONL file"""
    data = []
    with open(path, 'r') as file:
        for id, line in tqdm(enumerate(file), desc=f"Loading {path}"):
            if limit != -1 and id == limit:
                break
            data.append(json.loads(line))
    return data

def save_json(data, path:str):
    """Save a JSON file"""
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_jsonl(data:list[dict], path:str, mode='w'):
    """Save a JSONL file"""
    with open(path, mode, encoding='utf-8') as file:
        for ele in tqdm(data, desc=f"Saving to {path}"):
            json.dump(ele, file, ensure_ascii=False)
            file.write("\n")

def audio_cut(input_path, mode:str, output_dir:str, segment_length:int=30000):
    """
    Extract a segment of specified length from an audio file
    - mode: Cut type (random / middle)
    - output_dir: Output folder
    - segment_length: Segment length (milliseconds)
    """
    assert mode in ['random', 'middle']

    # Check if file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    # Load audio file
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(44100).set_channels(1) # Set sample rate and channels
    audio_duration = len(audio) # Duration control
    
    # If audio length is less than target segment length, use entire audio
    if audio_duration <= segment_length:
        print(f"Warning: Audio too short ({audio_duration}ms), using full audio: {input_path}")
        segment = audio
    else:
        # Calculate slice position based on mode
        if mode == "random":
            # Random cut
            max_start = max(0, audio_duration - segment_length)
            start = random.randint(0, max_start)
            end = start + segment_length
        else:
            # Cut from middle
            middle_point = audio_duration // 2
            start = max(0, middle_point - (segment_length // 2))
            end = min(audio_duration, start + segment_length)
            
            # If cutting from middle would exceed boundaries, adjust start position
            if end > audio_duration:
                end = audio_duration
                start = end - segment_length
            elif start < 0:
                start = 0
                end = segment_length
        
        # Ensure slice range is valid
        start = max(0, min(start, audio_duration))
        end = max(0, min(end, audio_duration))
        
        if start >= end:
            raise ValueError(f"Invalid slice range: start={start}, end={end}, duration={audio_duration}")
        
        # Execute slice
        segment = audio[start:end]
    
    # Generate output path
    basename = pure_name(input_path)
    output_path = os.path.join(output_dir, f"seg_{basename}.wav")
    
    # Save segment
    segment.export(
        output_path, 
        format="wav",
        codec="pcm_s16le",  # 16-bit little-endian encoding
        parameters=["-acodec", "pcm_s16le"]  # ffmpeg parameters
    )
    return output_path

def format_meta(dir:str, show:bool=True) -> list[dict]:
    """Recursively get all audio paths (wav / mp3) in a folder and build JSONL"""
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
    Remove already generated items from dataset
    - key is the primary key in raw dataset, foreign key in save
    - seg is the target field
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
    Determine the number of files that can fit in a block before compression (assuming uniform file sizes)
    - data_dir: Folder to compress
    - subfixes: File suffixes to compress (e.g., .mp3)
    - per: Check every N files on average
    - max_size: Maximum limit in GB
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
    """Compress files in a directory in chunks (non-recursive)"""
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
            print(f"Compression completed: {save_path}")
        else:
            print(f"Compression failed: tar return code={tar_process.returncode}, pigz return code={pigz_process.returncode}")

def music_avg_size(dir:str):
    """Average music size (MB), length (s)"""
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
    """Get N records from a JSONL file"""
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
    """Process data from one path"""
    with open(path, 'r') as file:
        data = json.load(file)
    new_data = {
        "id": f"{data['song_id']}_{data['track_index']}",
        field: data[field]
    }
    return new_data

def get_field_suno(dir:str, save_path:str, field:str, max_workers:int=8):
    """Extract a specific field from scattered JSON files in suno"""
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
    """Find JSONL / JSON files in a folder"""
    names = []
    for name in tqdm(os.listdir(dir), desc="Finding JSON/JSONL"):
        if name.endswith(".json") or name.endswith(".jsonl"):
            names.append(name)
    return names

def show_dir(dir:str):
    """Display all contents in a directory"""
    if not os.path.isdir(dir):
        return
    for name in os.listdir(dir):
        print(name)

def _convert_mp3(path:str, dir:str):
    """Process a single audio file"""
    purename = pure_name(path)
    output_path = os.path.join(dir, purename + ".mp3")
    if os.path.exists(output_path):
        # Already completed
        return "pass"
    try:
        audio = AudioSegment.from_file(path)
    except Exception:
        # Failed to read file
        print(f"fail to load {path}")
        return "fail"
    audio.export(output_path, format='mp3')
    return "finish"

def convert_mp3(meta_path:str, dir:str, max_workers:int=10):
    """Convert all specified audio files to mp3 and save in specified directory"""
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

# ===== GPU and Models =====

def get_free_gpu() -> int:
    """Return the GPU ID with the least memory usage"""
    cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    result = subprocess.check_output(cmd.split()).decode().strip().split("\n")

    free_list = []
    for line in result:
        idx, free_mem = line.split(",")
        free_list.append((int(idx), int(free_mem)))  # (GPU id, free memory MiB)
    
    # Sort by remaining memory
    free_list.sort(key=lambda x: x[1], reverse=True)
    return free_list[0][0]

# ===== Data Analysis =====

def compose_analyze(dataset:list[dict]):
    """Statistical analysis of music structure composition"""
    # Label count
    labels = defaultdict(int)
    for ele in tqdm(dataset):
        segments = ele['segments']
        for segment in segments:
            label = segment['label']
            labels[label] += 1
    print(f"Number of labels: {len(labels)}")
    print(dict_sort_print(labels))

    # Different combinations
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
    print(f"Number of combinations: {len(label_combs)}")
    print(dict_sort_print(label_combs))

def _filter_tag(content:str) -> list[str]:
    """Split and format tag fields"""
    tags = []
    raws = re.split(r'[,，.]', content)
    for raw in raws:
        raw = raw.strip().lower()   # Remove spaces and convert to lowercase
        if raw == "":
            continue
        seg_pos = raw.find(":")
        if seg_pos != -1:
            # If colon exists, only take the part after it
            tag = raw[seg_pos+1:].strip()
        else:
            tag = raw
        tags.append(tag)
    return tags

def tags_analyze(dataset:list[dict]):
    """Song tag analysis"""
    tag_count = defaultdict(int)
    for ele in tqdm(dataset, desc="Tag analyzing"):
        tags = _filter_tag(ele['style'])
        for tag in tags:
            tag_count[tag] += 1
    print(f"Number of tags: {len(tag_count.keys())}")
    print(dict_sort_print(tag_count))