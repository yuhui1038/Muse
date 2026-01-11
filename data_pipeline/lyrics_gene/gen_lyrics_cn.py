import os
import json
import time
import random
import re
from openai import OpenAI
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables
# Note: Set these environment variables before running the script
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your custom API URL
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ""  # Replace with your API key or set via environment variable
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # Replace with API URL or set via environment variable

# Initialize client
client = OpenAI()


def _extract_lyrics_timestamps(lyrics_text):
    """
    Extract timestamps from lyrics and convert to seconds
    Args:
        lyrics_text: Lyrics string
    Returns:
        List[float]: Timestamps in order (seconds)
    """
    if not isinstance(lyrics_text, str):
        return []
    pattern = re.compile(r'\[(\d{2}):(\d{2})(?:\.(\d{2}))?\]')
    timestamps = []
    for match in pattern.finditer(lyrics_text):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction = match.group(3)
        total_seconds = minutes * 60 + seconds
        if fraction is not None:
            divisor = 100 if len(fraction) == 2 else 10 ** len(fraction)
            total_seconds += int(fraction) / divisor
        timestamps.append(total_seconds)
    return timestamps


def _validate_timestamps(lyrics_text, min_last_timestamp=170, max_interval=35):
    """
    Validate if lyrics timestamps meet requirements
    Args:
        lyrics_text: Lyrics string
        min_last_timestamp: Minimum value of last timestamp (seconds)
        max_interval: Maximum interval between last two timestamps (seconds)
    Returns:
        bool: Whether validation passes
    """
    timestamps = _extract_lyrics_timestamps(lyrics_text)
    if len(timestamps) < 2:
        print("Validation failed: Timestamp count less than 2")
        return False
    last = timestamps[-1]
    second_last = timestamps[-2]
    if last < min_last_timestamp:
        print(f"Validation failed: Last timestamp {last:.2f}s is less than {min_last_timestamp}s")
        return False
    if last - second_last > max_interval:
        print(f"Validation failed: Interval between last two timestamps {last - second_last:.2f}s is greater than {max_interval}s")
        return False
    return True


def chat_gpt(text, model='gpt-4o-mini'):
    while True:
        try:
            # Call OpenAI chat completions API
            completion = client.chat.completions.create(
                model=model,  # Use GPT-4o-mini model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ]
            )
            # Get reply content
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content.strip()
                return content
            else:
                print('error_wait_2s')
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)


def chat_gpt_call(text, model='gpt-4o-mini'):
    # Call OpenAI chat completions API
    completion = client.chat.completions.create(
        model=model,  # Use GPT-4o-mini model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )
    # Get reply content
    if getattr(completion.choices[0].message, 'content', None):
        content = completion.choices[0].message.content.strip()
        return content
    else:
        print('error_wait_2s')


def generate_music_descriptions(all_music_data, index_pool, output_file, file_lock, sample_size=20, model='gpt-4o-mini', max_retries=0):
    """
    Read music data file, randomly sample, call GPT to generate new music descriptions and lyrics
    
    Args:
        all_music_data: All music data list
        index_pool: Index pool object (thread-safe)
        output_file: Output jsonl file path
        file_lock: File write lock
        sample_size: Number of random samples
        model: Model name to use
        max_retries: Maximum retry count
    
    Returns:
        (used_indices, success_count): List of used indices and count of successful generations
    """
    # duration_ranges = [
    #     ("3.00", "3.15", 50), ("3.15", "3.30", 50), ("3.30", "3.45", 60),
    #  ("3.45", "4.00", 60),
    #     ("4.00", "4.15", 70), ("4.15", "4.30", 70), ("4.30", "4.45", 70)
    # ]
    duration_ranges = [
        ("3.00", "3.15", 60), ("3.15", "3.30", 70), ("3.30", "3.45", 80),("3.45", "4.00", 90),
        ("4.00", "4.15", 100), ("4.15", "4.30", 100), ("4.30", "4.45", 100)
    ]
    selected_range = random.choice(duration_ranges)
    require_length = selected_range[2]
    
    # Directly convert to timestamp format (strictly corresponds to left and right ends of tuple)
    start_timestamp = f"[{selected_range[0].replace('.', ':')}.00]"
    end_timestamp = f"[{selected_range[1].replace('.', ':')}.00]"
    
    # Generate duration description
    start_duration = f"{selected_range[0].replace('.', '分')}秒"
    end_duration = f"{selected_range[1].replace('.', '分')}秒"
    
    # Generate random timestamps for examples (randomly generated within time range)
    # Parse time string to minutes and seconds
    start_parts = selected_range[0].split('.')
    end_parts = selected_range[1].split('.')
    
    start_minutes = int(start_parts[0])
    start_seconds = int(start_parts[1])
    end_minutes = int(end_parts[0])
    end_seconds = int(end_parts[1])
    
    # Convert to total seconds
    start_total_seconds = start_minutes * 60 + start_seconds
    end_total_seconds = end_minutes * 60 + end_seconds
    
    # Randomly generate within range
    example1_seconds = random.randint(start_total_seconds, end_total_seconds)
    example2_seconds = random.randint(start_total_seconds, end_total_seconds)
    
    example1_minutes = example1_seconds // 60
    example1_secs = example1_seconds % 60
    example2_minutes = example2_seconds // 60
    example2_secs = example2_seconds % 60
    
    example1_timestamp = f"[{example1_minutes:02d}:{example1_secs:02d}.00]"
    example2_timestamp = f"[{example2_minutes:02d}:{example2_secs:02d}.00]"
    
    # Get sample indices from index pool (thread-safe)
    selected_indices = index_pool.get_indices(sample_size)
    if not selected_indices:
        return [], 0
    
    sample_data = [all_music_data[i] for i in selected_indices]
    
    # Extract all unique styles
    styles = []
    for data in sample_data:
        style = data.get('style', '')
        if style and style not in styles:
            styles.append(style)
    
    styles_text = "、".join(styles)
    
    # Build example text - include all sampled data (excluding style)
    examples = []
    for i, data in enumerate(sample_data, 1):  
        lyrics_text = " ".join(data.get('lyrics', [])) if isinstance(data.get('lyrics'), list) else data.get('lyrics', '')
        description = data.get('description', '')
        examples.append(f"示例{i}:\ndescription: {description}\nlyrics: {lyrics_text}")
    
    examples_text = "\n\n".join(examples)

    prompt = f"""生成2首完整的歌曲，每首歌必须满足以下硬性指标：
- 严禁生成小于{require_length}行的歌词！
- 每首歌的歌词行数必须严格大于{require_length}行，这是硬性要求！
- 最后一句时间戳必须在{start_timestamp}到{end_timestamp}之间
- 两首歌的时长、行数必须有差异，严禁最后的时间戳均相同
- 相邻歌词行的时间戳间隔不得超过10秒！必须保证时间戳连续自然递进
- 严禁出现如"[03:25.00]在心中[04:25.00]最后一行歌词"的生硬间隔,严禁超过10s的间隔
如果生成的歌曲不满足以上任意一项，则视为不合格，请重新生成。
请生成2首新的、具有多样性的音乐描述和LRC格式歌词，语言为中文。


创作要求：
1.风格与流派需要确保多样性。
2.Description标签化要求（必须严格遵守）：
   description字段必须使用结构化的标签格式，包括以下标签，用逗号分隔：
   - 音乐风格标签
   - 音乐流派标签
   - 乐器标签
   - 情感基调标签
   - 氛围标签
   - 演唱方式和人声标签，仅限男声或女生二选一，单人独唱
   注意：每个标签简洁明了，多个同类标签可用斜杠分隔（如"钢琴/小提琴"）
3.歌词创造力：lyrics应该具有深度和艺术性：
   - 主题可以涉及爱情、人生、社会、自然、哲思、梦想、回忆等各个方面
   - 运用丰富的文学手法：比喻、意象、对比、排比等
   - 情感真挚，注重韵律和节奏感
   - 可以是叙事性、抒情性或意识流风格
4.歌词结构和长度要求（必须严格遵守）：
   - lyrics必须按照以下结构组织，并使用段落标签标注每个部分
   - 结构顺序必须严格遵循该顺序,共8个段落标签：[Verse 1]主歌1 → [Pre-Chorus]预副歌 → [Chorus]副歌 → [Verse 2]主歌2 → [Pre-Chorus]预副歌 → [Chorus]副歌 → [Bridge]桥段 → [Chorus (Outro)]副歌（结尾）
   - 一首歌词的段落标签只有8个，即[Verse 1]和[Verse 2]只出现一次,[Pre-Chorus]和[Chorus]各出现两次，[Bridge]和[Chorus (Outro)]各出现一次，禁止额外添加或重复更多的段落标签
   - 每个段落标签（如[Verse 1]、[Chorus]等）必须独占一行，后面紧跟该段落的LRC格式歌词
   - 段落之间用空行分隔
   - **总行数要求**：整首歌必须包含至少{require_length}行带时间戳的歌词（不包括段落标签行和空行）
5.LRC格式强制规则（必须严格遵守）：
   - 每行歌词格式必须为 `[mm:ss.xx]歌词内容`，时间戳与歌词间无空格，歌词内容需完整连贯
   - **每一行只能包含一小句歌词**，遇到逗号、句号等标点符号时必须换行。
   - **严禁将多句歌词合并在同一行**
   - 时间戳需自然分布，**第一句歌词起始时间不得为 [00:00.00]**，需考虑前奏空白（建议从[00:05.00]到[00:15.00]之间开始）
   - 时间戳间隔要求多样性：每首歌内部的时间戳间隔必须多样化，多采用小数点数间隔，严禁使用固定间隔：
     * 同一首歌内必须包含多种不同的间隔，不要所有句子都使用相同间隔（如不要全部都是4秒间隔）
     * 根据歌词内容的情感强度和音乐节拍来动态调整间隔
     * 相邻歌词行的间隔应该有所变化，体现音乐的节奏起伏
   - 时间戳分配应根据歌曲的风格、情感、节奏来合理推测，而非机械地按照歌词长度分配
   - 每行歌词长度应自然变化，切勿长度一致
   - **歌曲总时长必须达到{start_duration}到{end_duration}（即最后一句时间戳必须在{start_timestamp}到{end_timestamp}之间）这是硬性要求！**
6.歌词长度要求：lyrics字段的歌词行数必须大于{require_length}行，若生成长度过短请重新生成。
7.独特性和原创性：每首作品都应该是独一无二的，避免简单重复示例的内容。
8.格式要求：
   - 直接返回JSON数组格式，包含2个歌曲对象，每个对象只有description和lyrics两个字段
   - description字段：必须是标签格式，不是叙述性文本
   - lyrics字段：带段落标签的LRC格式字符串
   - 严禁在JSON中插入任何额外的符号、标记、注释或说明文字

LRC格式示例（带段落标签）：
[Verse 1]
[00:08.00]第一句歌词
[00:12.50]第二句歌词
[00:17.20]第三句歌词

[Pre-Chorus]
[00:22.00]预副歌歌词
[00:26.50]预副歌歌词

[Chorus]
[00:31.00]副歌歌词
[00:35.50]副歌歌词

负面示例（禁止出现）：
- 错误：[01:30.00](钢琴间奏) - 禁止在时间戳后加括号注释
- 错误：[00:00.00]开始的歌词 - 第一句不能从00:00.00开始
- 错误: [00:05.00]在那片熟悉的田野，阳光洒满金色的麦穗 - 严禁多句歌词放在同一行

现在，请充分发挥你的创造力，生成2首全新的、完整的音乐描述和LRC格式歌词作品。
特别提醒：每首歌必须是完整歌曲，不要缩写或省略！必须包含完整的8个段落（Verse 1, Pre-Chorus, Chorus, Verse 2, Pre-Chorus, Chorus, Bridge, Chorus Outro），严格确保大于{require_length}行歌词。

直接返回JSON数组格式：
[
  {{"description": "...", "lyrics": "..."}},
  {{"description": "...", "lyrics": "..."}}
]"""
    # Try to generate with retry mechanism
    for attempt in range(max_retries + 1):
        try:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are a creative music lyricist and composer. Please generate diverse and creative music tag-based descriptions and LRC format lyrics with song structure tags. CRITICAL REQUIREMENTS: 1) Description must be structured tags separated by commas, NOT narrative text. 2) Return ONLY pure, valid JSON format without any extra symbols, markers, or comments. 3) Each song must include structure tags like [Verse 1], [Chorus], [Bridge], etc., followed by LRC format lyrics [mm:ss.xx]lyric_content. 4) MANDATORY: Each song must have MORE than {require_length} lines of lyrics with timestamps. "},
                    {"role": "user", "content": prompt}
                ],
                n=1,  
                temperature=1.0,  
            )
            #print(prompt)
            # Extract all responses
            results = []
            filtered_count = 0
            last_content = None
            
            for i, choice in enumerate(completion.choices, 1):
                try:
                    content = choice.message.content.strip()
                    last_content = content
                    print(f"\n=== GPT Response {i} ===")
                    print(content)
                    print("=" * 50)
                    # Try to extract JSON content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # Clean trailing commas in JSON (extra commas)
                    # Remove commas after last element of object/array
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    
                    # Parse JSON array
                    result_array = json.loads(content)
                    
                    # Ensure it's a list
                    if isinstance(result_array, list):
                        # Validate each object in array
                        for song in result_array:
                            if isinstance(song, dict) and 'description' in song and 'lyrics' in song:
                                if _validate_timestamps(song.get('lyrics', '')):
                                    results.append(song)
                                else:
                                    filtered_count += 1
                    # If returned a single object (compatibility with old format)
                    elif isinstance(result_array, dict) and 'description' in result_array and 'lyrics' in result_array:
                        if _validate_timestamps(result_array.get('lyrics', '')):
                            results.append(result_array)
                        else:
                            filtered_count += 1
                        
                except json.JSONDecodeError:
                    continue
            
            if filtered_count:
                print(f"Total {filtered_count} songs filtered due to timestamp validation failure")
            
            # Print parsing results
            print(f"\nParsing complete, results length: {len(results)}")
            print(f"Results content: {results}")
            print(start_duration, end_duration,example1_timestamp,example2_timestamp,require_length)
            
            # If parsed result length is not 5, write model response content to test.txt
            if len(results) != 2:
                print(f"Warning: Parsed result length is not 2, actual is {len(results)}, will write to test.txt")
                with open('test.txt', 'w', encoding='utf-8') as f:
                    if last_content is not None:
                        f.write(last_content)
                print("Written to test.txt file")
            
            # Check if successfully generated 50 songs (10 responses * 5 each)
            if len(results) >= 50:
                # Append save results to file (use lock to ensure thread safety)
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for result in results[:50]:  # Only save first 50 songs
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                return selected_indices, min(len(results), 50)
            elif attempt < max_retries:
                print(f"Only successfully parsed {len(results)}/50 songs, retrying...")
                time.sleep(2)
            else:
                # Last attempt, save even if not 50 songs
                if len(results) > 0:
                    with file_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for result in results:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                return selected_indices, len(results)
                
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred during generation: {e}, retrying...")
                time.sleep(2)
            else:
                print(f"Generation failed: {e}")
                return selected_indices, 0
    
    return selected_indices, 0


class IndexPool:
    """Thread-safe index pool with automatic reset support"""
    
    def __init__(self, total_size, selected_file):
        self.total_size = total_size
        self.selected_file = selected_file
        self.lock = threading.Lock()
        self.available_indices = []
        self.selected_indices = set()
        self.reset_count = 0  # Record reset count
        
        # Load selected indices from file
        self._load_selected_indices()
        # Initialize available indices
        self._reset_pool()
    
    def _load_selected_indices(self):
        """Load selected indices from file"""
        if os.path.exists(self.selected_file):
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.selected_indices.add(int(line.strip()))
    
    def _reset_pool(self):
        """Reset index pool"""
        # Calculate available indices
        self.available_indices = [i for i in range(self.total_size) if i not in self.selected_indices]
        random.shuffle(self.available_indices)  # Shuffle order
        
        if len(self.available_indices) == 0:
            # If no available indices, all have been used, reset selected_indices
            self.reset_count += 1
            print(f"\nIndex pool exhausted, resetting pool for the {self.reset_count}th time, re-selecting from {self.total_size} songs")
            self.selected_indices.clear()
            self.available_indices = list(range(self.total_size))
            random.shuffle(self.available_indices)
    
    def get_indices(self, count):
        """
        Thread-safe get specified number of indices
        
        Args:
            count: Number of indices needed
            
        Returns:
            List of selected indices
        """
        with self.lock:
            # Check if pool needs to be reset
            if len(self.available_indices) < count:
                self._reset_pool()
            
            # Get indices
            selected = self.available_indices[:count]
            self.available_indices = self.available_indices[count:]
            
            # Add to selected set
            for idx in selected:
                self.selected_indices.add(idx)
            
            # Write to file
            with open(self.selected_file, 'a', encoding='utf-8') as f:
                for idx in selected:
                    f.write(f"{idx}\n")
            
            return selected
    
    def get_stats(self):
        """Get statistics"""
        with self.lock:
            return {
                'available': len(self.available_indices),
                'selected': len(self.selected_indices),
                'reset_count': self.reset_count
            }


def batch_generate_music(input_file, output_file, selected_file, total_songs=1000, sample_size=20, model='gpt-4o-mini', num_threads=10):
    """
    Batch generate music descriptions and lyrics (multi-threaded version)
    
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output jsonl file
        selected_file: Path to file recording selected indices
        total_songs: Total number of songs to generate
        sample_size: Number of samples to extract each time
        model: Model name to use
        num_threads: Number of threads
    """
    # Load all music data
    print("Loading music data...")
    all_music_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_music_data.append(data)
    print(f"Loaded {len(all_music_data)} songs")
    
    # Create thread-safe index pool
    index_pool = IndexPool(len(all_music_data), selected_file)
    stats = index_pool.get_stats()
    print(f"Currently selected indices: {stats['selected']}")
    print(f"Currently available indices: {stats['available']}")
    
    # Calculate number of calls needed (5 songs per call)
    num_iterations = (total_songs + 1) // 2  # Round up
    print(f"Need to call {num_iterations} times to generate approximately {total_songs} songs (5 per call)")
    print(f"Using {num_threads} threads for parallel processing\n")
    
    # Create file write lock
    file_lock = threading.Lock()
    
    # Statistics
    total_generated = 0
    generated_lock = threading.Lock()
    
    def worker_task(task_id):
        """Worker thread task"""
        try:
            used_indices, success_count = generate_music_descriptions(
                all_music_data=all_music_data,
                index_pool=index_pool,
                output_file=output_file,
                file_lock=file_lock,
                sample_size=sample_size,
                model=model,
                max_retries=0  # Retry
            )
            return success_count
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            return 0
    
    # Use thread pool and progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(worker_task, i): i for i in range(num_iterations)}
        
        # Use tqdm to show progress
        with tqdm(total=num_iterations, desc="Generation progress", unit="batch") as pbar:
            for future in as_completed(futures):
                success_count = future.result()
                
                with generated_lock:
                    total_generated += success_count
                
                # Get current statistics
                stats = index_pool.get_stats()
                
                # Update progress bar
                pbar.set_postfix({
                    'Batch': f'{success_count}/5',
                    'Total': total_generated,
                    'Remaining': stats['available'],
                    'Resets': stats['reset_count']
                })
                pbar.update(1)
    
    # Final statistics
    stats = index_pool.get_stats()
    print(f"\nGeneration complete!")
    print(f"Total generated: {total_generated} songs")
    print(f"Used {stats['selected']} indices")
    print(f"Remaining available indices: {stats['available']}")
    print(f"Pool reset count: {stats['reset_count']}")


if __name__ == '__main__':
    input_file = 'tagged_musics.jsonl'
    output_file = 'generate_lrc_5mini.jsonl'
    selected_file = 'selected.txt'
    # n=1, max_retries=0, sample 10 songs each time, generate 5 new songs
    batch_generate_music(
        input_file=input_file,
        output_file=output_file,
        selected_file=selected_file,
        total_songs=10,
        sample_size=4,
        model='gpt-5-mini',
        num_threads=5  # Test with 1 thread first
    )
    # Append to txt file