import os
import json
import time
import random
import re
from openai import OpenAI
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置环境变量
os.environ["OPENAI_API_KEY"] = ""  # 替换为你的API密钥
os.environ["OPENAI_BASE_URL"] = "https://chatapi.littlewheat.com/v1"  # 替换为API的URL

# 初始化客户端
client = OpenAI()


def _extract_lyrics_timestamps(lyrics_text):
    """
    提取歌词中的时间戳并转换为秒数
    Args:
        lyrics_text: 歌词字符串
    Returns:
        List[float]: 按顺序排列的时间戳（秒）
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
    校验歌词的时间戳是否满足要求
    Args:
        lyrics_text: 歌词字符串
        min_last_timestamp: 最后一个时间戳的最小值（秒）
        max_interval: 最后两个时间戳最大间隔（秒）
    Returns:
        bool: 是否通过校验
    """
    timestamps = _extract_lyrics_timestamps(lyrics_text)
    if len(timestamps) < 2:
        print("校验失败：时间戳数量少于2个")
        return False
    last = timestamps[-1]
    second_last = timestamps[-2]
    if last < min_last_timestamp:
        print(f"校验失败：最后一个时间戳 {last:.2f}s 小于 {min_last_timestamp}s")
        return False
    if last - second_last > max_interval:
        print(f"校验失败：最后两个时间戳间隔 {last - second_last:.2f}s 大于 {max_interval}s")
        return False
    return True


def chat_gpt(text, model='gpt-4o-mini'):
    while True:
        try:
            # 调用 OpenAI 的 chat completions 接口
            completion = client.chat.completions.create(
                model=model,  # 使用 GPT-4o-mini 模型
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ]
            )
            # 获取回复内容
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content.strip()
                return content
            else:
                print('error_wait_2s')
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)


def chat_gpt_call(text, model='gpt-4o-mini'):
    # 调用 OpenAI 的 chat completions 接口
    completion = client.chat.completions.create(
        model=model,  # 使用 GPT-4o-mini 模型
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )
    # 获取回复内容
    if getattr(completion.choices[0].message, 'content', None):
        content = completion.choices[0].message.content.strip()
        return content
    else:
        print('error_wait_2s')


def generate_music_descriptions(all_music_data, index_pool, output_file, file_lock, sample_size=20, model='gpt-4o-mini', max_retries=0):
    """
    读取音乐数据文件，随机抽取样本，调用GPT生成新的音乐描述和歌词
    
    Args:
        all_music_data: 所有音乐数据列表
        index_pool: 索引池对象（线程安全）
        output_file: 输出的jsonl文件路径
        file_lock: 文件写入锁
        sample_size: 随机抽取的样本数量
        model: 使用的模型名称
        max_retries: 最大重试次数
    
    Returns:
        (used_indices, success_count): 使用的索引列表和成功生成的数量
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
    
    # 直接转换为时间戳格式（严格对应元组的左右两端）
    start_timestamp = f"[{selected_range[0].replace('.', ':')}.00]"
    end_timestamp = f"[{selected_range[1].replace('.', ':')}.00]"
    
    # 生成时长描述
    start_duration = f"{selected_range[0].replace('.', '分')}秒"
    end_duration = f"{selected_range[1].replace('.', '分')}秒"
    
    # 生成示例中的随机时间戳（在时间范围内随机生成）
    # 解析时间字符串为分钟和秒
    start_parts = selected_range[0].split('.')
    end_parts = selected_range[1].split('.')
    
    start_minutes = int(start_parts[0])
    start_seconds = int(start_parts[1])
    end_minutes = int(end_parts[0])
    end_seconds = int(end_parts[1])
    
    # 转换为总秒数
    start_total_seconds = start_minutes * 60 + start_seconds
    end_total_seconds = end_minutes * 60 + end_seconds
    
    # 在范围内随机生成
    example1_seconds = random.randint(start_total_seconds, end_total_seconds)
    example2_seconds = random.randint(start_total_seconds, end_total_seconds)
    
    example1_minutes = example1_seconds // 60
    example1_secs = example1_seconds % 60
    example2_minutes = example2_seconds // 60
    example2_secs = example2_seconds % 60
    
    example1_timestamp = f"[{example1_minutes:02d}:{example1_secs:02d}.00]"
    example2_timestamp = f"[{example2_minutes:02d}:{example2_secs:02d}.00]"
    
    # 从索引池中获取样本索引（线程安全）
    selected_indices = index_pool.get_indices(sample_size)
    if not selected_indices:
        return [], 0
    
    sample_data = [all_music_data[i] for i in selected_indices]
    
    # 提取所有不重复的style
    styles = []
    for data in sample_data:
        style = data.get('style', '')
        if style and style not in styles:
            styles.append(style)
    
    styles_text = "、".join(styles)
    
    # 构建示例文本 - 将所有采样数据都纳入（不包含style）
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
    # 尝试生成，带重试机制
    for attempt in range(max_retries + 1):
        try:
            # 调用OpenAI API
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
            # 提取所有回答
            results = []
            filtered_count = 0
            last_content = None
            
            for i, choice in enumerate(completion.choices, 1):
                try:
                    content = choice.message.content.strip()
                    last_content = content
                    print(f"\n=== GPT回复 {i} ===")
                    print(content)
                    print("=" * 50)
                    # 尝试提取JSON内容
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # 清理JSON中的trailing commas（多余的逗号）
                    # 移除对象/数组最后一个元素后的逗号
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    
                    # 解析JSON数组
                    result_array = json.loads(content)
                    
                    # 确保是列表
                    if isinstance(result_array, list):
                        # 验证数组中的每个对象
                        for song in result_array:
                            if isinstance(song, dict) and 'description' in song and 'lyrics' in song:
                                if _validate_timestamps(song.get('lyrics', '')):
                                    results.append(song)
                                else:
                                    filtered_count += 1
                    # 如果返回的是单个对象（兼容旧格式）
                    elif isinstance(result_array, dict) and 'description' in result_array and 'lyrics' in result_array:
                        if _validate_timestamps(result_array.get('lyrics', '')):
                            results.append(result_array)
                        else:
                            filtered_count += 1
                        
                except json.JSONDecodeError:
                    continue
            
            if filtered_count:
                print(f"共有 {filtered_count} 首歌曲因时间戳校验未通过而被过滤")
            
            # 打印解析结果
            print(f"\n解析完成，results长度: {len(results)}")
            print(f"results内容: {results}")
            print(start_duration, end_duration,example1_timestamp,example2_timestamp,require_length)
            
            # 如果解析的result长度不为5，将模型的回复content写入test.txt
            if len(results) != 2:
                print(f"警告：解析结果长度不为2，实际为{len(results)}，将写入test.txt")
                with open('test.txt', 'w', encoding='utf-8') as f:
                    if last_content is not None:
                        f.write(last_content)
                print("已写入test.txt文件")
            
            # 检查是否成功生成了50首（10个回复 * 每个5首）
            if len(results) >= 50:
                # 追加保存结果到文件（使用锁保证线程安全）
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for result in results[:50]:  # 只保存前50首
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                return selected_indices, min(len(results), 50)
            elif attempt < max_retries:
                print(f"只成功解析了 {len(results)}/50 首歌曲，正在重试...")
                time.sleep(2)
            else:
                # 最后一次尝试，即使不够50首也保存
                if len(results) > 0:
                    with file_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for result in results:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                return selected_indices, len(results)
                
        except Exception as e:
            if attempt < max_retries:
                print(f"生成过程中出现错误: {e}，正在重试...")
                time.sleep(2)
            else:
                print(f"生成失败: {e}")
                return selected_indices, 0
    
    return selected_indices, 0


class IndexPool:
    """线程安全的索引池，支持自动重置"""
    
    def __init__(self, total_size, selected_file):
        self.total_size = total_size
        self.selected_file = selected_file
        self.lock = threading.Lock()
        self.available_indices = []
        self.selected_indices = set()
        self.reset_count = 0  # 记录重置次数
        
        # 从文件加载已选择的索引
        self._load_selected_indices()
        # 初始化可用索引
        self._reset_pool()
    
    def _load_selected_indices(self):
        """从文件加载已选择的索引"""
        if os.path.exists(self.selected_file):
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.selected_indices.add(int(line.strip()))
    
    def _reset_pool(self):
        """重置索引池"""
        # 计算可用索引
        self.available_indices = [i for i in range(self.total_size) if i not in self.selected_indices]
        random.shuffle(self.available_indices)  # 打乱顺序
        
        if len(self.available_indices) == 0:
            # 如果没有可用索引了，说明已经全部使用过，重置selected_indices
            self.reset_count += 1
            print(f"\n索引池已用完，第 {self.reset_count} 次重置池子，重新从 {self.total_size} 首歌曲中抽取")
            self.selected_indices.clear()
            self.available_indices = list(range(self.total_size))
            random.shuffle(self.available_indices)
    
    def get_indices(self, count):
        """
        线程安全地获取指定数量的索引
        
        Args:
            count: 需要获取的索引数量
            
        Returns:
            选中的索引列表
        """
        with self.lock:
            # 检查是否需要重置池子
            if len(self.available_indices) < count:
                self._reset_pool()
            
            # 获取索引
            selected = self.available_indices[:count]
            self.available_indices = self.available_indices[count:]
            
            # 添加到已选择集合
            for idx in selected:
                self.selected_indices.add(idx)
            
            # 写入文件
            with open(self.selected_file, 'a', encoding='utf-8') as f:
                for idx in selected:
                    f.write(f"{idx}\n")
            
            return selected
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'available': len(self.available_indices),
                'selected': len(self.selected_indices),
                'reset_count': self.reset_count
            }


def batch_generate_music(input_file, output_file, selected_file, total_songs=1000, sample_size=20, model='gpt-4o-mini', num_threads=10):
    """
    批量生成音乐描述和歌词（多线程版本）
    
    Args:
        input_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
        selected_file: 记录已选择索引的文件路径
        total_songs: 总共需要生成的歌曲数量
        sample_size: 每次抽取的样本数量
        model: 使用的模型名称
        num_threads: 线程数量
    """
    # 读取所有音乐数据
    print("正在加载音乐数据...")
    all_music_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_music_data.append(data)
    print(f"共加载了 {len(all_music_data)} 首歌曲")
    
    # 创建线程安全的索引池
    index_pool = IndexPool(len(all_music_data), selected_file)
    stats = index_pool.get_stats()
    print(f"当前已选择的索引数量: {stats['selected']}")
    print(f"当前可用的索引数量: {stats['available']}")
    
    # 计算需要调用的次数（每次生成5首）
    num_iterations = (total_songs + 1) // 2  # 向上取整
    print(f"需要调用 {num_iterations} 次，生成约 {total_songs} 首歌曲（每次5首）")
    print(f"使用 {num_threads} 个线程并行处理\n")
    
    # 创建文件写入锁
    file_lock = threading.Lock()
    
    # 统计信息
    total_generated = 0
    generated_lock = threading.Lock()
    
    def worker_task(task_id):
        """工作线程任务"""
        try:
            used_indices, success_count = generate_music_descriptions(
                all_music_data=all_music_data,
                index_pool=index_pool,
                output_file=output_file,
                file_lock=file_lock,
                sample_size=sample_size,
                model=model,
                max_retries=0  # 重试
            )
            return success_count
        except Exception as e:
            print(f"任务 {task_id} 失败: {e}")
            return 0
    
    # 使用线程池和进度条
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = {executor.submit(worker_task, i): i for i in range(num_iterations)}
        
        # 使用tqdm显示进度
        with tqdm(total=num_iterations, desc="生成进度", unit="批次") as pbar:
            for future in as_completed(futures):
                success_count = future.result()
                
                with generated_lock:
                    total_generated += success_count
                
                # 获取当前统计信息
                stats = index_pool.get_stats()
                
                # 更新进度条
                pbar.set_postfix({
                    '本批次': f'{success_count}/5',
                    '累计': total_generated,
                    '剩余索引': stats['available'],
                    '重置次数': stats['reset_count']
                })
                pbar.update(1)
    
    # 最终统计
    stats = index_pool.get_stats()
    print(f"\n生成完成！")
    print(f"总共生成了 {total_generated} 首歌曲")
    print(f"使用了 {stats['selected']} 个索引")
    print(f"剩余可用索引: {stats['available']}")
    print(f"池子重置次数: {stats['reset_count']}")


if __name__ == '__main__':
    input_file = 'tagged_musics.jsonl'
    output_file = 'generate_lrc_5mini.jsonl'
    selected_file = 'selected.txt'
    #n=1, max_retries=0，每次采样10首，生成5首新歌曲
    batch_generate_music(
        input_file=input_file,
        output_file=output_file,
        selected_file=selected_file,
        total_songs=10,
        sample_size=4,
        model='gpt-5-mini',
        num_threads=5  # 先用1个线程测试
    )
    #model='gemini-2.5-pro-preview-06-05'
    #3.30~3.45、4.30-4.45
    #txt追加写