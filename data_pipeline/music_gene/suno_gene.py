# -*- coding: utf-8 -*-
"""
Suno API 批量生成 - 完善时间统计版本
改进要点：
1. 精确的速率控制：10秒内准确发出20次请求
2. 分离请求提交和结果等待，提高并发效率
3. 动态调整并发数，避免资源浪费
4. 完善的时间统计和性能分析
"""
import json
import time
import requests
import os
import logging
import csv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from threading import Lock, Semaphore
from tqdm import tqdm
from config import SUNO_API_KEY


# 配置日志
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f"run_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 创建 logger
    logger = logging.getLogger('SunoBatch')
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handler
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 文件 Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_file

# 全局 logger
logger = logging.getLogger('SunoBatch')

# 替换 print 为 logger.info
def print_log(msg):
    logger.info(msg)



class SunoAPI:
    """简化的 Suno API 客户端"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.sunoapi.org/api/v1'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 配置重试策略
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,  # 最大重试次数
            backoff_factor=1,  # 重试间隔 (1s, 2s, 4s, 8s...)
            status_forcelist=[500, 502, 503, 504],  # 需要重试的状态码
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]  # 允许重试的方法
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def generate_music(self, prompt, model='V5', vocalGender=None, **options):
        """生成音乐"""
        payload = {
            'prompt': prompt,
            'model': model,
            'callBackUrl': 'https://example.com/callback',
            **options
        }
        
        if vocalGender:
            payload['vocalGender'] = vocalGender
        
        try:
            response = self.session.post(
                f'{self.base_url}/generate',
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # 检查 HTTP 错误
            response.raise_for_status()
            
            # 尝试解析 JSON
            try:
                result = response.json()
            except json.JSONDecodeError:
                raise Exception(f"API 返回非 JSON 响应: {response.text[:200]}")
                
            if result.get('code') != 200:
                raise Exception(f"生成失败: {result.get('msg', result)}")
            
            return result['data']['taskId']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求异常: {str(e)}")
    
    def get_task_status(self, task_id):
        """获取任务状态"""
        try:
            response = self.session.get(
                f'{self.base_url}/generate/record-info?taskId={task_id}',
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            # 状态查询失败不应导致程序崩溃，返回空字典或抛出特定异常
            # print_log(f"获取状态失败: {e}")
            raise e
    
    def get_timestamped_lyrics(self, task_id, audio_id):
        """获取带时间戳的歌词"""
        payload = {
            'taskId': task_id,
            'audioId': audio_id
        }
        
        try:
            response = self.session.post(
                f'{self.base_url}/generate/get-timestamped-lyrics',
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}  # 歌词获取失败非致命错误
    
    def wait_for_completion(self, task_id, max_wait_time=600, check_interval=5):
        """等待任务完成，返回结果和轮询统计"""
        start_time = time.time()
        poll_count = 0
        total_poll_time = 0
        
        while time.time() - start_time < max_wait_time:
            try:
                poll_start = time.time()
                status = self.get_task_status(task_id)
                poll_time = time.time() - poll_start
                poll_count += 1
                total_poll_time += poll_time
                
                current_status = status.get('status')
                
                if current_status == 'SUCCESS':
                    return {
                        'result': status.get('response'),
                        'wait_time': time.time() - start_time,
                        'poll_count': poll_count,
                        'avg_poll_time': total_poll_time / poll_count if poll_count > 0 else 0
                    }
                elif current_status == 'FAILED':
                    raise Exception(f"任务失败: {status.get('errorMessage')}")
                
                time.sleep(check_interval)
            except Exception as e:
                if time.time() - start_time >= max_wait_time:
                    raise
                time.sleep(check_interval)
        
        raise Exception('任务超时')
    
    def download_file(self, url, save_path):
        """下载文件到本地，返回下载统计"""
        try:
            start_time = time.time()
            downloaded_bytes = 0
            
            # 使用 session 下载
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
            
            download_time = time.time() - start_time
            return {
                'success': True,
                'bytes': downloaded_bytes,
                'time': download_time,
                'speed': downloaded_bytes / download_time if download_time > 0 else 0
            }
        except Exception as e:
            print_log(f"下载失败 {url}: {e}")
            return {'success': False, 'error': str(e)}


# 结果记录锁
result_lock = Lock()

def save_result_record(output_dir, record):
    """实时保存单条结果到 CSV"""
    file_path = os.path.join(output_dir, "generation_results.csv")
    file_exists = os.path.isfile(file_path)
    
    # 只需要记录关键信息
    row = {
        'song_id': record.get('song_id'),
        'task_id': record.get('task_id'),
        'status': 'SUCCESS' if record.get('success') else 'FAILED',
        'error': record.get('error', ''),
        'submit_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.get('submit_time', 0))),
        'total_time': f"{record.get('total_time', 0):.1f}",
        'tracks_count': record.get('tracks_count', 0)
    }
    
    with result_lock:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['song_id', 'task_id', 'status', 'error', 'submit_time', 'total_time', 'tracks_count'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


class ImprovedRateLimiter:
    """改进的速率限制器（带统计功能）
    
    精确控制：每10秒最多20次请求
    使用滑动窗口算法，确保任意10秒时间窗口内不超过20次请求
    """
    
    def __init__(self, max_requests=20, time_window=10):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = Lock()
        self.semaphore = Semaphore(max_requests)
        
        # 统计信息
        self.total_wait_time = 0
        self.wait_count = 0
        self.total_requests = 0
    
    def acquire(self):
        """获取请求许可"""
        with self.lock:
            now = time.time()
            
            # 清理过期的请求记录
            while self.request_times and now - self.request_times[0] >= self.time_window:
                self.request_times.popleft()
            
            # 如果已达到限制，计算需要等待的时间
            wait_time = 0
            if len(self.request_times) >= self.max_requests:
                oldest_request = self.request_times[0]
                wait_time = self.time_window - (now - oldest_request) + 0.05  # 加缓冲
                
                if wait_time > 0:
                    print_log(f"  [速率限制] 等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    
                    # 统计等待时间
                    self.total_wait_time += wait_time
                    self.wait_count += 1
                    
                    # 重新清理
                    now = time.time()
                    while self.request_times and now - self.request_times[0] >= self.time_window:
                        self.request_times.popleft()
            
            # 记录本次请求时间
            self.request_times.append(time.time())
            self.total_requests += 1
            
    def get_current_rate(self):
        """获取当前速率（最近10秒内的请求数）"""
        with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] >= self.time_window:
                self.request_times.popleft()
            return len(self.request_times)
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'total_requests': self.total_requests,
                'total_wait_time': self.total_wait_time,
                'wait_count': self.wait_count,
                'avg_wait_time': self.total_wait_time / self.wait_count if self.wait_count > 0 else 0
            }


# 全局速率限制器
rate_limiter = ImprovedRateLimiter(max_requests=20, time_window=10)


def submit_generation_task(api, song_index, data):
    """第一阶段：提交生成任务（受速率限制）"""
    # 使用 suno_test_cn_000001 格式
    song_id = data.get("id", f"suno_test_cn_{song_index:06d}")
    
    try:
        description = data.get("description", "")
        lyrics = data.get("lyrics", "")
        vocal_gender = data.get("vocalGender")
        
        print_log(f"[歌曲 {song_id}] 提交任务... (当前速率: {rate_limiter.get_current_rate()}/20)")
        
        # 记录请求开始时间
        request_start = time.time()
        
        # 速率限制
        rate_limiter.acquire()
        
        # 提交任务
        submit_start = time.time()
        task_id = api.generate_music(
            prompt=lyrics,
            style=description,
            title=f"Song_{song_id}",
            model='V5',
            customMode=True,
            instrumental=False,
            vocalGender=vocal_gender
        )
        request_time = time.time() - submit_start
        
        print_log(f"[歌曲 {song_id}] ✓ 任务已提交，ID: {task_id}")
        
        return {
            'song_id': song_id,
            'song_index': song_index,
            'task_id': task_id,
            'data': data,
            'submit_time': time.time(),
            'request_time': request_time,
            'success': True
        }
        
    except Exception as e:
        print_log(f"[歌曲 {song_id}] ✗ 提交失败: {e}")
        # 如果提交失败，也记录下来（虽然还没到下载阶段）
        # 这里暂时不记录到 generation_results.csv，因为那个文件主要用于记录最终结果
        # 但如果需要全量审计，可以在这里增加记录
        return {
            'song_id': song_id,
            'song_index': song_index,
            'success': False,
            'error': str(e)
        }


def wait_and_download_result(api, task_info, output_dir):
    """第二阶段：等待结果并下载（不受速率限制）"""
    if not task_info['success']:
        return task_info
    
    song_id = task_info['song_id']
    song_index = task_info['song_index']
    task_id = task_info['task_id']
    data = task_info['data']
    start_time = task_info['submit_time']
    
    try:
        original_lyrics = data.get("original_lyrics", data.get("lyrics", ""))
        lyrics = data.get("lyrics", "")
        description = data.get("description", "")
        
        print_log(f"[歌曲 {song_id}] 等待生成完成...")
        
        # 等待完成（返回详细统计）
        wait_result = api.wait_for_completion(task_id, max_wait_time=600, check_interval=8)
        result = wait_result['result']
        
        # 处理返回结果
        tracks = []
        if isinstance(result, dict):
            if 'data' in result:
                tracks = result['data']
            elif 'sunoData' in result:
                tracks = result['sunoData']
            else:
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 0 and 'audioUrl' in value[0]:
                        tracks = value
                        break
        
        if not tracks:
            raise Exception("未找到音频轨道数据")
        
        # 下载阶段统计
        download_start = time.time()
        downloaded_files = []
        total_download_bytes = 0
        download_count = 0
        
        # 处理每个轨道
        for track_idx, track in enumerate(tracks):
            audio_url = track.get('audioUrl') or track.get('audio_url')
            audio_id = track.get('id')
            
            base_filename = f"{song_id}_{track_idx}"
            audio_path = os.path.join(output_dir, f"{base_filename}.mp3")
            lyrics_path = os.path.join(output_dir, f"{base_filename}_lyrics.json")
            
            # 下载音频
            if audio_url:
                download_result = api.download_file(audio_url, audio_path)
                if download_result['success']:
                    downloaded_files.append(audio_path)
                    total_download_bytes += download_result['bytes']
                    download_count += 1
            
            # 获取时间戳歌词
            timestamped_lyrics_data = None
            if audio_id:
                try:
                    lyrics_response = api.get_timestamped_lyrics(task_id, audio_id)
                    if lyrics_response.get('code') == 200:
                        timestamped_lyrics_data = lyrics_response.get('data')
                except Exception as e:
                    print_log(f"[歌曲 {song_id}] 轨道 {track_idx+1}: 获取歌词失败: {e}")
            
            # 保存歌词和元数据
            lyrics_content = {
                "song_id": song_id,
                "song_index": song_index,
                "track_index": track_idx,
                "original_lyrics": original_lyrics,
                "cleaned_lyrics": lyrics,
                "timestamped_lyrics": timestamped_lyrics_data,
                "style": description,
                "full_track_data": track
            }
            
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                json.dump(lyrics_content, f, ensure_ascii=False, indent=2)
            downloaded_files.append(lyrics_path)
        
        download_time = time.time() - download_start
        total_time = time.time() - start_time
        
        print_log(f"[歌曲 {song_id}] ✓ 完成! {len(tracks)} 个轨道，耗时 {total_time:.1f} 秒")
        
        final_result = {
            'song_id': song_id,
            'song_index': song_index,
            'task_id': task_id,
            'success': True,
            'tracks_count': len(tracks),
            'files': downloaded_files,
            'total_time': total_time,
            'submit_time': start_time,
            'wait_time': wait_result['wait_time'],
            'poll_count': wait_result['poll_count'],
            'avg_poll_time': wait_result['avg_poll_time'],
            'download_time': download_time,
            'download_bytes': total_download_bytes,
            'download_count': download_count,
            'avg_download_speed': total_download_bytes / download_time if download_time > 0 else 0
        }
        
        # 实时保存结果
        save_result_record(output_dir, final_result)
        return final_result
        
    except Exception as e:
        total_time = time.time() - start_time
        print_log(f"[歌曲 {song_id}] ✗ 处理失败: {e} (耗时 {total_time:.1f} 秒)")
        
        error_result = {
            'song_id': song_id,
            'song_index': song_index,
            'task_id': task_id,
            'success': False,
            'error': str(e),
            'total_time': total_time,
            'submit_time': start_time
        }
        
        # 实时保存结果
        save_result_record(output_dir, error_result)
        return error_result


def format_bytes(bytes_size):
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_speed(bytes_per_sec):
    """格式化速度"""
    return f"{format_bytes(bytes_per_sec)}/s"


def main():
    """主程序 - 两阶段并发处理"""
    input_file = "test_invalid.json"
    output_dir = "test_invalid"
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志
    global logger
    logger, log_file = setup_logging(output_dir)
    
    print_log("=" * 70)
    print_log("Suno API 批量生成 - 完善时间统计版本")
    print_log("策略: 快速提交(20次/10秒) + 并行等待 + 详细性能分析")
    print_log(f"日志文件: {log_file}")
    print_log("=" * 70)
    
    # 读取输入文件
    try:
        all_data = []
        if input_file.endswith('.jsonl'):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    # 尝试读取第一行来判断格式
                    first_line = f.readline().strip()
                    if first_line.startswith('['):
                        # 看起来像普通 JSON 数组
                        f.seek(0)
                        all_data = json.load(f)
                    else:
                        # 尝试逐行读取
                        f.seek(0)
                        for i, line in enumerate(f):
                            # # 限制读取第 11 到 20 条 (索引 10 到 19)
                            # if i < 10:
                            #     continue
                            # if i >= 20:
                            #     break
                            
                            line = line.strip()
                            if line:
                                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                # 如果上述解析失败，最后尝试一次整体读取
                print_log(f"注意: 按 JSONL 格式解析 {input_file} 失败，尝试作为普通 JSON 读取...")
                with open(input_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        
        # 移除 20000 条限制
        # if len(all_data) > 20000:
        #     print_log(f"数据量超过20000条，截取前20000条。")
        #     all_data = all_data[:20000]
            
    except FileNotFoundError:
        print_log(f"文件 {input_file} 未找到.")
        return
    except json.JSONDecodeError as e:
        print_log(f"JSON 解析错误: {e}")
        return
    
    # 创建输出目录
    # os.makedirs(output_dir, exist_ok=True)  # 已经移到上面了
    
    # 初始化 API
    api = SunoAPI(SUNO_API_KEY)
    
    print_log(f"\n准备生成 {len(all_data)} 首歌曲...")
    print_log(f"开始时间: {time.strftime('%H:%M:%S')}\n")
    
    overall_start_time = time.time()
    
    # ===== 第一阶段：批量提交 =====
    print_log("\n" + "=" * 70)
    print_log("第一阶段：批量提交 (从第 20181 首开始)")
    print_log("=" * 70 + "\n")
    
    submit_start_time = time.time()
    submitted_tasks = []
    total_request_time = 0
    
    # 调整速率限制：每10秒最多10个请求
    rate_limiter.max_requests = 10
    rate_limiter.time_window = 10
    rate_limiter.request_times.clear()
    print_log(f"速率限制: {rate_limiter.max_requests}次 / {rate_limiter.time_window}秒")
    
    # 只提交需要运行的任务
    tasks_to_run = []
    for i, data in enumerate(all_data, 1):
        tasks_to_run.append((i, data))
        
    print_log(f"需要提交的任务数: {len(tasks_to_run)}")

    # 使用线程池进行提交
    # 提交并发数受 rate_limiter 控制，可以设为 10
    with ThreadPoolExecutor(max_workers=10) as executor:
        submit_futures = {
            executor.submit(submit_generation_task, api, idx, data): idx
            for idx, data in tasks_to_run
        }
        
        with tqdm(total=len(tasks_to_run), desc="提交任务", unit="首") as pbar:
            for future in as_completed(submit_futures):
                result = future.result()
                submitted_tasks.append(result)
                if result.get('success') and 'request_time' in result:
                    total_request_time += result['request_time']
                pbar.update(1)
    
    submit_phase_time = time.time() - submit_start_time
    success_submits = sum(1 for t in submitted_tasks if t['success'])
    
    # 获取速率限制统计
    rate_limit_stats = rate_limiter.get_stats()
    
    print_log(f"\n提交阶段完成: {success_submits}/{len(tasks_to_run)} 成功")
    print_log(f"  总耗时: {submit_phase_time:.1f} 秒")
    print_log(f"  实际请求时间: {total_request_time:.2f} 秒")
    print_log(f"  速率限制等待: {rate_limit_stats['total_wait_time']:.2f} 秒 ({rate_limit_stats['wait_count']} 次)")
    if rate_limit_stats['wait_count'] > 0:
        print_log(f"  平均等待时间: {rate_limit_stats['avg_wait_time']:.2f} 秒/次")
    
    # ===== 第二阶段：并行等待和下载 =====
    print_log("\n" + "=" * 70)
    print_log("第二阶段：等待生成并下载")
    print_log("=" * 70 + "\n")
    
    wait_start_time = time.time()
    final_results = []
    
    # 使用更多线程并行等待（不受速率限制）
    with ThreadPoolExecutor(max_workers=20) as executor:
        download_futures = {
            executor.submit(wait_and_download_result, api, task, output_dir): task
            for task in submitted_tasks if task['success']
        }
        
        # 添加提交失败的任务到结果
        for task in submitted_tasks:
            if not task['success']:
                final_results.append(task)
        
        with tqdm(total=len(download_futures), desc="下载结果", unit="首") as pbar:
            for future in as_completed(download_futures):
                result = future.result()
                final_results.append(result)
                pbar.update(1)
    
    wait_phase_time = time.time() - wait_start_time
    
    # ===== 详细统计和报告 =====
    overall_time = time.time() - overall_start_time
    
    print_log("\n" + "=" * 70)
    print_log("批量生成完成 - 详细性能报告")
    print_log("=" * 70)
    
    success_count = sum(1 for r in final_results if r.get('success'))
    fail_count = len(final_results) - success_count
    total_tracks = sum(r.get('tracks_count', 0) for r in final_results if r.get('success'))
    
    successful_results = [r for r in final_results if r.get('success')]
    
    # 基本统计
    print_log(f"\n【基本统计】")
    print_log(f"  总歌曲数: {len(all_data)} 首")
    print_log(f"  成功: {success_count} 首")
    print_log(f"  失败: {fail_count} 首")
    print_log(f"  总轨道数: {total_tracks} 个")
    if success_count > 0:
        avg_tracks = total_tracks / success_count
        print_log(f"  平均每首轨道数: {avg_tracks:.2f} 个")
    
    # 时间统计
    print_log(f"\n【时间统计】")
    print_log(f"  ├── 提交阶段: {submit_phase_time:.1f} 秒")
    print_log(f"  │   ├── 实际请求时间: {total_request_time:.2f} 秒")
    print_log(f"  │   └── 速率限制等待: {rate_limit_stats['total_wait_time']:.2f} 秒")
    print_log(f"  ├── 生成等待阶段: {wait_phase_time:.1f} 秒")
    
    if successful_results:
        wait_times = [r.get('wait_time', 0) for r in successful_results if 'wait_time' in r]
        download_times = [r.get('download_time', 0) for r in successful_results if 'download_time' in r]
        
        if wait_times:
            avg_wait = sum(wait_times) / len(wait_times)
            min_wait = min(wait_times)
            max_wait = max(wait_times)
            print_log(f"  │   ├── 平均等待时间: {avg_wait:.1f} 秒/首")
            print_log(f"  │   ├── 最快: {min_wait:.1f} 秒")
            print_log(f"  │   └── 最慢: {max_wait:.1f} 秒")
        
        if download_times:
            total_download_time = sum(download_times)
            avg_download = total_download_time / len(download_times)
            print_log(f"  ├── 下载阶段: {total_download_time:.1f} 秒")
            print_log(f"  │   └── 平均下载时间: {avg_download:.2f} 秒/首")
    
    print_log(f"  └── 总耗时: {overall_time:.1f} 秒 ({overall_time/60:.1f} 分钟)")
    
    # 单首生成统计
    if successful_results:
        total_times = [r.get('total_time', 0) for r in successful_results if 'total_time' in r]
        if total_times:
            print_log(f"\n【单首生成统计】")
            avg_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            print_log(f"  平均每首总耗时: {avg_time:.1f} 秒")
            print_log(f"  最快生成: {min_time:.1f} 秒")
            print_log(f"  最慢生成: {max_time:.1f} 秒")
    
    # 下载统计
    total_download_bytes = sum(r.get('download_bytes', 0) for r in successful_results)
    total_download_count = sum(r.get('download_count', 0) for r in successful_results)
    
    if total_download_bytes > 0:
        print_log(f"\n【下载统计】")
        print_log(f"  总下载量: {format_bytes(total_download_bytes)}")
        print_log(f"  下载文件数: {total_download_count} 个")
        print_log(f"  平均文件大小: {format_bytes(total_download_bytes / total_download_count)}")
        
        download_speeds = [r.get('avg_download_speed', 0) for r in successful_results if r.get('avg_download_speed', 0) > 0]
        if download_speeds:
            avg_speed = sum(download_speeds) / len(download_speeds)
            print_log(f"  平均下载速度: {format_speed(avg_speed)}")
    
    # 轮询统计
    poll_counts = [r.get('poll_count', 0) for r in successful_results if 'poll_count' in r]
    if poll_counts:
        total_polls = sum(poll_counts)
        avg_polls = total_polls / len(poll_counts)
        print_log(f"\n【轮询统计】")
        print_log(f"  总轮询次数: {total_polls} 次")
        print_log(f"  平均每首轮询: {avg_polls:.1f} 次")
    
    # 效率分析
    print_log(f"\n【效率分析】")
    if success_count > 0:
        throughput = success_count / (overall_time / 60)
        print_log(f"  实际吞吐率: {throughput:.2f} 首/分钟")
        
        # 理论最快时间（假设无速率限制）
        if wait_times:
            ideal_time = submit_phase_time - rate_limit_stats['total_wait_time'] + max(wait_times)
            efficiency = (ideal_time / overall_time) * 100
            print_log(f"  理论最快时间: {ideal_time:.1f} 秒")
            print_log(f"  并发效率: {efficiency:.1f}%")
    
    # 显示失败的歌曲
    if fail_count > 0:
        print_log("\n" + "=" * 70)
        print_log("失败的歌曲列表")
        print_log("=" * 70)
        for r in sorted(final_results, key=lambda x: x.get('song_index', 0)):
            if not r.get('success'):
                song_id = r.get('song_id', r.get('song_index', 'Unknown'))
                print_log(f"  [{song_id}] {r.get('error', 'Unknown error')}")
    
    print_log("\n" + "=" * 70)
    print_log(f"所有文件已保存到: {os.path.abspath(output_dir)}")
    print_log("=" * 70)


if __name__ == '__main__':
    main()