# -*- coding: utf-8 -*-
"""
Suno API Batch Generation - V4.5 Special Edition
Supported models: V4_5 (default), V4_5PLUS, V4_5ALL
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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUNO_API_KEY


# Configure logging
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f"run_log_v4_5_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create logger
    logger = logging.getLogger('SunoBatchV4_5')
    logger.setLevel(logging.INFO)
    
    # Clear old handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_file

# Global logger
logger = logging.getLogger('SunoBatchV4_5')

# Replace print with logger.info
def print_log(msg):
    logger.info(msg)


class SunoAPI:
    """Simplified Suno API client"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.sunoapi.org/api/v1'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Configure retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,  # Maximum retry count
            backoff_factor=1,  # Retry interval (1s, 2s, 4s, 8s...)
            status_forcelist=[500, 502, 503, 504],  # Status codes that need retry
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]  # Allowed retry methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def generate_music(self, prompt, model='V4_5', vocalGender=None, **options):
        """Generate music"""
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
            
            # Check HTTP errors
            response.raise_for_status()
            
            # Try to parse JSON
            try:
                result = response.json()
            except json.JSONDecodeError:
                raise Exception(f"API returned non-JSON response: {response.text[:200]}")
                
            if result.get('code') != 200:
                raise Exception(f"Generation failed: {result.get('msg', result)}")
            
            return result['data']['taskId']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request exception: {str(e)}")
    
    def get_task_status(self, task_id):
        """Get task status"""
        try:
            response = self.session.get(
                f'{self.base_url}/generate/record-info?taskId={task_id}',
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            # Status query failure should not crash the program, return empty dict or throw specific exception
            # print_log(f"Failed to get status: {e}")
            raise e
    
    def get_timestamped_lyrics(self, task_id, audio_id):
        """Get timestamped lyrics"""
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
            return {}  # Lyrics retrieval failure is non-fatal error
    
    def wait_for_completion(self, task_id, max_wait_time=600, check_interval=5):
        """Wait for task completion, return result and polling statistics"""
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
                    raise Exception(f"Task failed: {status.get('errorMessage')}")
                
                time.sleep(check_interval)
            except Exception as e:
                if time.time() - start_time >= max_wait_time:
                    raise
                time.sleep(check_interval)
        
        raise Exception('Task timeout')
    
    def download_file(self, url, save_path):
        """Download file to local, return download statistics"""
        try:
            start_time = time.time()
            downloaded_bytes = 0
            
            # Use session to download
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
            print_log(f"Download failed {url}: {e}")
            return {'success': False, 'error': str(e)}


# Result record lock
result_lock = Lock()

def save_result_record(output_dir, record):
    """Save single result to CSV in real-time"""
    file_path = os.path.join(output_dir, "generation_results.csv")
    file_exists = os.path.isfile(file_path)
    
    # Only record key information
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
    """Improved rate limiter (with statistics)
    
    Precise control: maximum 8 requests per 10 seconds
    Uses sliding window algorithm to ensure no more than 8 requests in any 10-second time window
    """
    
    def __init__(self, max_requests=5, time_window=10):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = Lock()
        self.semaphore = Semaphore(max_requests)
        
        # Statistics
        self.total_wait_time = 0
        self.wait_count = 0
        self.total_requests = 0
    
    def acquire(self):
        """Acquire request permission"""
        with self.lock:
            now = time.time()
            
            # Clean expired request records
            while self.request_times and now - self.request_times[0] >= self.time_window:
                self.request_times.popleft()
            
            # If limit reached, calculate wait time needed
            wait_time = 0
            if len(self.request_times) >= self.max_requests:
                oldest_request = self.request_times[0]
                wait_time = self.time_window - (now - oldest_request) + 0.05  # Add buffer
                
                if wait_time > 0:
                    print_log(f"  [Rate Limit] Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    
                    # Record wait time
                    self.total_wait_time += wait_time
                    self.wait_count += 1
                    
                    # Re-clean
                    now = time.time()
                    while self.request_times and now - self.request_times[0] >= self.time_window:
                        self.request_times.popleft()
            
            # Record this request time
            self.request_times.append(time.time())
            self.total_requests += 1
            
    def get_current_rate(self):
        """Get current rate (number of requests in last 10 seconds)"""
        with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] >= self.time_window:
                self.request_times.popleft()
            return len(self.request_times)
    
    def get_stats(self):
        """Get statistics"""
        with self.lock:
            return {
                'total_requests': self.total_requests,
                'total_wait_time': self.total_wait_time,
                'wait_count': self.wait_count,
                'avg_wait_time': self.total_wait_time / self.wait_count if self.wait_count > 0 else 0
            }


# Global rate limiter (5 requests per 10 seconds)
rate_limiter = ImprovedRateLimiter(max_requests=5, time_window=10)


def submit_generation_task(api, song_index, data):
    """Phase 1: Submit generation task (rate limited)"""
    # Use sunov4_5_000001 format
    song_id = data.get("id", f"sunov4_5_{song_index:06d}")
    
    try:
        description = data.get("description", "")
        lyrics = data.get("lyrics", "")
        vocal_gender = data.get("vocalGender")
        
        print_log(f"[Song {song_id}] Submitting task... (current rate: {rate_limiter.get_current_rate()}/5)")
        
        # Record request start time
        request_start = time.time()
        
        # Rate limiting
        rate_limiter.acquire()
        
        # Submit task
        submit_start = time.time()
        task_id = api.generate_music(
            prompt=lyrics,
            style=description,
            title=f"Song_{song_id}",
            model='V4_5', # Explicitly specify V4.5 model
            customMode=True,
            instrumental=False,
            vocalGender=vocal_gender
        )
        request_time = time.time() - submit_start
        
        print_log(f"[Song {song_id}] ✓ Task submitted, ID: {task_id}")
        
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
        print_log(f"[Song {song_id}] ✗ Submission failed: {e}")
        # If submission fails, also record it (even though not at download stage yet)
        return {
            'song_id': song_id,
            'song_index': song_index,
            'success': False,
            'error': str(e)
        }


def wait_and_download_result(api, task_info, output_dir):
    """Phase 2: Wait for result and download (not rate limited)"""
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
        
        print_log(f"[Song {song_id}] Waiting for generation to complete...")
        
        # Wait for completion (returns detailed statistics)
        wait_result = api.wait_for_completion(task_id, max_wait_time=600, check_interval=8)
        result = wait_result['result']
        
        # Process returned result
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
            raise Exception("Audio track data not found")
        
        # Download phase statistics
        download_start = time.time()
        downloaded_files = []
        total_download_bytes = 0
        download_count = 0
        
        # Process each track
        for track_idx, track in enumerate(tracks):
            audio_url = track.get('audioUrl') or track.get('audio_url')
            audio_id = track.get('id')
            
            base_filename = f"{song_id}_{track_idx}"
            audio_path = os.path.join(output_dir, f"{base_filename}.mp3")
            lyrics_path = os.path.join(output_dir, f"{base_filename}_lyrics.json")
            
            # Download audio
            if audio_url:
                download_result = api.download_file(audio_url, audio_path)
                if download_result['success']:
                    downloaded_files.append(audio_path)
                    total_download_bytes += download_result['bytes']
                    download_count += 1
            
            # Get timestamped lyrics
            timestamped_lyrics_data = None
            if audio_id:
                try:
                    lyrics_response = api.get_timestamped_lyrics(task_id, audio_id)
                    if lyrics_response.get('code') == 200:
                        timestamped_lyrics_data = lyrics_response.get('data')
                except Exception as e:
                    print_log(f"[Song {song_id}] Track {track_idx+1}: Failed to get lyrics: {e}")
            
            # Save lyrics and metadata
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
        
        print_log(f"[Song {song_id}] ✓ Complete! {len(tracks)} tracks, took {total_time:.1f} seconds")
        
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
        
        # Save result in real-time
        save_result_record(output_dir, final_result)
        return final_result
        
    except Exception as e:
        total_time = time.time() - start_time
        print_log(f"[Song {song_id}] ✗ Processing failed: {e} (took {total_time:.1f} seconds)")
        
        error_result = {
            'song_id': song_id,
            'song_index': song_index,
            'task_id': task_id,
            'success': False,
            'error': str(e),
            'total_time': total_time,
            'submit_time': start_time
        }
        
        # Save result in real-time
        save_result_record(output_dir, error_result)
        return error_result


def format_bytes(bytes_size):
    """Format byte size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_speed(bytes_per_sec):
    """Format speed"""
    return f"{format_bytes(bytes_per_sec)}/s"


def main():
    """Main program - two-phase concurrent processing"""
    input_file = "cleaned_data_truncated.json"
    output_dir = "sunov4_5_truncated"
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logging
    global logger
    logger, log_file = setup_logging(output_dir)
    
    print_log("=" * 70)
    print_log("Suno API Batch Generation - V4.5 Special Edition")
    print_log("Strategy: Fast submission (5 requests/10s) + Parallel waiting + Detailed performance analysis")
    print_log(f"Log file: {log_file}")
    print_log("=" * 70)
    
    # Read input file
    try:
        all_data = []
        if input_file.endswith('.jsonl'):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    # Try reading first line to determine format
                    first_line = f.readline().strip()
                    if first_line.startswith('['):
                        # Looks like regular JSON array
                        f.seek(0)
                        all_data = json.load(f)
                    else:
                        # Try reading line by line
                        f.seek(0)
                        for i, line in enumerate(f):
                            line = line.strip()
                            if line:
                                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                # If above parsing fails, try one final read as regular JSON
                print_log(f"Note: Failed to parse {input_file} as JSONL format, trying as regular JSON...")
                with open(input_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        
    except FileNotFoundError:
        print_log(f"File {input_file} not found.")
        return
    except json.JSONDecodeError as e:
        print_log(f"JSON parsing error: {e}")
        return
    
    # Initialize API
    api = SunoAPI(SUNO_API_KEY)
    
    print_log(f"\nPreparing to generate {len(all_data)} songs...")
    print_log(f"Start time: {time.strftime('%H:%M:%S')}\n")
    
    overall_start_time = time.time()
    
    # ===== Phase 1: Batch Submission =====
    print_log("\n" + "=" * 70)
    print_log("Phase 1: Batch Submission")
    print_log("=" * 70 + "\n")
    
    submit_start_time = time.time()
    submitted_tasks = []
    total_request_time = 0
    
    # Adjust rate limit: maximum 5 requests per 10 seconds
    rate_limiter.max_requests = 5
    rate_limiter.time_window = 10
    rate_limiter.request_times.clear()
    print_log(f"Rate limit: {rate_limiter.max_requests} requests / {rate_limiter.time_window} seconds")
    
    # Only submit tasks that need to run
    tasks_to_run = []
    for i, data in enumerate(all_data, 1):
        tasks_to_run.append((i, data))
        
    print_log(f"Number of tasks to submit: {len(tasks_to_run)}")

    # Use thread pool for submission
    # Submission concurrency is controlled by rate_limiter, can be set to 5
    with ThreadPoolExecutor(max_workers=5) as executor:
        submit_futures = {
            executor.submit(submit_generation_task, api, idx, data): idx
            for idx, data in tasks_to_run
        }
        
        with tqdm(total=len(tasks_to_run), desc="Submitting tasks", unit="song") as pbar:
            for future in as_completed(submit_futures):
                result = future.result()
                submitted_tasks.append(result)
                if result.get('success') and 'request_time' in result:
                    total_request_time += result['request_time']
                pbar.update(1)
    
    submit_phase_time = time.time() - submit_start_time
    success_submits = sum(1 for t in submitted_tasks if t['success'])
    
    # Get rate limit statistics
    rate_limit_stats = rate_limiter.get_stats()
    
    print_log(f"\nSubmission phase complete: {success_submits}/{len(tasks_to_run)} successful")
    print_log(f"  Total time: {submit_phase_time:.1f} seconds")
    print_log(f"  Actual request time: {total_request_time:.2f} seconds")
    print_log(f"  Rate limit waiting: {rate_limit_stats['total_wait_time']:.2f} seconds ({rate_limit_stats['wait_count']} times)")
    if rate_limit_stats['wait_count'] > 0:
        print_log(f"  Average wait time: {rate_limit_stats['avg_wait_time']:.2f} seconds/time")
    
    # ===== Phase 2: Parallel Waiting and Download =====
    print_log("\n" + "=" * 70)
    print_log("Phase 2: Wait for Generation and Download")
    print_log("=" * 70 + "\n")
    
    wait_start_time = time.time()
    final_results = []
    
    # Use more threads for parallel waiting (not rate limited)
    with ThreadPoolExecutor(max_workers=20) as executor:
        download_futures = {
            executor.submit(wait_and_download_result, api, task, output_dir): task
            for task in submitted_tasks if task['success']
        }
        
        # Add failed submission tasks to results
        for task in submitted_tasks:
            if not task['success']:
                final_results.append(task)
        
        with tqdm(total=len(download_futures), desc="Downloading results", unit="song") as pbar:
            for future in as_completed(download_futures):
                result = future.result()
                final_results.append(result)
                pbar.update(1)
    
    wait_phase_time = time.time() - wait_start_time
    
    # ===== Detailed Statistics and Report =====
    overall_time = time.time() - overall_start_time
    
    print_log("\n" + "=" * 70)
    print_log("Batch Generation Complete - Detailed Performance Report")
    print_log("=" * 70)
    
    success_count = sum(1 for r in final_results if r.get('success'))
    fail_count = len(final_results) - success_count
    total_tracks = sum(r.get('tracks_count', 0) for r in final_results if r.get('success'))
    
    successful_results = [r for r in final_results if r.get('success')]
    
    # Basic Statistics
    print_log(f"\n[Basic Statistics]")
    print_log(f"  Total songs: {len(all_data)}")
    print_log(f"  Successful: {success_count}")
    print_log(f"  Failed: {fail_count}")
    print_log(f"  Total tracks: {total_tracks}")
    if success_count > 0:
        avg_tracks = total_tracks / success_count
        print_log(f"  Average tracks per song: {avg_tracks:.2f}")
    
    # Time Statistics
    print_log(f"\n[Time Statistics]")
    print_log(f"  ├── Submission phase: {submit_phase_time:.1f} seconds")
    print_log(f"  │   ├── Actual request time: {total_request_time:.2f} seconds")
    print_log(f"  │   └── Rate limit waiting: {rate_limit_stats['total_wait_time']:.2f} seconds")
    print_log(f"  ├── Generation waiting phase: {wait_phase_time:.1f} seconds")
    
    if successful_results:
        wait_times = [r.get('wait_time', 0) for r in successful_results if 'wait_time' in r]
        download_times = [r.get('download_time', 0) for r in successful_results if 'download_time' in r]
        
        if wait_times:
            avg_wait = sum(wait_times) / len(wait_times)
            min_wait = min(wait_times)
            max_wait = max(wait_times)
            print_log(f"  │   ├── Average wait time: {avg_wait:.1f} seconds/song")
            print_log(f"  │   ├── Fastest: {min_wait:.1f} seconds")
            print_log(f"  │   └── Slowest: {max_wait:.1f} seconds")
        
        if download_times:
            total_download_time = sum(download_times)
            avg_download = total_download_time / len(download_times)
            print_log(f"  ├── Download phase: {total_download_time:.1f} seconds")
            print_log(f"  │   └── Average download time: {avg_download:.2f} seconds/song")
    
    print_log(f"  └── Total time: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")
    
    # Single Song Generation Statistics
    if successful_results:
        total_times = [r.get('total_time', 0) for r in successful_results if 'total_time' in r]
        if total_times:
            print_log(f"\n[Single Song Generation Statistics]")
            avg_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            print_log(f"  Average total time per song: {avg_time:.1f} seconds")
            print_log(f"  Fastest generation: {min_time:.1f} seconds")
            print_log(f"  Slowest generation: {max_time:.1f} seconds")
    
    # Download Statistics
    total_download_bytes = sum(r.get('download_bytes', 0) for r in successful_results)
    total_download_count = sum(r.get('download_count', 0) for r in successful_results)
    
    if total_download_bytes > 0:
        print_log(f"\n[Download Statistics]")
        print_log(f"  Total download: {format_bytes(total_download_bytes)}")
        print_log(f"  Number of files: {total_download_count}")
        print_log(f"  Average file size: {format_bytes(total_download_bytes / total_download_count)}")
        
        download_speeds = [r.get('avg_download_speed', 0) for r in successful_results if r.get('avg_download_speed', 0) > 0]
        if download_speeds:
            avg_speed = sum(download_speeds) / len(download_speeds)
            print_log(f"  Average download speed: {format_speed(avg_speed)}")
    
    # Polling Statistics
    poll_counts = [r.get('poll_count', 0) for r in successful_results if 'poll_count' in r]
    if poll_counts:
        total_polls = sum(poll_counts)
        avg_polls = total_polls / len(poll_counts)
        print_log(f"\n[Polling Statistics]")
        print_log(f"  Total polling count: {total_polls}")
        print_log(f"  Average polling per song: {avg_polls:.1f}")
    
    # Efficiency Analysis
    print_log(f"\n[Efficiency Analysis]")
    if success_count > 0:
        throughput = success_count / (overall_time / 60)
        print_log(f"  Actual throughput: {throughput:.2f} songs/minute")
        
        # Theoretical fastest time (assuming no rate limit)
        if wait_times:
            ideal_time = submit_phase_time - rate_limit_stats['total_wait_time'] + max(wait_times)
            efficiency = (ideal_time / overall_time) * 100
            print_log(f"  Theoretical fastest time: {ideal_time:.1f} seconds")
            print_log(f"  Concurrency efficiency: {efficiency:.1f}%")
    
    # Show failed songs
    if fail_count > 0:
        print_log("\n" + "=" * 70)
        print_log("Failed Songs List")
        print_log("=" * 70)
        for r in sorted(final_results, key=lambda x: x.get('song_index', 0)):
            if not r.get('success'):
                song_id = r.get('song_id', r.get('song_index', 'Unknown'))
                print_log(f"  [{song_id}] {r.get('error', 'Unknown error')}")
    
    print_log("\n" + "=" * 70)
    print_log(f"All files saved to: {os.path.abspath(output_dir)}")
    print_log("=" * 70)


if __name__ == '__main__':
    main()

