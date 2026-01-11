#!/usr/bin/env python3
"""
Script for batch generating songs using Mureka API
Processes the first 100 songs in cleaned_data_no_desc.json file
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional
from pathlib import Path

# API Configuration
API_URL = "https://api.mureka.cn/v1/song/generate"
QUERY_API_URL = "https://api.mureka.cn/v1/song/query"
API_KEY_ENV = "MUREKA_API_KEY"
MODEL = "mureka-o2"

# Configuration Parameters
MAX_SONGS = 100
RETRY_TIMES = 3
RETRY_DELAY = 2  # seconds
REQUEST_DELAY = 60  # Delay between requests (seconds) - set to 60 seconds (1 minute) to avoid rate limiting
RATE_LIMIT_DELAY = 60  # Wait time when encountering 429 error (seconds)
QUERY_INTERVAL = 10  # Interval for querying task status (seconds)
MAX_QUERY_TIME = 3600  # Maximum query time (seconds), 1 hour

def load_songs(json_file: str, max_count: int = MAX_SONGS) -> List[Dict]:
    """Load song data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Only take first max_count songs
    return data[:max_count]

def is_song_processed(output_file: Path) -> bool:
    """
    Check if song has been processed (including completed tasks)
    
    Args:
        output_file: Output file path
    
    Returns:
        True if file exists and contains valid API response
    """
    if not output_file.exists():
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Check if contains api_response field
            if 'api_response' in data and data['api_response']:
                status = data['api_response'].get('status', '')
                # If status is succeeded, failed, timeouted or cancelled, consider as processed
                if status in ['succeeded', 'failed', 'timeouted', 'cancelled']:
                    return True
                # If status is preparing, queued, running, streaming, reviewing, also consider as processed (task created)
                if status in ['preparing', 'queued', 'running', 'streaming', 'reviewing']:
                    return True
    except (json.JSONDecodeError, KeyError, IOError):
        # File corrupted or format incorrect, consider as not processed
        return False
    
    return False

def load_processed_song(output_file: Path) -> Optional[Dict]:
    """
    Load processed song results from existing file
    
    Args:
        output_file: Output file path
    
    Returns:
        Processed song data, returns None if loading fails
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('api_response')
    except (json.JSONDecodeError, KeyError, IOError):
        return None

def query_task_status(task_id: str, api_key: str) -> Optional[Dict]:
    """
    Query task status
    
    Args:
        task_id: Task ID
        api_key: API key
    
    Returns:
        Task status data, returns None on failure
    """
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    url = f"{QUERY_API_URL}/{task_id}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    Failed to query task status: {str(e)}")
        return None

def wait_for_task_completion(task_id: str, api_key: str) -> Optional[Dict]:
    """
    Wait for task completion and return final result
    
    Args:
        task_id: Task ID
        api_key: API key
    
    Returns:
        Complete data after task completion, returns None on failure
    """
    start_time = time.time()
    last_status = None
    
    print(f"  Waiting for task completion (Task ID: {task_id})...")
    
    while time.time() - start_time < MAX_QUERY_TIME:
        result = query_task_status(task_id, api_key)
        
        if not result:
            time.sleep(QUERY_INTERVAL)
            continue
        
        status = result.get('status', '')
        
        # If status changed, print new status
        if status != last_status:
            print(f"    Status: {status}")
            last_status = status
        
        # Task completed (success or failure)
        if status in ['succeeded', 'failed', 'timeouted', 'cancelled']:
            if status == 'succeeded':
                print(f"  ✓ Task completed!")
                if 'choices' in result and result['choices']:
                    print(f"    Found {len(result['choices'])} generated songs")
            else:
                print(f"  ✗ Task failed: {status}")
                if 'failed_reason' in result:
                    print(f"    Failure reason: {result['failed_reason']}")
            return result
        
        # Task still processing, continue waiting
        time.sleep(QUERY_INTERVAL)
    
    print(f"  ⚠ Query timeout (exceeded {MAX_QUERY_TIME} seconds)")
    return None

def generate_song(lyrics: str, prompt: str, api_key: str) -> Optional[Dict]:
    """
    Call API to generate a single song (serial processing, ensuring concurrency = 1)
    
    Args:
        lyrics: Lyrics content
        prompt: Prompt (corresponds to description)
        api_key: API key
    
    Returns:
        API response data, returns None on failure
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "lyrics": lyrics,
        "model": MODEL,
        "prompt": prompt
    }
    
    for attempt in range(RETRY_TIMES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            
            # Check if it's a 429 error (rate limit)
            if response.status_code == 429:
                # Try to get Retry-After time from response header
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = RATE_LIMIT_DELAY
                
                print(f"  Attempt {attempt + 1}/{RETRY_TIMES} failed: 429 Too Many Requests")
                print(f"  Waiting {wait_time} seconds before retry...")
                if attempt < RETRY_TIMES - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  All retries failed")
                    return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                # 429 error already handled above
                continue
            print(f"  Attempt {attempt + 1}/{RETRY_TIMES} failed: {str(e)}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All retries failed")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1}/{RETRY_TIMES} failed: {str(e)}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All retries failed")
                return None
    
    return None

def main():
    """主函数"""
    # 检查 API 密钥
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"Error: Please set environment variable {API_KEY_ENV}")
        print(f"Example: export {API_KEY_ENV}=your_api_key")
        return
    
    # 加载歌曲数据
    json_file = "cleaned_data_no_desc.json"
    if not os.path.exists(json_file):
        print(f"错误: 找不到文件 {json_file}")
        return
    
    print(f"正在加载歌曲数据从 {json_file}...")
    songs = load_songs(json_file, MAX_SONGS)
    print(f"已加载 {len(songs)} 首歌曲")
    
    # 创建输出目录
    output_dir = Path("generated_songs")
    output_dir.mkdir(exist_ok=True)
    
    # 保存结果的列表
    results = []
    
    # 处理每首歌曲（串行处理，确保并发为 1）
    for idx, song in enumerate(songs, 1):
        print(f"\n[{idx}/{len(songs)}] 正在处理歌曲...")
        print(f"  Description: {song.get('description', 'N/A')[:50]}...")
        
        # 检查输出文件路径
        output_file = output_dir / f"song_{idx:03d}.json"
        
        # 检查是否已经处理过
        if is_song_processed(output_file):
            print(f"  ⊙ 已处理，跳过")
            # 从文件加载已处理的结果
            existing_result = load_processed_song(output_file)
            results.append({
                "index": idx,
                "status": "already_processed",
                "output_file": str(output_file)
            })
            # 已处理的歌曲不需要延迟
            continue
        
        lyrics = song.get('lyrics', '')
        description = song.get('description', '')
        
        if not lyrics or not description:
            print(f"  跳过: 缺少 lyrics 或 description")
            results.append({
                "index": idx,
                "status": "skipped",
                "reason": "missing data"
            })
            continue
        
        # 调用 API（串行执行，确保并发为 1）
        result = generate_song(lyrics, description, api_key)
        
        if result:
            task_id = result.get('id')
            initial_status = result.get('status', '')
            print(f"  ✓ 任务已创建 (ID: {task_id}, 状态: {initial_status})")
            
            # 如果任务状态不是最终状态，等待任务完成
            if initial_status not in ['succeeded', 'failed', 'timeouted', 'cancelled']:
                final_result = wait_for_task_completion(task_id, api_key)
                if final_result:
                    result = final_result
                else:
                    # 查询超时，使用初始结果
                    print(f"  ⚠ 使用初始结果（查询超时）")
            
            # 保存单个结果（包含最终状态和 choices）
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "index": idx,
                    "original_data": song,
                    "api_response": result,
                    "task_id": task_id
                }, f, ensure_ascii=False, indent=2)
            
            # 检查是否成功完成
            final_status = result.get('status', '')
            if final_status == 'succeeded':
                results.append({
                    "index": idx,
                    "status": "success",
                    "output_file": str(output_file),
                    "task_id": task_id,
                    "has_audio": 'choices' in result and len(result.get('choices', [])) > 0
                })
            elif final_status in ['failed', 'timeouted', 'cancelled']:
                results.append({
                    "index": idx,
                    "status": "failed",
                    "output_file": str(output_file),
                    "task_id": task_id,
                    "failed_reason": result.get('failed_reason', final_status)
                })
            else:
                # 任务还在处理中
                results.append({
                    "index": idx,
                    "status": "processing",
                    "output_file": str(output_file),
                    "task_id": task_id,
                    "current_status": final_status
                })
        else:
            print(f"  ✗ 生成失败")
            # 保存失败信息，包括原始数据
            error_file = output_dir / f"song_{idx:03d}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "index": idx,
                    "original_data": song,
                    "error": "API调用失败，generate_song返回None",
                    "timestamp": time.time()
                }, f, ensure_ascii=False, indent=2)
            
            results.append({
                "index": idx,
                "status": "failed",
                "error_file": str(error_file),
                "reason": "API调用失败"
            })
        
        # 请求之间的延迟，避免频率限制（确保并发为 1）
        if idx < len(songs):
            print(f"  等待 {REQUEST_DELAY} 秒后处理下一首...")
            time.sleep(REQUEST_DELAY)
    
    # 保存汇总结果
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(songs),
            "success": sum(1 for r in results if r.get("status") == "success"),
            "processing": sum(1 for r in results if r.get("status") == "processing"),
            "already_processed": sum(1 for r in results if r.get("status") == "already_processed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"总计: {len(songs)} 首")
    print(f"成功完成: {sum(1 for r in results if r.get('status') == 'success')} 首")
    print(f"处理中: {sum(1 for r in results if r.get('status') == 'processing')} 首")
    print(f"已处理: {sum(1 for r in results if r.get('status') == 'already_processed')} 首")
    print(f"失败: {sum(1 for r in results if r.get('status') == 'failed')} 首")
    print(f"跳过: {sum(1 for r in results if r.get('status') == 'skipped')} 首")
    print(f"结果保存在: {output_dir}/")
    print(f"汇总文件: {summary_file}")
    print(f"\n提示: 如果任务还在处理中，可以稍后重新运行脚本查询状态")

if __name__ == "__main__":
    main()

