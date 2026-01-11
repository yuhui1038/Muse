#!/usr/bin/env python3
"""
使用 Mureka API 批量生成歌曲的脚本
处理 cleaned_data_no_desc.json 文件中的前 100 首歌
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional
from pathlib import Path

# API 配置
API_URL = "https://api.mureka.cn/v1/song/generate"
QUERY_API_URL = "https://api.mureka.cn/v1/song/query"
API_KEY_ENV = "MUREKA_API_KEY"
MODEL = "mureka-o2"

# 配置参数
MAX_SONGS = 100
RETRY_TIMES = 3
RETRY_DELAY = 2  # 秒
REQUEST_DELAY = 60  # 请求之间的延迟（秒）- 设置为 60 秒（1 分钟）避免频率限制
RATE_LIMIT_DELAY = 60  # 遇到 429 错误时的等待时间（秒）
QUERY_INTERVAL = 10  # 查询任务状态的间隔（秒）
MAX_QUERY_TIME = 3600  # 最大查询时间（秒），1 小时

def load_songs(json_file: str, max_count: int = MAX_SONGS) -> List[Dict]:
    """从 JSON 文件加载歌曲数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 只取前 max_count 首
    return data[:max_count]

def is_song_processed(output_file: Path) -> bool:
    """
    检查歌曲是否已经处理过（包括任务已完成）
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        如果文件存在且包含有效的 API 响应，返回 True
    """
    if not output_file.exists():
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 检查是否包含 api_response 字段
            if 'api_response' in data and data['api_response']:
                status = data['api_response'].get('status', '')
                # 如果状态是 succeeded、failed、timeouted 或 cancelled，视为已处理
                if status in ['succeeded', 'failed', 'timeouted', 'cancelled']:
                    return True
                # 如果状态是 preparing、queued、running、streaming、reviewing，也视为已处理（任务已创建）
                if status in ['preparing', 'queued', 'running', 'streaming', 'reviewing']:
                    return True
    except (json.JSONDecodeError, KeyError, IOError):
        # 文件损坏或格式不正确，视为未处理
        return False
    
    return False

def load_processed_song(output_file: Path) -> Optional[Dict]:
    """
    从已存在的文件中加载已处理的歌曲结果
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        已处理的歌曲数据，如果加载失败返回 None
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('api_response')
    except (json.JSONDecodeError, KeyError, IOError):
        return None

def query_task_status(task_id: str, api_key: str) -> Optional[Dict]:
    """
    查询任务状态
    
    Args:
        task_id: 任务 ID
        api_key: API 密钥
    
    Returns:
        任务状态数据，失败返回 None
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
        print(f"    查询任务状态失败: {str(e)}")
        return None

def wait_for_task_completion(task_id: str, api_key: str) -> Optional[Dict]:
    """
    等待任务完成并返回最终结果
    
    Args:
        task_id: 任务 ID
        api_key: API 密钥
    
    Returns:
        任务完成后的完整数据，失败返回 None
    """
    start_time = time.time()
    last_status = None
    
    print(f"  等待任务完成 (任务 ID: {task_id})...")
    
    while time.time() - start_time < MAX_QUERY_TIME:
        result = query_task_status(task_id, api_key)
        
        if not result:
            time.sleep(QUERY_INTERVAL)
            continue
        
        status = result.get('status', '')
        
        # 如果状态改变，打印新状态
        if status != last_status:
            print(f"    状态: {status}")
            last_status = status
        
        # 任务完成（成功或失败）
        if status in ['succeeded', 'failed', 'timeouted', 'cancelled']:
            if status == 'succeeded':
                print(f"  ✓ 任务完成！")
                if 'choices' in result and result['choices']:
                    print(f"    找到 {len(result['choices'])} 个生成的歌曲")
            else:
                print(f"  ✗ 任务失败: {status}")
                if 'failed_reason' in result:
                    print(f"    失败原因: {result['failed_reason']}")
            return result
        
        # 任务还在处理中，继续等待
        time.sleep(QUERY_INTERVAL)
    
    print(f"  ⚠ 查询超时（超过 {MAX_QUERY_TIME} 秒）")
    return None

def generate_song(lyrics: str, prompt: str, api_key: str) -> Optional[Dict]:
    """
    调用 API 生成单首歌曲（串行处理，确保并发为 1）
    
    Args:
        lyrics: 歌词内容
        prompt: 提示词（对应 description）
        api_key: API 密钥
    
    Returns:
        API 响应数据，失败返回 None
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
            
            # 检查是否是 429 错误（频率限制）
            if response.status_code == 429:
                # 尝试从响应头获取 Retry-After 时间
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = RATE_LIMIT_DELAY
                
                print(f"  尝试 {attempt + 1}/{RETRY_TIMES} 失败: 429 Too Many Requests")
                print(f"  等待 {wait_time} 秒后重试...")
                if attempt < RETRY_TIMES - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  所有重试均失败")
                    return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                # 429 错误已在上面处理
                continue
            print(f"  尝试 {attempt + 1}/{RETRY_TIMES} 失败: {str(e)}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  所有重试均失败")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  尝试 {attempt + 1}/{RETRY_TIMES} 失败: {str(e)}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  所有重试均失败")
                return None
    
    return None

def main():
    """主函数"""
    # 检查 API 密钥
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"错误: 请设置环境变量 {API_KEY_ENV}")
        print(f"例如: export {API_KEY_ENV}=your_api_key")
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

