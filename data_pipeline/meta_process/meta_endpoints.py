import json
import webrtcvad
import collections
from tqdm import tqdm
from my_tool import dup_remove
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed

def _frame_generator(frame_duration_ms, audio, sample_rate):
    """音频分帧"""
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_duration_ms / 1000.0) * bytes_per_sample
    offset = 0
    timestamp = 0.0
    frame_duration = frame_duration_ms / 1000.0
    while offset + frame_size < len(audio):
        yield audio[offset:offset + frame_size], timestamp
        timestamp += frame_duration
        offset += frame_size

def _vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """根据webrtcvad合并连续的人声段"""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    speech_segments = []

    for frame_bytes, timestamp in frames:
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame_bytes, timestamp, is_speech))
            num_voiced = len([f for f in ring_buffer if f[2]])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time = ring_buffer[0][1]
                ring_buffer.clear()
        else:
            ring_buffer.append((frame_bytes, timestamp, is_speech))
            num_unvoiced = len([f for f in ring_buffer if not f[2]])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = timestamp + (frame_duration_ms / 1000.0)
                speech_segments.append((start_time, end_time))
                triggered = False
                ring_buffer.clear()

    # 如果最后还在说话状态，则闭合最后一个段
    if triggered:
        end_time = timestamp + (frame_duration_ms / 1000.0)
        speech_segments.append((start_time, end_time))

    return speech_segments

def _one_process(path):
    """检测一段音频中的人声段"""
    # 1. 压缩音频
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    sample_rate = audio.frame_rate
    audio_data = audio.raw_data

    # 2. 初始化VAD(0-3, 越大越容易被当作人声)
    vad = webrtcvad.Vad(0)

    # 3. 生成帧
    frames = list(_frame_generator(30, audio_data, sample_rate))

    # 4. 检测人声区间
    segments = _vad_collector(sample_rate, 30, 300, vad, frames)

    # 如果没有人声则将start和end均置为-1
    if len(segments) == 0:
        return {
            "start": -1,
            "end": -1,
            "segments": [],
        }

    return {
        "start": segments[0][0],
        "end": segments[-1][1],
        "segments": segments,
    }

# ===== 对外接口 =====

def get_endpoints_meta(dataset:list[dict], save_path:str, max_workers:int=4, save_middle:bool=True):
    """
    为数据集中的每段音频添加端点标签(主要针对已分轨的人声音频)
    - 要求dataset中每条数据路径字段为'path'
    - 写入字段为endpoints.start/end
    - 实时写入save_path中
    - save_middle决定是否将每句的端点记录到save.segments字段
    """
    dataset = dup_remove(dataset, save_path, 'path', 'endpoints')
    new_dataset = []
    with open(save_path, 'a', encoding='utf-8') as file:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_one_process, ele['path']): ele for ele in dataset}
            for future in tqdm(as_completed(futures), desc="Detecting endpoints"):
                ele = futures[future]   # 获取原始元素
                try:
                    result = future.result()
                    ele['endpoints'] = {
                        "start": result['start'],
                        "end": result['end']
                    }
                    if save_middle:
                        if "save" not in ele:
                            ele['save'] = {}
                        ele['save']['segments'] = result['segments']
                    new_dataset.append(ele)
                    json.dump(ele, file, ensure_ascii=False)
                    file.write("\n")
                except Exception:
                    pass
    return new_dataset
