import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import re
import random
import uuid
import copy
import json
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help="The model checkpoint path or identifier for the Stage 1 model.")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general", help="The model checkpoint path or identifier for the Stage 2 model.")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.")
parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during generation. Each segment is ~30s (with default max_new_tokens=3000). For example: 2=~1min, 6=~3min, 8=~4min.")
parser.add_argument("--stage2_batch_size", type=int, default=4, help="The batch size used in Stage 2 inference.")
parser.add_argument(
    "--no_sample",
    action="store_true",
    help="If set, disable sampling in Stage 1 generation (i.e., use deterministic decoding). When enabled, top_p/temperature will be ignored.",
)
# Prompt - 批量处理参数
parser.add_argument("--jsonl_path", type=str, required=True, help="The file path to a JSONL file containing genre and lyrics for batch processing.")
parser.add_argument("--start_idx", type=int, default=0, help="Start index in the JSONL file for batch processing.")
parser.add_argument("--end_idx", type=int, default=-1, help="End index in the JSONL file for batch processing. -1 means process all.")
parser.add_argument("--use_audio_prompt", action="store_true", help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.")
parser.add_argument("--audio_prompt_path", type=str, default="", help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0, help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--use_dual_tracks_prompt", action="store_true", help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.")
parser.add_argument("--vocal_track_prompt_path", type=str, default="", help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="", help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
# Output 
parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=42, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for xcodec configurations.')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the xcodec checkpoint.')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml', help='Path to Vocos config file.')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')


args = parser.parse_args()
if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
    raise FileNotFoundError("Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!")

stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
do_sample_stage1 = (not args.no_sample)

def seed_everything(seed=42): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

# 读取 JSONL 文件
print(f"正在读取 JSONL 文件: {args.jsonl_path}")
music_data_list = []
with open(args.jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            music_data_list.append(json.loads(line))

# 确定处理范围
start_idx = args.start_idx
end_idx = len(music_data_list) if args.end_idx == -1 else min(args.end_idx, len(music_data_list))
music_data_list = music_data_list[start_idx:end_idx]
print(f"共有 {len(music_data_list)} 首歌曲待生成 (索引 {start_idx} 到 {end_idx-1})")

# 检测已处理的歌曲 - 检查各阶段完成状态
def check_song_status(song_idx, output_dir):
    """
    检查歌曲的处理状态
    返回: (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)
    """
    if not os.path.exists(output_dir):
        return False, False, False, None, None, None
    
    # 查找该歌曲的目录（可能有多个，取最新的或第一个）
    song_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith('song_') and os.path.isdir(os.path.join(output_dir, item)):
            try:
                idx = int(item.split('_')[1])
                if idx == song_idx:
                    song_dirs.append(os.path.join(output_dir, item))
            except (ValueError, IndexError):
                continue
    
    if not song_dirs:
        return False, False, False, None, None, None
    
    # 使用最新的目录（按修改时间排序）
    song_dir = max(song_dirs, key=lambda x: os.path.getmtime(x))
    
    # 检查 Stage 1: stage1 目录下是否有 vtrack 和 itrack 的 .npy 文件
    stage1_dir = os.path.join(song_dir, "stage1")
    stage1_done = False
    stage1_output_set = []
    if os.path.exists(stage1_dir):
        stage1_files = [f for f in os.listdir(stage1_dir) if f.endswith('.npy')]
        vtrack_files = [f for f in stage1_files if '_vtrack' in f]
        itrack_files = [f for f in stage1_files if '_itrack' in f]
        if vtrack_files and itrack_files:
            stage1_done = True
            # 构建 stage1_output_set
            for f in vtrack_files + itrack_files:
                stage1_output_set.append(os.path.join(stage1_dir, f))
    
    # 检查 Stage 2: stage2 目录下是否有对应的 .npy 文件
    stage2_dir = os.path.join(song_dir, "stage2")
    stage2_done = False
    if stage1_done and os.path.exists(stage2_dir):
        stage2_files = [f for f in os.listdir(stage2_dir) if f.endswith('.npy')]
        # 检查是否所有 stage1 文件都有对应的 stage2 文件
        if stage1_output_set:
            stage1_basenames = {os.path.basename(f) for f in stage1_output_set}
            stage2_basenames = set(stage2_files)
            if stage1_basenames.issubset(stage2_basenames):
                stage2_done = True
    
    # 检查 Stage 3: 是否有最终混音文件（在 song_dir 根目录下）
    stage3_done = False
    for root, dirs, files in os.walk(song_dir):
        if any(f.endswith('_mixed.mp3') for f in files):
            stage3_done = True
            break
    
    return stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_dir

# 检测所有歌曲的处理状态
song_status_map = {}  # {song_idx: (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)}
if os.path.exists(args.output_dir):
    print(f"\n正在检测已处理的歌曲...")
    for list_idx in range(len(music_data_list)):
        song_idx = start_idx + list_idx
        stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir = check_song_status(song_idx, args.output_dir)
        if stage1_done or stage2_done or stage3_done:
            song_status_map[song_idx] = (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)
    
    if song_status_map:
        fully_completed = [idx for idx, (s1, s2, s3, _, _, _) in song_status_map.items() if s3]
        partial_completed = [idx for idx, (s1, s2, s3, _, _, _) in song_status_map.items() if not s3]
        print(f"✓ 发现 {len(fully_completed)} 首完全完成的歌曲: {sorted(fully_completed)}")
        if partial_completed:
            print(f"✓ 发现 {len(partial_completed)} 首部分完成的歌曲: {sorted(partial_completed)}")
            for idx in sorted(partial_completed):
                s1, s2, s3, _, _, _ = song_status_map[idx]
                status_parts = []
                if s1: status_parts.append("Stage1")
                if s2: status_parts.append("Stage2")
                if s3: status_parts.append("Stage3")
                print(f"  索引 {idx}: 已完成 {', '.join(status_parts)}")
        remaining_count = len(music_data_list) - len(fully_completed)
        print(f"✓ 将跳过完全完成的歌曲，还需处理 {remaining_count} 首")
    else:
        print(f"✓ 未发现已处理的歌曲，将从头开始处理")
else:
    print(f"✓ 输出目录不存在，将从头开始处理")

# load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
print("正在加载 Stage 1 模型...")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Using flash_attention_2 for better performance
    # device_map="auto",
    )
# to device, if gpu is available
model.to(device)
model.eval()

if torch.__version__ >= "2.0.0":
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Warning: torch.compile not available: {e}")

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
# Load checkpoint with weights_only=False to allow OmegaConf types
# Note: Only use this if you trust the checkpoint source
parameter_dict = torch.load(args.resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes

def split_lyrics(lyrics):
    """
    将歌词按段落分割，遵循YuE官方最佳实践：
    
    官方要求：
    1. 歌词应该分段，使用结构标签：[verse], [chorus], [bridge], [outro] 等
    2. 每个段落用两个换行符 "\n\n" 分隔
    3. 每段约30秒（--max_new_tokens 3000时），不要放太多词
    4. 避免使用 [intro] 标签（不太稳定），建议从 [verse] 或 [chorus] 开始
    5. 支持多种语言：英语、中文、粤语、日语、韩语等
    
    参数:
        lyrics: 原始歌词字符串
    
    返回:
        结构化的歌词段落列表，每段以 [标签]\n内容\n\n 格式
    """
    # 正则表达式：匹配 [任意标签] 及其后的内容
    # 支持: [Verse 1], [Pre-Chorus], [Chorus (Outro)] 等复杂标签
    pattern = r"\[([^\]]+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics

def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def stage2_generate(model, prompt, batch_size=16):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)
    
    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1
        )
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten the 2D array to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    
    block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

    # Teacher forcing generate loop
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx:frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model.generate(input_ids=input_ids, 
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )
        
        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output

def sanitize_genres_for_filename(genres, max_length=80):
    """
    清理和截断 genres 字符串，用于生成文件名
    确保文件名不会过长（Linux 文件名限制为 255 字节）
    
    Args:
        genres: 原始 genres 字符串
        max_length: genres 部分的最大长度（默认 80，为其他参数留出空间）
    
    Returns:
        清理后的 genres 字符串
    """
    if not genres:
        return "Unknown"
    
    # 清理不安全字符
    genres_clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', genres)
    genres_clean = genres_clean.strip('_').strip()
    
    # 如果包含逗号分隔的标签，尝试保留前几个标签
    if ',' in genres_clean:
        tags = [tag.strip() for tag in genres_clean.split(',')]
        # 尝试保留前几个标签，直到达到长度限制
        result_tags = []
        current_length = 0
        for tag in tags:
            if current_length + len(tag) + 1 <= max_length:  # +1 for comma
                result_tags.append(tag)
                current_length += len(tag) + 1
            else:
                break
        if result_tags:
            genres_clean = ','.join(result_tags)
        else:
            # 如果第一个标签就太长，直接截断
            genres_clean = tags[0][:max_length] if tags else genres_clean[:max_length]
    
    # 如果仍然太长，直接截断
    if len(genres_clean) > max_length:
        genres_clean = genres_clean[:max_length]
    
    # 替换空格为连字符（保持一致性）
    genres_clean = genres_clean.replace(' ', '-')
    
    return genres_clean

def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set)), desc="Stage 2 inference"):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            stage2_result.append(output_filename)
            continue
        
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)
        
        # Process the ending part of the prompt
        if output_duration*50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result

def process_one_song(music_data, song_idx, total_songs):
    """处理单首歌曲的 Stage 1"""
    
    # 兼容 genre 和 description 字段
    genres = music_data.get('genre') or music_data.get('description', '')
    lyrics_raw = music_data['lyrics']
    description = music_data.get('description', '')
    
    print(f"描述: {description[:100]}...")
    print(f"流派标签: {genres}")
    
    # ===== 打印原始歌词 =====
    print("\n" + "="*60)
    print("【原始歌词 (lyrics_raw)】")
    print("="*60)
    print(lyrics_raw)
    print("="*60 + "\n")
    
    lyrics = split_lyrics(lyrics_raw)
    
    # 验证歌词格式并给出警告（遵循官方最佳实践）
    print(f"歌词分析: 共识别到 {len(lyrics)} 个段落")
    
    # ===== 打印分割后的歌词段落 =====
    print("\n" + "="*60)
    print("【分割后的歌词段落 (lyrics)】")
    print("="*60)
    for i, seg in enumerate(lyrics):
        tag = seg.split('\n')[0].strip()
        # 检查是否使用了不稳定的 [intro] 标签
        if 'intro' in tag.lower():
            print(f"  ⚠️  警告: 段落 {i+1} 使用了 {tag} 标签，官方建议避免使用 [intro]，推荐用 [verse] 或 [chorus]")
        else:
            print(f"  段落 {i+1}. {tag}")
        # 打印每个段落的内容（限制长度）
        content = seg.strip()
        if len(content) > 150:
            print(f"    内容预览: {content[:150]}...")
        else:
            print(f"    内容: {content}")
        print()
    print("="*60 + "\n")
    
    # 创建此歌曲专属的输出目录
    random_id = uuid.uuid4()
    song_output_dir = os.path.join(args.output_dir, f"song_{song_idx:04d}_{random_id}")
    stage1_output_dir = os.path.join(song_output_dir, "stage1")
    stage2_output_dir = os.path.join(song_output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Stage 1: 生成音频 tokens
    print("--- Stage 1: 生成音频 tokens ---")
    stage1_output_set = []
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics
    
    # ===== 打印传入模型的提示文本 =====
    print("\n" + "="*60)
    print("【传入模型的提示文本 (prompt_texts)】")
    print("="*60)
    print(f"总共 {len(prompt_texts)} 个提示（第1个是完整提示，后续是各个段落）\n")
    for i, pt in enumerate(prompt_texts):
        if i == 0:
            print(f"提示 {i} [完整提示头部]:")
            if len(pt) > 300:
                print(f"{pt[:300]}...")
            else:
                print(pt)
        else:
            print(f"\n提示 {i} [段落 {i}]:")
            if len(pt) > 200:
                print(f"{pt[:200]}...")
            else:
                print(pt)
    print("="*60 + "\n")
    
    output_seq = None
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = args.repetition_penalty
    if not do_sample_stage1:
        print("注意: 已启用 --no_sample，Stage 1 将使用确定性解码；top_p/temperature 将被忽略。")
    # special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    # Format text prompt
    # +1是因为prompt_texts[0]是完整提示会被跳过，所以需要len(lyrics)+1来处理所有段落
    run_n_segments = min(args.run_n_segments+1, len(lyrics)+1)
    
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference")):
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <=1 else 1.2
        
        # ===== 打印当前处理的段落 =====
        if i == 0:
            print(f"\n[段落 {i}] 跳过（完整提示头部）")
        else:
            print(f"\n" + "-"*60)
            print(f"[正在处理段落 {i}/{len(prompt_texts[:run_n_segments])-1}]")
            print("-"*60)
            tag_line = section_text.split('\n')[0] if '\n' in section_text else section_text[:50]
            print(f"段落标签: {tag_line}")
            print(f"段落内容长度: {len(section_text)} 字符")
            if len(section_text) > 200:
                print(f"段落内容预览: {section_text[:200]}...")
            else:
                print(f"段落内容: {section_text}")
            print("-"*60)
        
        if i==0:
            continue
        if i==1:
            if args.use_dual_tracks_prompt or args.use_audio_prompt:
                if args.use_dual_tracks_prompt:
                    vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                    instrumental_ids = load_audio_mono(args.instrumental_track_prompt_path)
                    vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                    instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                    vocals_ids = codectool.npy2ids(vocals_ids[0])
                    instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                    ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
                    audio_prompt_codec = ids_segment_interleaved[int(args.prompt_start_time*50*2): int(args.prompt_end_time*50*2)]
                    audio_prompt_codec = audio_prompt_codec.tolist()
                elif args.use_audio_prompt:
                    audio_prompt = load_audio_mono(args.audio_prompt_path)
                    raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                    # Format audio prompt
                    code_ids = codectool.npy2ids(raw_codes[0])
                    audio_prompt_codec = code_ids[int(args.prompt_start_time *50): int(args.prompt_end_time *50)] # 50 is tps of xcodec
                audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
                sentence_ids = mmtokenizer.tokenize("[start_of_reference]") +  audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384-max_new_tokens-1
        if input_ids.shape[-1] > max_context:
            print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=100, 
                do_sample=do_sample_stage1,
                top_p=top_p,
                temperature=temperature, 
                repetition_penalty=repetition_penalty, 
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
                )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx)!=len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    # 清理 genres 字符串，避免文件名过长
    genres_clean = sanitize_genres_for_filename(genres, max_length=80)
    vocal_save_path = os.path.join(stage1_output_dir, f"{genres_clean}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
    inst_save_path = os.path.join(stage1_output_dir, f"{genres_clean}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)
    
    return stage1_output_set, stage2_output_dir, song_output_dir

# 加载 Stage 2 模型和 vocoder（只加载一次）
print("\n" + "="*60)
print("正在加载 Stage 2 模型...")
print("="*60)
model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Using flash_attention_2 for better performance
    # device_map="auto",
    )
model_stage2.to(device)
model_stage2.eval()

if torch.__version__ >= "2.0.0":
    try:
        model_stage2 = torch.compile(model_stage2)
    except Exception as e:
        print(f"Warning: torch.compile not available: {e}")

print("正在加载 vocoder...")
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)

# 批量处理所有歌曲 - 每首歌曲完整处理后再继续下一首
all_results = []
skipped_count = 0
for list_idx, music_data in enumerate(music_data_list):
    # 计算真实的歌曲索引（考虑 start_idx 偏移）
    song_idx = start_idx + list_idx
    
    try:
        # 兼容 genre 和 description 字段
        genres = music_data.get('genre') or music_data.get('description', '')
        
        # 检查处理状态
        stage1_done = False
        stage2_done = False
        stage3_done = False
        song_output_dir = None
        stage1_output_set = None
        stage2_output_dir = None
        
        if song_idx in song_status_map:
            stage1_done, stage2_done, stage3_done, song_output_dir, stage1_output_set, stage2_output_dir = song_status_map[song_idx]
        
        # 如果全部完成，跳过
        if stage3_done:
            print(f"\n{'='*60}")
            print(f"⏭️  跳过第 {list_idx+1}/{len(music_data_list)} 首歌曲（索引 {song_idx}，已完全完成）")
            print(f"{'='*60}")
            skipped_count += 1
            continue
        
        # 根据完成状态决定从哪个阶段开始
        print(f"\n{'='*60}")
        print(f"开始处理第 {list_idx+1}/{len(music_data_list)} 首歌曲（索引 {song_idx}）")
        if stage1_done:
            print(f"  ✓ Stage 1 已完成，将从 Stage 2 开始")
        if stage2_done:
            print(f"  ✓ Stage 2 已完成，将从 Stage 3 开始")
        print(f"{'='*60}")
        
        # Stage 1: 生成音频 tokens（如果未完成）
        if not stage1_done:
            stage1_output_set, stage2_output_dir, song_output_dir = process_one_song(music_data, song_idx, len(music_data_list))
            print(f"✓ Stage 1 完成，生成了 {len(stage1_output_set)} 个文件")
            for f in stage1_output_set:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"⏭️  跳过 Stage 1（已完成）")
            print(f"  使用已有的 Stage 1 输出:")
            for f in stage1_output_set:
                print(f"    - {os.path.basename(f)}")
        
        # 注意：不要在这里卸载 Stage 1 模型，因为后续歌曲还需要使用
        # Stage 1 模型会在所有歌曲处理完成后统一卸载
        
        # Stage 2: 处理音频 tokens（如果未完成）
        if not stage2_done:
            print(f"\n--- Stage 2: 处理第 {list_idx+1} 首歌曲（索引 {song_idx}）---")
            stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
            print(f"✓ Stage 2 完成，生成了 {len(stage2_result)} 个文件")
            for f in stage2_result:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"\n⏭️  跳过 Stage 2（已完成）")
            # 获取已有的 stage2 结果
            stage2_result = []
            if os.path.exists(stage2_output_dir):
                for f in stage1_output_set:
                    basename = os.path.basename(f)
                    stage2_file = os.path.join(stage2_output_dir, basename)
                    if os.path.exists(stage2_file):
                        stage2_result.append(stage2_file)
            print(f"  使用已有的 Stage 2 输出:")
            for f in stage2_result:
                print(f"    - {os.path.basename(f)}")
        
        # Stage 3: 重建音频和混音（如果未完成）
        final_output = None
        if not stage3_done:
            print(f"\n--- Stage 3: 重建第 {list_idx+1} 首歌曲的音频（索引 {song_idx}）---")
            
            # reconstruct tracks
            recons_output_dir = os.path.join(song_output_dir, "recons")
            recons_mix_dir = os.path.join(recons_output_dir, 'mix')
            os.makedirs(recons_mix_dir, exist_ok=True)
            tracks = []
            for npy in stage2_result:
                codec_result = np.load(npy)
                decodec_rlt=[]
                with torch.no_grad():
                    decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
                decoded_waveform = decoded_waveform.cpu().squeeze(0)
                decodec_rlt.append(torch.as_tensor(decoded_waveform))
                decodec_rlt = torch.cat(decodec_rlt, dim=-1)
                save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
                tracks.append(save_path)
                save_audio(decodec_rlt, save_path, 16000)
            
            # mix tracks
            recons_mix = None
            for inst_path in tracks:
                try:
                    if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                        and '_itrack' in inst_path:
                        # find pair
                        vocal_path = inst_path.replace('_itrack', '_vtrack')
                        if not os.path.exists(vocal_path):
                            continue
                        # mix
                        recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
                        vocal_stem, sr = sf.read(inst_path)
                        instrumental_stem, _ = sf.read(vocal_path)
                        mix_stem = (vocal_stem + instrumental_stem) / 1
                        sf.write(recons_mix, mix_stem, sr)
                except Exception as e:
                    print(e)

            # vocoder to upsample audios
            vocoder_output_dir = os.path.join(song_output_dir, 'vocoder')
            vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
            vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
            os.makedirs(vocoder_mix_dir, exist_ok=True)
            os.makedirs(vocoder_stems_dir, exist_ok=True)
            
            for npy in stage2_result:
                if '_itrack' in npy:
                    # Process instrumental
                    instrumental_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'itrack.mp3'),
                        args.rescale,
                        args,
                        inst_decoder,
                        codec_model
                    )
                else:
                    # Process vocal
                    vocal_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
                        args.rescale,
                        args,
                        vocal_decoder,
                        codec_model
                    )
            
            # mix tracks
            vocoder_mix = None
            try:
                mix_output = instrumental_output + vocal_output
                vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
                save_audio(mix_output, vocoder_mix, 44100, args.rescale)
                print(f"创建混音: {vocoder_mix}")
            except RuntimeError as e:
                print(e)
                print(f"混音失败! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

            # Post process
            if recons_mix and vocoder_mix:
                final_output = os.path.join(song_output_dir, os.path.basename(recons_mix))
                replace_low_freq_with_energy_matched(
                    a_file=recons_mix,     # 16kHz
                    b_file=vocoder_mix,     # 48kHz
                    c_file=final_output,
                    cutoff_freq=5500.0
                )
                print(f"✓ 第 {list_idx+1} 首歌曲（索引 {song_idx}）完成! 输出: {final_output}")
        else:
            print(f"\n⏭️  跳过 Stage 3（已完成）")
            # 查找最终输出文件（通常在 song_dir 根目录下）
            # 先检查根目录
            root_files = [f for f in os.listdir(song_output_dir) if f.endswith('_mixed.mp3')]
            if root_files:
                final_output = os.path.join(song_output_dir, root_files[0])
            else:
                # 如果根目录没有，遍历子目录查找
                for root, dirs, files in os.walk(song_output_dir):
                    for f in files:
                        if f.endswith('_mixed.mp3'):
                            final_output = os.path.join(root, f)
                            break
                    if final_output:
                        break
            if final_output:
                print(f"  最终输出: {final_output}")
        
        all_results.append({
            'song_idx': song_idx,
            'genres': genres,
            'output_path': final_output if recons_mix and vocoder_mix else None
        })
        
    except Exception as e:
        print(f"✗ 处理第 {list_idx+1} 首歌曲（索引 {song_idx}）时出错: {e}")
        import traceback
        traceback.print_exc()
        continue

# 所有歌曲处理完成后，卸载模型释放内存
if not args.disable_offload_model:
    print("\n清理模型以释放内存...")
    if 'model' in locals():
        model.cpu()
        del model
    if 'model_stage2' in locals():
        model_stage2.cpu()
        del model_stage2
    torch.cuda.empty_cache()
    print("模型已卸载")

print("\n" + "="*60)
print("批量生成完成!")
newly_processed = len([r for r in all_results if r.get('output_path')])
print(f"✓ 本次新处理: {newly_processed} 首歌曲")
if skipped_count > 0:
    print(f"⏭️  跳过已完成: {skipped_count} 首歌曲")
print(f"📊 总计完成: {newly_processed + skipped_count} 首歌曲")
print("="*60)
for result in all_results:
    if result.get('output_path'):
        print(f"歌曲 {result['song_idx']+1}: {result['output_path']}")

