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
# Prompt - Batch processing parameters
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

# Read JSONL file
print(f"Reading JSONL file: {args.jsonl_path}")
music_data_list = []
with open(args.jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            music_data_list.append(json.loads(line))

# Determine processing range
start_idx = args.start_idx
end_idx = len(music_data_list) if args.end_idx == -1 else min(args.end_idx, len(music_data_list))
music_data_list = music_data_list[start_idx:end_idx]
print(f"Total {len(music_data_list)} songs to generate (indices {start_idx} to {end_idx-1})")

# Detect processed songs - check completion status of each stage
def check_song_status(song_idx, output_dir):
    """
    Check song processing status
    Returns: (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)
    """
    if not os.path.exists(output_dir):
        return False, False, False, None, None, None
    
    # Find song directory (may have multiple, take the latest or first)
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
    
    # Use the latest directory (sorted by modification time)
    song_dir = max(song_dirs, key=lambda x: os.path.getmtime(x))
    
    # Check Stage 1: whether stage1 directory has vtrack and itrack .npy files
    stage1_dir = os.path.join(song_dir, "stage1")
    stage1_done = False
    stage1_output_set = []
    if os.path.exists(stage1_dir):
        stage1_files = [f for f in os.listdir(stage1_dir) if f.endswith('.npy')]
        vtrack_files = [f for f in stage1_files if '_vtrack' in f]
        itrack_files = [f for f in stage1_files if '_itrack' in f]
        if vtrack_files and itrack_files:
            stage1_done = True
            # Build stage1_output_set
            for f in vtrack_files + itrack_files:
                stage1_output_set.append(os.path.join(stage1_dir, f))
    
    # Check Stage 2: whether stage2 directory has corresponding .npy files
    stage2_dir = os.path.join(song_dir, "stage2")
    stage2_done = False
    if stage1_done and os.path.exists(stage2_dir):
        stage2_files = [f for f in os.listdir(stage2_dir) if f.endswith('.npy')]
        # Check if all stage1 files have corresponding stage2 files
        if stage1_output_set:
            stage1_basenames = {os.path.basename(f) for f in stage1_output_set}
            stage2_basenames = set(stage2_files)
            if stage1_basenames.issubset(stage2_basenames):
                stage2_done = True
    
    # Check Stage 3: whether there is a final mixed file (in song_dir root directory)
    stage3_done = False
    for root, dirs, files in os.walk(song_dir):
        if any(f.endswith('_mixed.mp3') for f in files):
            stage3_done = True
            break
    
    return stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_dir

# Detect processing status of all songs
song_status_map = {}  # {song_idx: (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)}
if os.path.exists(args.output_dir):
    print(f"\nDetecting processed songs...")
    for list_idx in range(len(music_data_list)):
        song_idx = start_idx + list_idx
        stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir = check_song_status(song_idx, args.output_dir)
        if stage1_done or stage2_done or stage3_done:
            song_status_map[song_idx] = (stage1_done, stage2_done, stage3_done, song_dir, stage1_output_set, stage2_output_dir)
    
    if song_status_map:
        fully_completed = [idx for idx, (s1, s2, s3, _, _, _) in song_status_map.items() if s3]
        partial_completed = [idx for idx, (s1, s2, s3, _, _, _) in song_status_map.items() if not s3]
        print(f"✓ Found {len(fully_completed)} fully completed songs: {sorted(fully_completed)}")
        if partial_completed:
            print(f"✓ Found {len(partial_completed)} partially completed songs: {sorted(partial_completed)}")
            for idx in sorted(partial_completed):
                s1, s2, s3, _, _, _ = song_status_map[idx]
                status_parts = []
                if s1: status_parts.append("Stage1")
                if s2: status_parts.append("Stage2")
                if s3: status_parts.append("Stage3")
                print(f"  Index {idx}: Completed {', '.join(status_parts)}")
        remaining_count = len(music_data_list) - len(fully_completed)
        print(f"✓ Will skip fully completed songs, {remaining_count} songs remaining to process")
    else:
        print(f"✓ No processed songs found, will start from the beginning")
else:
    print(f"✓ Output directory does not exist, will start from the beginning")

# Load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
print("Loading Stage 1 model...")
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
    Split lyrics by segments, following YuE official best practices:
    
    Official requirements:
    1. Lyrics should be segmented using structure tags: [verse], [chorus], [bridge], [outro], etc.
    2. Each segment is separated by two newlines "\n\n"
    3. Each segment is about 30 seconds (when --max_new_tokens 3000), don't put too many words
    4. Avoid using [intro] tag (not very stable), recommend starting with [verse] or [chorus]
    5. Supports multiple languages: English, Chinese, Cantonese, Japanese, Korean, etc.
    
    Args:
        lyrics: Raw lyrics string
    
    Returns:
        Structured lyrics segment list, each segment in [tag]\ncontent\n\n format
    """
    # Regular expression: match [any tag] and its following content
    # Supports: [Verse 1], [Pre-Chorus], [Chorus (Outro)] and other complex tags
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
    Clean and truncate genres string for filename generation
    Ensure filename is not too long (Linux filename limit is 255 bytes)
    
    Args:
        genres: Raw genres string
        max_length: Maximum length of genres part (default 80, leaving space for other parameters)
    
    Returns:
        Cleaned genres string
    """
    if not genres:
        return "Unknown"
    
    # Clean unsafe characters
    genres_clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', genres)
    genres_clean = genres_clean.strip('_').strip()
    
    # If contains comma-separated tags, try to keep first few tags
    if ',' in genres_clean:
        tags = [tag.strip() for tag in genres_clean.split(',')]
        # Try to keep first few tags until reaching length limit
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
            # If first tag is too long, directly truncate
            genres_clean = tags[0][:max_length] if tags else genres_clean[:max_length]
    
    # If still too long, directly truncate
    if len(genres_clean) > max_length:
        genres_clean = genres_clean[:max_length]
    
    # Replace spaces with hyphens (for consistency)
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
    """Process Stage 1 for a single song"""
    
    # Compatible with genre and description fields
    genres = music_data.get('genre') or music_data.get('description', '')
    lyrics_raw = music_data['lyrics']
    description = music_data.get('description', '')
    
    print(f"Description: {description[:100]}...")
    print(f"Genre tags: {genres}")
    
    # ===== Print original lyrics =====
    print("\n" + "="*60)
    print("【Original Lyrics (lyrics_raw)】")
    print("="*60)
    print(lyrics_raw)
    print("="*60 + "\n")
    
    lyrics = split_lyrics(lyrics_raw)
    
    # Validate lyrics format and give warnings (following official best practices)
    print(f"Lyrics analysis: Identified {len(lyrics)} segments")
    
    # ===== Print segmented lyrics =====
    print("\n" + "="*60)
    print("【Segmented Lyrics (lyrics)】")
    print("="*60)
    for i, seg in enumerate(lyrics):
        tag = seg.split('\n')[0].strip()
        # Check if unstable [intro] tag is used
        if 'intro' in tag.lower():
            print(f"  ⚠️  Warning: Segment {i+1} uses {tag} tag, official recommendation is to avoid [intro], use [verse] or [chorus] instead")
        else:
            print(f"  Segment {i+1}. {tag}")
        # Print each segment's content (limit length)
        content = seg.strip()
        if len(content) > 150:
            print(f"    Content preview: {content[:150]}...")
        else:
            print(f"    Content: {content}")
        print()
    print("="*60 + "\n")
    
    # Create output directory for this song
    random_id = uuid.uuid4()
    song_output_dir = os.path.join(args.output_dir, f"song_{song_idx:04d}_{random_id}")
    stage1_output_dir = os.path.join(song_output_dir, "stage1")
    stage2_output_dir = os.path.join(song_output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Stage 1: Generate audio tokens
    print("--- Stage 1: Generate audio tokens ---")
    stage1_output_set = []
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics
    
    # ===== Print prompt texts passed to model =====
    print("\n" + "="*60)
    print("【Prompt Texts Passed to Model (prompt_texts)】")
    print("="*60)
    print(f"Total {len(prompt_texts)} prompts (first is full prompt, subsequent are segments)\n")
    for i, pt in enumerate(prompt_texts):
        if i == 0:
            print(f"Prompt {i} [Full prompt header]:")
            if len(pt) > 300:
                print(f"{pt[:300]}...")
            else:
                print(pt)
        else:
            print(f"\nPrompt {i} [Segment {i}]:")
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
        print("Note: --no_sample is enabled, Stage 1 will use deterministic decoding; top_p/temperature will be ignored.")
    # special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    # Format text prompt
    # +1 because prompt_texts[0] is the full prompt which will be skipped, so need len(lyrics)+1 to process all segments
    run_n_segments = min(args.run_n_segments+1, len(lyrics)+1)
    
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference")):
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <=1 else 1.2
        
        # ===== Print currently processing segment =====
        if i == 0:
            print(f"\n[Segment {i}] Skipped (full prompt header)")
        else:
            print(f"\n" + "-"*60)
            print(f"[Processing segment {i}/{len(prompt_texts[:run_n_segments])-1}]")
            print("-"*60)
            tag_line = section_text.split('\n')[0] if '\n' in section_text else section_text[:50]
            print(f"Segment tag: {tag_line}")
            print(f"Segment content length: {len(section_text)} characters")
            if len(section_text) > 200:
                print(f"Segment content preview: {section_text[:200]}...")
            else:
                print(f"Segment content: {section_text}")
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
    # Clean genres string to avoid filename being too long
    genres_clean = sanitize_genres_for_filename(genres, max_length=80)
    vocal_save_path = os.path.join(stage1_output_dir, f"{genres_clean}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
    inst_save_path = os.path.join(stage1_output_dir, f"{genres_clean}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)
    
    return stage1_output_set, stage2_output_dir, song_output_dir

# Load Stage 2 model and vocoder (load only once)
print("\n" + "="*60)
print("Loading Stage 2 model...")
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

print("Loading vocoder...")
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)

# Batch process all songs - process each song completely before continuing to next
all_results = []
skipped_count = 0
for list_idx, music_data in enumerate(music_data_list):
    # Calculate actual song index (considering start_idx offset)
    song_idx = start_idx + list_idx
    
    try:
        # Compatible with genre and description fields
        genres = music_data.get('genre') or music_data.get('description', '')
        
        # Check processing status
        stage1_done = False
        stage2_done = False
        stage3_done = False
        song_output_dir = None
        stage1_output_set = None
        stage2_output_dir = None
        
        if song_idx in song_status_map:
            stage1_done, stage2_done, stage3_done, song_output_dir, stage1_output_set, stage2_output_dir = song_status_map[song_idx]
        
        # If all completed, skip
        if stage3_done:
            print(f"\n{'='*60}")
            print(f"⏭️  Skipping song {list_idx+1}/{len(music_data_list)} (index {song_idx}, fully completed)")
            print(f"{'='*60}")
            skipped_count += 1
            continue
        
        # Decide which stage to start from based on completion status
        print(f"\n{'='*60}")
        print(f"Starting to process song {list_idx+1}/{len(music_data_list)} (index {song_idx})")
        if stage1_done:
            print(f"  ✓ Stage 1 completed, will start from Stage 2")
        if stage2_done:
            print(f"  ✓ Stage 2 completed, will start from Stage 3")
        print(f"{'='*60}")
        
        # Stage 1: Generate audio tokens (if not completed)
        if not stage1_done:
            stage1_output_set, stage2_output_dir, song_output_dir = process_one_song(music_data, song_idx, len(music_data_list))
            print(f"✓ Stage 1 completed, generated {len(stage1_output_set)} files")
            for f in stage1_output_set:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"⏭️  Skipping Stage 1 (completed)")
            print(f"  Using existing Stage 1 outputs:")
            for f in stage1_output_set:
                print(f"    - {os.path.basename(f)}")
        
        # Note: Do not unload Stage 1 model here, as subsequent songs still need it
        # Stage 1 model will be unloaded uniformly after all songs are processed
        
        # Stage 2: Process audio tokens (if not completed)
        if not stage2_done:
            print(f"\n--- Stage 2: Processing song {list_idx+1} (index {song_idx}) ---")
            stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
            print(f"✓ Stage 2 completed, generated {len(stage2_result)} files")
            for f in stage2_result:
                print(f"  - {os.path.basename(f)}")
        else:
            print(f"\n⏭️  Skipping Stage 2 (completed)")
            # Get existing stage2 results
            stage2_result = []
            if os.path.exists(stage2_output_dir):
                for f in stage1_output_set:
                    basename = os.path.basename(f)
                    stage2_file = os.path.join(stage2_output_dir, basename)
                    if os.path.exists(stage2_file):
                        stage2_result.append(stage2_file)
            print(f"  Using existing Stage 2 outputs:")
            for f in stage2_result:
                print(f"    - {os.path.basename(f)}")
        
        # Stage 3: Reconstruct audio and mix (if not completed)
        final_output = None
        if not stage3_done:
            print(f"\n--- Stage 3: Reconstructing audio for song {list_idx+1} (index {song_idx}) ---")
            
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
                print(f"Created mix: {vocoder_mix}")
            except RuntimeError as e:
                print(e)
                print(f"Mix failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

            # Post process
            if recons_mix and vocoder_mix:
                final_output = os.path.join(song_output_dir, os.path.basename(recons_mix))
                replace_low_freq_with_energy_matched(
                    a_file=recons_mix,     # 16kHz
                    b_file=vocoder_mix,     # 48kHz
                    c_file=final_output,
                    cutoff_freq=5500.0
                )
                print(f"✓ Song {list_idx+1} (index {song_idx}) completed! Output: {final_output}")
        else:
            print(f"\n⏭️  Skipping Stage 3 (completed)")
            # Find final output file (usually in song_dir root directory)
            # First check root directory
            root_files = [f for f in os.listdir(song_output_dir) if f.endswith('_mixed.mp3')]
            if root_files:
                final_output = os.path.join(song_output_dir, root_files[0])
            else:
                # If root directory doesn't have it, traverse subdirectories to find
                for root, dirs, files in os.walk(song_output_dir):
                    for f in files:
                        if f.endswith('_mixed.mp3'):
                            final_output = os.path.join(root, f)
                            break
                    if final_output:
                        break
            if final_output:
                print(f"  Final output: {final_output}")
        
        all_results.append({
            'song_idx': song_idx,
            'genres': genres,
            'output_path': final_output if recons_mix and vocoder_mix else None
        })
        
    except Exception as e:
        print(f"✗ Error processing song {list_idx+1} (index {song_idx}): {e}")
        import traceback
        traceback.print_exc()
        continue

# After all songs are processed, unload models to free memory
if not args.disable_offload_model:
    print("\nCleaning up models to free memory...")
    if 'model' in locals():
        model.cpu()
        del model
    if 'model_stage2' in locals():
        model_stage2.cpu()
        del model_stage2
    torch.cuda.empty_cache()
    print("Models unloaded")

print("\n" + "="*60)
print("Batch generation complete!")
newly_processed = len([r for r in all_results if r.get('output_path')])
print(f"✓ Newly processed: {newly_processed} songs")
if skipped_count > 0:
    print(f"⏭️  Skipped (already completed): {skipped_count} songs")
print(f"📊 Total completed: {newly_processed + skipped_count} songs")
print("="*60)
for result in all_results:
    if result.get('output_path'):
        print(f"Song {result['song_idx']+1}: {result['output_path']}")

