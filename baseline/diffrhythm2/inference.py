# Copyright 2025 ASLP Lab and Xiaomi Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchaudio
import argparse
import json
import os
from tqdm import tqdm
import random
import pedalboard
import numpy as np

from muq import MuQMuLan
from diffrhythm2.cfm import CFM
from diffrhythm2.backbones.dit import DiT
from bigvgan.model import Generator
from huggingface_hub import hf_hub_download


STRUCT_INFO = {
    "[start]": 500,
    "[end]": 501,
    "[intro]": 502,
    "[verse]": 503,
    "[chorus]": 504,
    "[outro]": 505,
    "[inst]": 506,
    "[solo]": 507,
    "[bridge]": 508,
    "[hook]": 509,
    "[break]": 510,
    "[stop]": 511,
    "[space]": 512
}

lrc_tokenizer = None


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # best-effort deterministic behavior; some ops may still be nondeterministic on certain GPUs/kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

class CNENTokenizer():
    def __init__(self):
        curr_path = os.path.abspath(__file__)
        vocab_path = os.path.join(os.path.dirname(curr_path), "g2p/g2p/vocab.json")
        with open(vocab_path, 'r') as file:
            self.phone2id:dict = json.load(file)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        from g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p
    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x+1 for x in token]
        return token
    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])


def prepare_model(repo_id, device):
    diffrhythm2_ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir="./ckpt",
        local_files_only=False,
    )
    diffrhythm2_config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        local_dir="./ckpt",
        local_files_only=False,
    )
    with open(diffrhythm2_config_path) as f:
        model_config = json.load(f)

    model_config['use_flex_attn'] = False
    diffrhythm2 = CFM(
        transformer=DiT(
            **model_config
        ),
        num_channels=model_config['mel_dim'],
        block_size=model_config['block_size'],
    )

    total_params = sum(p.numel() for p in diffrhythm2.parameters())

    diffrhythm2 = diffrhythm2.to(device)
    if diffrhythm2_ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(diffrhythm2_ckpt_path)
    else:
        ckpt = torch.load(diffrhythm2_ckpt_path, map_location='cpu')
    diffrhythm2.load_state_dict(ckpt)
    print(f"Total params: {total_params:,}")

    # load Mulan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./ckpt").to(device)

    # load frontend
    lrc_tokenizer = CNENTokenizer()

    # load decoder
    decoder_ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="decoder.bin",
        local_dir="./ckpt",
        local_files_only=False,
    )
    decoder_config_path = hf_hub_download(
        repo_id=repo_id,
        filename="decoder.json",
        local_dir="./ckpt",
        local_files_only=False,
    )
    decoder = Generator(decoder_config_path, decoder_ckpt_path)
    decoder = decoder.to(device)
    return diffrhythm2, mulan, lrc_tokenizer, decoder


def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.split("\n")
    for line in lyrics:
        struct_idx = STRUCT_INFO.get(line, None)
        if struct_idx is not None:
            lyrics_with_time.append([struct_idx, STRUCT_INFO['[stop]']])
        else:
            tokens = lrc_tokenizer.encode(line.strip())
            tokens = tokens + [STRUCT_INFO['[stop]']]
            lyrics_with_time.append(tokens)
    return lyrics_with_time


def make_fake_stereo(audio, sampling_rate):
    left_channel = audio
    right_channel = audio.copy()
    right_channel = right_channel * 0.8
    delay_samples = int(0.01 * sampling_rate)
    right_channel = np.roll(right_channel, delay_samples)
    right_channel[:,:delay_samples] = 0
    stereo_audio = np.concatenate([left_channel, right_channel], axis=0)
    
    return stereo_audio
    

def inference(
        model, 
        decoder,
        text,
        style_prompt,
        duration,
        output_dir, 
        song_name,
        cfg_strength,
        sample_steps=32,
        process_bar=True,
        fake_stereo=True,
    ):
    with torch.inference_mode():
        latent = model.sample_block_cache(
            text=text.unsqueeze(0),
            duration=int(duration * 5),
            style_prompt=style_prompt.unsqueeze(0),
            steps=sample_steps,
            cfg_strength=cfg_strength,
            process_bar=process_bar,
        )
        latent = latent.transpose(1, 2)
        audio = decoder.decode_audio(latent, overlap=5, chunk_size=20)

        basename = f"{song_name}.mp3"
        output_path = os.path.join(output_dir, basename)

        num_channels = 1
        audio = audio.float().cpu().numpy().squeeze()[None, :]
        if fake_stereo:
            audio = make_fake_stereo(audio, decoder.h.sampling_rate)
            num_channels = 2
        
        with pedalboard.io.AudioFile(output_path, "w", decoder.h.sampling_rate, num_channels) as f:
            f.write(audio)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--input-jsonl', type=str, default=None)
    parser.add_argument('--cfg-strength', type=float, default=2.0)
    parser.add_argument('--max-secs', type=float, default=210.0)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--fake-stereo', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do-sample', action='store_true', default=False)

    args = parser.parse_args()
    
    output_dir = args.output_dir
    input_jsonl = args.input_jsonl
    cfg_strength = args.cfg_strength
    max_secs = args.max_secs
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # reproducibility
    set_seed(args.seed, deterministic=(not args.do_sample))

    # load diffrhythm2
    diffrhythm2, mulan, lrc_tokenizer, decoder = prepare_model(args.repo_id, device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_jsonl, 'r') as f:
        input_info = [json.loads(i.strip()) for i in f.readlines()]

    for i in tqdm(range(len(input_info))):
        info = input_info[i]
        song_name = info.get('song_name', f"{i:04d}")
        lyrics = info.get('lyrics', None)
        style_prompt = info.get('style_prompt', None)
        if lyrics is None or style_prompt is None:
            print(f"lyrics or style_prompt is None, skip {song_name}")
            continue

        # preprocess lyrics
        with open(lyrics, 'r') as f:
            lyrics = f.read()
        lyrics_token = parse_lyrics(lyrics)
        lyrics_token = torch.tensor(sum(lyrics_token, []), dtype=torch.long, device=device)

        # preprocess style prompt
        if os.path.isfile(style_prompt):
            prompt_wav, sr = torchaudio.load(style_prompt)
            prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, 24000)
            if prompt_wav.shape[1] > 24000 * 10:
                if args.do_sample:
                    start = random.randint(0, prompt_wav.shape[1] - 24000 * 10)
                else:
                    start = 0
                prompt_wav = prompt_wav[:, start:start+24000*10]
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            with torch.no_grad():
                style_prompt_embed = mulan(wavs = prompt_wav)
        else:
            with torch.no_grad():
                style_prompt_embed = mulan(texts = [style_prompt])
        style_prompt_embed = style_prompt_embed.to(device).squeeze(0)

        if device.type != 'cpu':
            diffrhythm2 = diffrhythm2.half()
            decoder = decoder.half()
            style_prompt_embed = style_prompt_embed.half()

        inference(
            model=diffrhythm2,
            decoder=decoder,
            text=lyrics_token,
            style_prompt=style_prompt_embed,
            duration=max_secs,
            output_dir=output_dir,
            song_name=song_name,
            sample_steps=args.steps,
            cfg_strength=cfg_strength,
            fake_stereo=args.fake_stereo,
        )


