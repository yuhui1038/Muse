import argparse, json, os, re, sys, glob
import librosa, torch
from tqdm import tqdm
from pydub import AudioSegment

# 逻辑就是把sunov5中的音频按照段落和歌词时间戳进行拆分，然后比对
# 有的段落描述没有传，只传了段落结构标签，这种情况就不进行mulan-T测试，我们只测试有描述的
# cleaned_lyrics是实际传给suno的，划分应该是参考timestamped_lyrics，不然没有时间戳

sys.path.append("Music_eval")
from muq import MuQMuLan

def get_data(dir:str):
    dataset = []
    """获取需要处理的数据"""
    for name in sorted(os.listdir(dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(dir, name)
        with open(path, 'r') as file:
            data = json.load(file)

        if not data['timestamped_lyrics']:
            continue
        words = data['timestamped_lyrics']['alignedWords']
        
        # 记录有效切片
        segments = []
        start = 0
        end = 0
        desc = ""
        valid = False
        for word in words:
            text:str = word['word']
            if text[0] != '[':
                end = word['endS']
                continue
            # 若上一段有效，进行添加
            if valid:
                segments.append({
                    "desc": desc,
                    "start": start,
                    "end": end
                })
                valid = False

            start_pos = text.find(": ")
            if start_pos == -1:
                # 无效段
                continue
            
            valid = True
            end_pos = text.rfind("]")
            desc = text[start_pos+2:end_pos]
            start = word['startS']
            end = word['endS']
        # 最后一段
        if valid:
            segments.append({
                "desc": desc,
                "start": start,
                "end": end
            })
        # 标记音乐路径
        song_id = data['song_id']
        music_path = os.path.join(dir, f"{song_id}.mp3")
        dataset.append({
            "song_id": song_id,
            "path": music_path,
            "segments": segments
        })
    return dataset

def split_audio(path:str, output_dir:str, time_pairs:list[tuple]):
    """按照节点进行划分"""
    audio = AudioSegment.from_mp3(path)
    os.makedirs(output_dir, exist_ok=True)

    total_ms = len(audio)
    for id, time_pair in enumerate(time_pairs):
        start, end = time_pair
        start = start * 1000
        end = min(end * 1000, total_ms)
        chunk = audio[start:end]
        out_path = os.path.join(output_dir, f"chunk_{id:03d}.mp3")
        chunk.export(out_path, format="mp3")

def preprocess():
    """预处理，为Mulan-T准备数据"""
    # 获取段落标签
    print("[1/3] Getting Segments Tags")
    data_dir = "sunov4_5"
    dataset = get_data(data_dir)

    # 段落切割
    middle_dir = "middle2"
    for ele in tqdm(dataset, desc="[2/3] Cutting Segments"):
        song_id = ele['song_id']
        music_path = ele['path']
        segments = ele['segments']
        time_pairs = []
        save_dir = os.path.join(middle_dir, song_id)

        for id, seg in enumerate(segments):
            time_pairs.append((seg['start'], seg['end']))
            seg['path'] = os.path.join(save_dir, f"chunk_{id:03d}.mp3")
        if not os.path.exists(save_dir):
            split_audio(music_path, save_dir, time_pairs)
    return dataset

def mulan_t_compute(dataset:list[dict]):
    device = "cuda:5"
    model_dir = "MuQ-MuLan-large"
    model = MuQMuLan.from_pretrained(model_dir).to(device).eval()

    scores = []
    for ele in tqdm(dataset, desc="[3/3] Computing Mulan-T"):
        segments = ele['segments']
        for seg in segments:
            path = seg['path']
            prompt = seg['desc']

            try:
                wav, _ = librosa.load(path, sr=24000)
                wavs = torch.tensor(wav).unsqueeze(0).to(device)
                with torch.no_grad():
                    audio_emb = model(wavs=wavs)
                    text_emb = model(texts=[prompt])
                    sim = model.calc_similarity(audio_emb, text_emb).item()
                scores.append(sim)
            except Exception as e:
                print(f"Error {path}: {e}")
    avg = sum(scores)/len(scores) if scores else 0
    print(avg)

def main():
    dataset = preprocess()
    mulan_t_compute(dataset)

if __name__ == "__main__":
    main()