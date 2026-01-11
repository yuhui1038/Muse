import os
from tqdm import tqdm
from pydub import AudioSegment

def cut_dir(dir:str, save_dir:str):
    os.makedirs(save_dir, exist_ok=True)
    for name in tqdm(os.listdir(dir), desc="Cutting Audios"):
        if name.endswith(".txt") or name.endswith(".jsonl"):
            continue
        path = os.path.join(dir, name)
        audio = AudioSegment.from_file(path)
        three_minutes = 60 * 1000
        audio_3min = audio[:three_minutes]

        new_path = os.path.join(save_dir, name)
        audio_3min.export(new_path, format="mp3")

dirs = ["./audio/yue_cn", "./audio/yue_en", "./audio/ace-step_cn", "./audio/ace-step_en"]
save_dirs = ["./audio/yue_cut2_cn", "./audio/yue_cut2_en", "./audio/ace-step_cut2_cn", "./audio/ace-step_cut2_en"]

for dir, save_dir in zip(dirs, save_dirs):
    cut_dir(dir, save_dir)