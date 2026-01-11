import os
import math
from typing import List, Dict

import torch
import librosa
import numpy as np

from demucs.pretrained import get_model
from demucs.apply import apply_model


# ======================
# 基础配置
# ======================

SAMPLE_RATE = 44100
MAX_DURATION = 5          # 只取前 30 秒
BATCH_SIZE = 8               # 80G 显存可调到 16~32
VOCAL_DB_THRESHOLD = -35.0   # 人声存在阈值（经验值）
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:1"


# ======================
# 音频加载（只前 30s）
# ======================

def load_audio_30s(path: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    返回 shape: [channels=2, samples]
    """
    y, _ = librosa.load(
        path,
        sr=sr,
        mono=False,
        duration=MAX_DURATION
    )

    if y.ndim == 1:
        y = np.stack([y, y], axis=0)

    return torch.from_numpy(y)


# ======================
# dB 计算
# ======================

def rms_db(wav: torch.Tensor) -> float:
    """
    wav: [channels, samples]
    """
    rms = torch.sqrt(torch.mean(wav ** 2))
    db = 20 * torch.log10(rms + 1e-8)
    return db.item()


# ======================
# Demucs 人声判断
# ======================

class DemucsVocalDetector:
    def __init__(self):
        self.model = (
            get_model("htdemucs")
            .to(DEVICE)
            .eval()
        )
        self.vocal_idx = self.model.sources.index("vocals")

    @torch.no_grad()
    def batch_has_vocal(self, audio_paths: List[str]) -> Dict[str, bool]:
        """
        输入：音频路径列表
        输出：{path: 是否有人声}
        """
        results = {}

        batch_wavs = []
        batch_paths = []

        print("start load")
        for path in audio_paths:
            try:
                wav = load_audio_30s(path)
                batch_wavs.append(wav)
                batch_paths.append(path)
            except Exception as e:
                results[path] = False
                continue

        print("finish load")
        self._process_batch(batch_wavs, batch_paths, results)

        return results

    def _process_batch(self, wavs, paths, results):
        max_len = max(w.shape[1] for w in wavs)
        padded = []

        for w in wavs:
            if w.shape[1] < max_len:
                w = torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            padded.append(w)

        batch = torch.stack(padded, dim=0).to(DEVICE)

        print("start demucs")
        torch.cuda.synchronize()

        sources = apply_model(
            self.model,
            batch,
            SAMPLE_RATE,
            device=DEVICE,
            split=False,       # 🔥 核心
            progress=False
        )

        torch.cuda.synchronize()
        print("demucs done")

        vocals = sources[:, self.vocal_idx]

        for i, path in enumerate(paths):
            db = rms_db(vocals[i])
            results[path] = db > VOCAL_DB_THRESHOLD

# ======================
# 使用示例
# ======================

if __name__ == "__main__":
    audio_list = []

    detector = DemucsVocalDetector()
    result = detector.batch_has_vocal(audio_list)

    for k, v in result.items():
        print(f"{k}: {'有' if v else '无'}人声")