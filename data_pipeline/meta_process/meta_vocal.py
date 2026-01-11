import os
import math
from typing import List, Dict

import torch
import librosa
import numpy as np

from demucs.pretrained import get_model
from demucs.apply import apply_model


# ======================
# Basic Configuration
# ======================

SAMPLE_RATE = 44100
MAX_DURATION = 5          # Only take first 30 seconds
BATCH_SIZE = 8               # Can adjust to 16~32 for 80G GPU memory
VOCAL_DB_THRESHOLD = -35.0   # Vocal presence threshold (empirical value)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:1"


# ======================
# Audio Loading (first 30s only)
# ======================

def load_audio_30s(path: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Returns shape: [channels=2, samples]
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
# dB Calculation
# ======================

def rms_db(wav: torch.Tensor) -> float:
    """
    wav: [channels, samples]
    """
    rms = torch.sqrt(torch.mean(wav ** 2))
    db = 20 * torch.log10(rms + 1e-8)
    return db.item()


# ======================
# Demucs Vocal Detection
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
        Input: List of audio paths
        Output: {path: whether has vocals}
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
            split=False,       # 🔥 Core
            progress=False
        )

        torch.cuda.synchronize()
        print("demucs done")

        vocals = sources[:, self.vocal_idx]

        for i, path in enumerate(paths):
            db = rms_db(vocals[i])
            results[path] = db > VOCAL_DB_THRESHOLD

# ======================
# Usage Example
# ======================

if __name__ == "__main__":
    audio_list = []

    detector = DemucsVocalDetector()
    result = detector.batch_has_vocal(audio_list)

    for k, v in result.items():
        print(f"{k}: {'Has' if v else 'No'} vocals")