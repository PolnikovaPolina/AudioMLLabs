import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import (
    load_audio_chunk, augment_audio,
    audio_to_melspec, audio_to_melspec_at, mel_to_tensor,
)
from config import SAMPLE_RATE, DURATION, NUM_CLASSES


# ─── Підхід 1: Базовий датасет ────────────────────────────────────────────
class BirdDataset(Dataset):
    """
    Mel Spectrogram Dataset для Підходу 1 (CNN Baseline).
    Повертає: (mel_tensor: (3, F, T), label: (NUM_CLASSES,))
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        audio = load_audio_chunk(
            row["filepath"],
            mode="random" if self.augment else "center",
        )
        if self.augment:
            audio = augment_audio(audio)

        tensor = mel_to_tensor(audio_to_melspec(audio))   # (3, F, T)
        label  = self._make_label(row)
        return tensor, label

    def _make_label(self, row) -> torch.Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[int(row["label_idx"])] = 1.0
        if "secondary_idx" in row.index and isinstance(row["secondary_idx"], list):
            for idx in row["secondary_idx"]:
                label[int(idx)] = 1.0
        return label


# ─── Підхід 2: Датасет з геокоординатами ──────────────────────────────────
class BirdDatasetWithGeo(Dataset):
    """
    Mel + Geographic coords Dataset для Підходу 2.
    Повертає: (mel_tensor: (3, F, T), geo: (2,), label: (NUM_CLASSES,))

    geo = [lat / 90, lon / 180]  →  нормалізовано до [-1, 1].
    NaN координати замінюються нулями.
    """

    LAT_SCALE = 90.0
    LON_SCALE = 180.0

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        audio = load_audio_chunk(
            row["filepath"],
            mode="random" if self.augment else "center",
        )
        if self.augment:
            audio = augment_audio(audio)

        tensor = mel_to_tensor(audio_to_melspec(audio))

        lat = float(row["latitude"])  if pd.notna(row.get("latitude"))  else 0.0
        lon = float(row["longitude"]) if pd.notna(row.get("longitude")) else 0.0
        geo = torch.tensor(
            [lat / self.LAT_SCALE, lon / self.LON_SCALE],
            dtype=torch.float32,
        )

        label = self._make_label(row)
        return tensor, geo, label

    def _make_label(self, row) -> torch.Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[int(row["label_idx"])] = 1.0
        if "secondary_idx" in row.index and isinstance(row["secondary_idx"], list):
            for idx in row["secondary_idx"]:
                label[int(idx)] = 1.0
        return label


# ─── Підхід 3: Audio-pretrained нормалізація ──────────────────────────────
class BirdDatasetAT(Dataset):
    """
    Per-frequency Instance Norm Dataset для Підходу 3.
    Повертає: (mel_tensor: (3, F, T), label: (NUM_CLASSES,))
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        audio = load_audio_chunk(
            row["filepath"],
            mode="random" if self.augment else "center",
        )
        if self.augment:
            audio = augment_audio(audio)

        tensor = mel_to_tensor(audio_to_melspec_at(audio))
        label  = self._make_label(row)
        return tensor, label

    def _make_label(self, row) -> torch.Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[int(row["label_idx"])] = 1.0
        if "secondary_idx" in row.index and isinstance(row["secondary_idx"], list):
            for idx in row["secondary_idx"]:
                label[int(idx)] = 1.0
        return label


# ─── Inference: Test Soundscapes ──────────────────────────────────────────
class SoundscapeDataset(Dataset):
    """
    Нарізає test soundscape на 5-секундні чанки для інференсу.
    Повертає: (mel_tensor: (3, F, T), row_id: str)

    mel_type:
        'base' — для Підходів 1 і 2 (min-max norm)
        'at'   — для Підходу 3 (per-frequency norm)
    """

    def __init__(self, filepath: str,
                 mel_type: str = "base",
                 chunk_duration: int = DURATION,
                 sr: int = SAMPLE_RATE):
        import librosa
        self.mel_type = mel_type
        self.chunk    = chunk_duration * sr
        self.stem     = os.path.splitext(os.path.basename(filepath))[0]

        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        n_chunks = len(audio) // self.chunk

        self.chunks    = [audio[i * self.chunk: (i + 1) * self.chunk] for i in range(n_chunks)]
        self.end_times = [(i + 1) * chunk_duration                    for i in range(n_chunks)]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        audio  = self.chunks[idx].astype(np.float32)
        row_id = f"{self.stem}_{self.end_times[idx]}"
        mel_fn = audio_to_melspec_at if self.mel_type == "at" else audio_to_melspec
        return mel_to_tensor(mel_fn(audio)), row_id
