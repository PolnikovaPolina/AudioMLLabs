import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import (
    load_audio_chunk,
)
from config import SAMPLE_RATE, DURATION, NUM_CLASSES, DATA_DIR, ALL_CLASSES

import soundfile as sf
import torchaudio


def load_pseudolabels(
    pseudolabels_cache_file: str = "pseudolabels.npz",
    validation_files: list[str] | None = None,
    filter_by_confidence: float | None = 0.95,
    filter_test: bool = True,
):
    if validation_files is None:
        validation_files = []

    pseudolabels = np.load(pseudolabels_cache_file, allow_pickle=True)["v"]
    filenames = pseudolabels[0]

    mask = np.ones(pseudolabels.shape[1], dtype=np.bool)

    # Filter out test/validation
    if filter_test:
        test_filenames = [
            "train_audio/" + os.path.basename(fn) for fn in validation_files
        ]
        test_filenames += [
            "train_soundscapes/" + fn
            for fn in pd.read_csv(DATA_DIR + "/train_soundscapes_labels.csv")[
                "filename"
            ].unique()
        ]

        is_test = np.isin(filenames, test_filenames)
        mask &= ~is_test

    if filter_by_confidence is not None:
        has_confident_preds = np.any(pseudolabels[3:] > filter_by_confidence, axis=0)
        mask &= has_confident_preds

    pseudolabels = pseudolabels.T[mask]
    return pseudolabels


class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = load_audio_chunk(
            row["filepath"],
            mode="random" if self.augment else "center",
            # mode="start",
        )
        label = self._make_label(row)
        return audio, label

    def _make_label(self, row) -> torch.Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[int(row["label_idx"])] = 1.0
        if "secondary_idx" in row.index and isinstance(row["secondary_idx"], list):
            for idx in row["secondary_idx"]:
                label[int(idx)] = 1.0
        return label


class PseudolabelsDataset(Dataset):
    def __init__(self):
        pseudolabels = load_pseudolabels(filter_test=True, filter_by_confidence=0.8)

        self.pseudolabels = pd.DataFrame(
            pseudolabels, columns=["filename", "start_time", "end_time"] + ALL_CLASSES
        )

    def __len__(self):
        return len(self.pseudolabels)

    def __getitem__(self, idx: int):
        row = self.pseudolabels.iloc[idx]
        sr = SAMPLE_RATE
        duration = row["end_time"] - row["start_time"]

        audio = torchaudio.load(
            DATA_DIR + "/" + row["filename"],
            frame_offset=row["start_time"] * sr,
            num_frames=duration * sr,
        )[0].mean(dim=0)
        assert audio.shape == (duration * sr,)

        def map_prob_to_label(prob: float) -> float:
            if prob > 0.95:
                return 0.95
            elif prob > 0.7:
                return 0.7
            else:
                return 0.1

        label = torch.tensor([map_prob_to_label(row[cl]) for cl in ALL_CLASSES])
        return audio, label


class SoundscapeDataset(Dataset):
    def __init__(
        self,
        filepath: str | list[str],
        chunk_duration: float = 5,
        sr: int = SAMPLE_RATE,
    ):
        if isinstance(filepath, str):
            filepaths = [filepath]
        else:
            filepaths = filepath

        chunksize = chunk_duration * sr

        items = []

        for filepath in filepaths:
            stem = os.path.splitext(os.path.basename(filepath))[0]
            info = sf.info(filepath)
            n_chunks = info.frames // chunksize

            for i in range(n_chunks):
                items.append(
                    (
                        stem,
                        filepath,
                        i * chunksize,
                        (i + 1) * chunksize,
                        (i + 1) * chunk_duration,
                    )
                )

        self.items = items

        self.cached_audio = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        stem, filepath, start_idx, end_idx, end_time = self.items[idx]

        if self.cached_audio is None or self.cached_audio[0] != filepath:
            audio = torchaudio.load(filepath)[0].mean(dim=0)
            self.cached_audio = (filepath, audio)

        audio = self.cached_audio[1]
        audio = audio[start_idx:end_idx]

        row_id = f"{stem}_{end_time}"
        return audio, row_id


class ChunkedFilesDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        chunk_duration: int = DURATION,
        sr: int = SAMPLE_RATE,
    ):
        chunksize = chunk_duration * sr

        items = []

        for i, filepath in enumerate(files):
            info = sf.info(filepath)
            n_chunks = info.frames // chunksize

            for i in range(n_chunks):
                items.append(
                    (
                        i,
                        filepath,
                        i * chunksize,
                        (i + 1) * chunksize,
                        i * chunk_duration,
                        (i + 1) * chunk_duration,
                    )
                )

        self.items = items
        self.cached_audio = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        i, filepath, start_idx, end_idx, start_time, end_time = self.items[idx]

        if self.cached_audio is None or self.cached_audio[0] != filepath:
            audio = torchaudio.load(filepath)[0].mean(dim=0)
            self.cached_audio = (filepath, audio)

        audio = self.cached_audio[1]
        audio = audio[start_idx:end_idx]

        return i, start_time, end_time, audio
