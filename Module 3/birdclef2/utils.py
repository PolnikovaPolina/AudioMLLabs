import random
import os
import numpy as np
import torch
import torchaudio

from config import (
    SAMPLE_RATE,
    DURATION,
    SEED,
)


def seed_everything(seed: int = SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loop_pad_audio(audio: torch.Tensor, target: int) -> torch.Tensor:
    n = audio.size(-1)

    if n >= target:
        return audio[..., :target]

    reps = (target + n - 1) // n
    repeat_pattern = [1] * (audio.ndim - 1) + [reps]
    return audio.repeat(*repeat_pattern)[..., :target]


def energy_crop(audio, sr, crop_len=5, n_candidates=10):
    crop_samples = crop_len * sr
    candidates = np.random.randint(0, audio.shape[0] - crop_samples, n_candidates)
    energies = [torch.mean(audio[s : s + crop_samples] ** 2).item() for s in candidates]
    best_start = candidates[np.argmax(energies)]
    return audio[best_start : best_start + crop_samples]


def load_audio_chunk(
    filepath: str, duration: int = DURATION, sr: int = SAMPLE_RATE, mode: str = "random"
) -> torch.Tensor:
    audio, _ = torchaudio.load(filepath)
    audio = audio.mean(dim=0)

    target = sr * duration
    n = audio.size(-1)

    if n < target:
        audio = loop_pad_audio(audio, target)
    elif n == target:
        pass
    elif mode == "energy":
        audio = energy_crop(audio, sr)
    elif mode == "random":
        start = random.randint(0, n - target)
        audio = audio[start : start + target]
    elif mode == "center":
        start = (n - target) // 2
        audio = audio[start : start + target]
    elif mode == "start":
        audio = audio[:target]
    else:
        raise ValueError(mode)

    return audio
