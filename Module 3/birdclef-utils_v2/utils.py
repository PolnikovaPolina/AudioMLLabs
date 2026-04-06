import math
import random
import os
import numpy as np
import librosa
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import (
    SAMPLE_RATE, DURATION,
    N_MELS, N_FFT, HOP_LENGTH, FMAX,
    AT_N_MELS, AT_N_FFT, AT_HOP_LENGTH, AT_FMIN, AT_FMAX,
    SEED,
)

# ─── Відтворюваність ──────────────────────────────────────────────────────
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


# ─── Аудіо ────────────────────────────────────────────────────────────────
def load_audio_chunk(filepath: str,
                     duration: int = DURATION,
                     sr: int = SAMPLE_RATE,
                     mode: str = "random") -> np.ndarray:
    """
    mode:
        'random' — випадковий відрізок (тренування)
        'center' — центральний відрізок (валідація / інференс)
        'start'  — початок файлу
    """
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
    except Exception:
        return np.zeros(sr * duration, dtype=np.float32)

    target = sr * duration
    n = len(audio)

    if n < target:
        audio = np.tile(audio, math.ceil(target / n))[:target]
    elif mode == "random":
        start = random.randint(0, n - target)
        audio = audio[start: start + target]
    elif mode == "center":
        start = (n - target) // 2
        audio = audio[start: start + target]
    else:
        audio = audio[:target]

    return audio.astype(np.float32)


def augment_audio(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Gaussian noise + випадкове підсилення гучності."""
    audio = audio + noise_level * np.random.randn(len(audio)).astype(np.float32)
    gain = random.uniform(0.8, 1.2)
    return audio * gain


# ─── Мел-спектрограми ─────────────────────────────────────────────────────
def audio_to_melspec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Підходи 1, 2: Log-Mel + min-max нормалізація → [0, 1].
    Повертає float32 (N_MELS, T).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.astype(np.float32)


def audio_to_melspec_at(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Підхід 3: Log-Mel + per-frequency Instance Norm.
    Краще підходить для audio-pretrained backbone-ів.
    Повертає float32 (AT_N_MELS, T).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=AT_N_MELS, n_fft=AT_N_FFT,
        hop_length=AT_HOP_LENGTH,
        fmin=AT_FMIN, fmax=AT_FMAX,
    )
    mel = librosa.power_to_db(mel, ref=1.0)
    mean = mel.mean(axis=1, keepdims=True)
    std  = mel.std(axis=1,  keepdims=True) + 1e-8
    mel  = (mel - mean) / std
    return mel.astype(np.float32)


def mel_to_tensor(mel: np.ndarray) -> torch.Tensor:
    """(F, T) → (3, F, T): дублюємо канал для ImageNet backbone."""
    return torch.tensor(mel).unsqueeze(0).repeat(3, 1, 1)


# ─── Метрики ──────────────────────────────────────────────────────────────
def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged ROC-AUC (офіційна метрика BirdCLEF 2026).
    Пропускає класи без жодного позитивного прикладу.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


def calculate_map(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro mean Average Precision (mAP)."""
    aps = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aps)) if aps else 0.0


def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        threshold: float = 0.5) -> float:
    """Macro F1-Score з бінаризацією по порогу."""
    y_bin = (y_pred >= threshold).astype(int)
    f1s = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            f1s.append(f1_score(y_true[:, i], y_bin[:, i], zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def calculate_top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                               k: int = 3) -> float:
    """Top-K Accuracy: чи є хоча б один правильний клас серед топ-K передбачень."""
    correct = 0
    total = y_true.shape[0]
    for i in range(total):
        top_k = np.argsort(y_pred[i])[-k:]
        true_idx = np.where(y_true[i] == 1)[0]
        if len(set(top_k).intersection(set(true_idx))) > 0:
            correct += 1
    return correct / total if total > 0 else 0.0
