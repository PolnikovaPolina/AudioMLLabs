import os
import pandas as pd

# ─── Шляхи ────────────────────────────────────────────────────────────────
DATA_DIR        = "/kaggle/input/competitions/birdclef-2026"
AUDIO_DIR       = os.path.join(DATA_DIR, "train_audio")
TRAIN_CSV       = os.path.join(DATA_DIR, "train.csv")
TAXONOMY_CSV    = os.path.join(DATA_DIR, "taxonomy.csv")
SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "test_soundscapes")
SAMPLE_SUB_CSV  = os.path.join(DATA_DIR, "sample_submission.csv")

FOLDS_CSV       = "/kaggle/working/train_with_folds.csv"
OUTPUT_DIR      = "/kaggle/working"

# ─── Класи ────────────────────────────────────────────────────────────────
try:
    _taxonomy   = pd.read_csv(TAXONOMY_CSV)
    ALL_CLASSES = sorted(_taxonomy["primary_label"].unique().tolist())
    NUM_CLASSES = len(ALL_CLASSES)
    CLASS2IDX   = {c: i for i, c in enumerate(ALL_CLASSES)}
except Exception:
    NUM_CLASSES = 234
    ALL_CLASSES = []
    CLASS2IDX   = {}

# ─── Аудіо ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
DURATION    = 5
N_SAMPLES   = SAMPLE_RATE * DURATION

# ─── Мел-спектрограма (Підходи 1, 2) ─────────────────────────────────────
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
FMAX       = 16_000

# ─── Мел-спектрограма (Підхід 3 — per-frequency norm) ────────────────────
AT_N_MELS     = 128
AT_N_FFT      = 2048
AT_HOP_LENGTH = 512
AT_FMIN       = 50
AT_FMAX       = 14_000

# ─── SpecAugment ──────────────────────────────────────────────────────────
SPEC_FREQ_MASK = 20
SPEC_TIME_MASK = 40
SPEC_NUM_MASKS = 2

# ─── Навчання ─────────────────────────────────────────────────────────────
SEED        = 42
N_FOLDS     = 5
BATCH_SIZE  = 32
N_EPOCHS    = 10
LR          = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP   = 1.0

# ─── Fine-tuning (Підхід 3) ───────────────────────────────────────────────
WARMUP_EPOCHS = 3       # Stage 1: тільки голова
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4

# ─── Конфіги моделей ──────────────────────────────────────────────────────
MODEL_CONFIGS = {
    1: {
        "name":        "CNN_Baseline",
        "backbone":    "efficientnet_b0",
        "pretrained":  True,
        "dropout":     0.3,
        "type":        "base",
        "description": "EfficientNet-B0 + Mel Spectrogram (baseline)",
    },
    2: {
        "name":        "CNN_GeoFusion",
        "backbone":    "efficientnet_b0",
        "pretrained":  True,
        "dropout":     0.3,
        "geo_dim":     64,
        "type":        "geo",
        "description": "EfficientNet-B0 + Geographic Metadata fusion",
    },
    3: {
        "name":        "AudioPretrained_B2",
        "backbone":    "efficientnet_b2",
        "pretrained":  True,
        "dropout":     0.4,
        "type":        "at",
        "description": "EfficientNet-B2 with 2-stage fine-tuning",
    },
}
