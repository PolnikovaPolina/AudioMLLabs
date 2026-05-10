import os
import pandas as pd
from pydantic import BaseModel

# ─── Шляхи ────────────────────────────────────────────────────────────────
DATA_DIR = os.environ["BIRDCLEF_DATA"]
AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TAXONOMY_CSV = os.path.join(DATA_DIR, "taxonomy.csv")
SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "train_soundscapes")

OUTPUT_DIR = "output"
FOLDS_CSV = OUTPUT_DIR + "/train_with_folds.csv"

_taxonomy = pd.read_csv(TAXONOMY_CSV)
ALL_CLASSES = sorted(_taxonomy["primary_label"].unique().tolist())
NUM_CLASSES = len(ALL_CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

# ─── Аудіо ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
DURATION = 5
N_SAMPLES = SAMPLE_RATE * DURATION

# ─── Мел-спектрограма (Підходи 1, 2) ─────────────────────────────────────
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 16_000

# ─── Навчання ─────────────────────────────────────────────────────────────
SEED = 42


class ModelConfig(BaseModel):
    model: str = "cnn"
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0


class MixupConfig(BaseModel):
    prob: float = 0.5
    alpha: float = 0.8


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    max_epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    train_limit: int | None = None
    pred_threshold: float = 0.5
    weight_decay: float = 1e-4
    focal_loss_alpha: float = 0.25
    mixup: MixupConfig | None = None
    specaugment_prob: float = 0.5
    loss: str = "bce"
    k: int = 3
    grad_clip: float = 1.0
    warmup_pct: float = 0.2
