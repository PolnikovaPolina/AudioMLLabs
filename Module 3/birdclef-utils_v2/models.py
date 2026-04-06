import random
import torch
import torch.nn as nn
import timm

from config import (
    NUM_CLASSES,
    SPEC_FREQ_MASK, SPEC_TIME_MASK, SPEC_NUM_MASKS,
    MODEL_CONFIGS,
)


# ─── SpecAugment ──────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    """
    Маскує випадкові смуги по частотній та часовій осях.
    Застосовується тільки під час тренування (self.training=True).
    Джерело: Park et al., 2019.
    """

    def __init__(self,
                 freq_mask: int = SPEC_FREQ_MASK,
                 time_mask: int = SPEC_TIME_MASK,
                 num_masks: int = SPEC_NUM_MASKS):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, F, T = x.shape
        out = x.clone()
        for _ in range(self.num_masks):
            f0 = random.randint(0, max(0, F - self.freq_mask))
            out[:, :, f0: f0 + self.freq_mask, :] = 0.0
            t0 = random.randint(0, max(0, T - self.time_mask))
            out[:, :, :, t0: t0 + self.time_mask] = 0.0
        return out


# ─── Підхід 1: CNN Baseline ───────────────────────────────────────────────
class CNNBaseline(nn.Module):
    """
    EfficientNet-B0 на Log-Mel Spectrogram.

    Вхід:  (B, 3, 128, T)
    Вихід: (B, NUM_CLASSES)  — logits (без sigmoid)
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 backbone: str = "efficientnet_b0",
                 pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.spec_aug = SpecAugment()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spec_aug(x)
        x = self.backbone(x)
        return self.head(x)


# ─── Підхід 2: CNN + Geographic Metadata ─────────────────────────────────
class CNNWithGeo(nn.Module):
    """
    EfficientNet-B0 + Geographic MLP fusion.

    Архітектура:
        audio      → CNN backbone → feat_audio (1280-dim)
        [lat, lon] → GeoMLP      → feat_geo   (geo_dim-dim)
        concat([feat_audio, feat_geo]) → Classifier → logits

    Вхід:  mel (B, 3, 128, T),  geo (B, 2)
    Вихід: (B, NUM_CLASSES)
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 backbone: str = "efficientnet_b0",
                 pretrained: bool = True,
                 geo_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.spec_aug = SpecAugment()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        audio_feat = self.backbone.num_features   # 1280 для B0

        self.geo_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, geo_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_feat + geo_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, mel: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        mel   = self.spec_aug(mel)
        f_aud = self.backbone(mel)
        f_geo = self.geo_mlp(geo)
        return self.head(torch.cat([f_aud, f_geo], dim=1))


# ─── Підхід 3: Audio-Pretrained EfficientNet-B2 ───────────────────────────
class AudioPretrainedModel(nn.Module):
    """
    EfficientNet-B2 з двоетапним fine-tuning.

    Stage 1 — backbone заморожений, тренується тільки голова (великий LR).
    Stage 2 — весь backbone розморожується (backbone LR = lr_head / 10).

    Голова: LayerNorm → Dropout → Linear(512) → GELU → Linear(num_classes).
    LayerNorm стабілізує градієнти при розмороженні backbone.

    Вхід:  (B, 3, 128, T)
    Вихід: (B, NUM_CLASSES)
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 backbone: str = "efficientnet_b2",
                 pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        # Більш агресивний SpecAugment для сильнішого backbone
        self.spec_aug = SpecAugment(freq_mask=24, time_mask=64, num_masks=2)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool="avg",
        )
        in_feat = self.backbone.num_features   # 1408 для B2

        self.head = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Dropout(dropout),
            nn.Linear(in_feat, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Stage 1] Backbone заморожений. Trainable params: {n:,}")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Stage 2] Backbone розморожений. Trainable params: {n:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spec_aug(x)
        x = self.backbone(x)
        return self.head(x)


# ─── Фабрика ──────────────────────────────────────────────────────────────
def build_model(model_id: int) -> nn.Module:
    """
    Повертає ненавчену модель за її номером (1, 2 або 3).
    Конфіги читаються з config.py → MODEL_CONFIGS.
    """
    cfg = MODEL_CONFIGS[model_id]
    if model_id == 1:
        return CNNBaseline(
            backbone=cfg["backbone"],
            pretrained=cfg["pretrained"],
            dropout=cfg["dropout"],
        )
    elif model_id == 2:
        return CNNWithGeo(
            backbone=cfg["backbone"],
            pretrained=cfg["pretrained"],
            geo_dim=cfg["geo_dim"],
            dropout=cfg["dropout"],
        )
    elif model_id == 3:
        return AudioPretrainedModel(
            backbone=cfg["backbone"],
            pretrained=cfg["pretrained"],
            dropout=cfg["dropout"],
        )
    else:
        raise ValueError(f"Невідомий model_id={model_id}. Доступні: 1, 2, 3.")
