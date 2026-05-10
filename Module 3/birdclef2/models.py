import torch
import torch.nn as nn
import timm

from config import (
    ModelConfig,
    NUM_CLASSES,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    FMAX,
    HOP_LENGTH,
)

from torchaudio.transforms import (
    MelSpectrogram,
    AmplitudeToDB,
)
from torchvision.transforms import Normalize

from debug import debugger


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            f_max=FMAX,
        )

        self.top_db = 80.0
        self.to_db = AmplitudeToDB(stype="power", top_db=self.top_db)
        self.mel_normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x = self.mel_transform(x)
        x = self.to_db(x.unsqueeze(-3)).squeeze(-3)

        x = x / self.top_db + 1

        x = x.unsqueeze(-3).expand(*x.shape[:-2], 3, *x.shape[-2:])
        x = self.mel_normalize(x)
        return x


class ModelBase(nn.Module):
    def make_feature_extractor(self) -> FeatureExtractor:
        raise NotImplementedError


class CNNBaseline(ModelBase):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            drop_rate=dropout,
            global_pool="avg",
        )

    def make_feature_extractor(self) -> FeatureExtractor:
        return FeatureExtractor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SEDModel(ModelBase):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )

        in_features = self.backbone.num_features

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)
        self.attention = nn.Linear(in_features, num_classes)

    def make_feature_extractor(self) -> FeatureExtractor:
        return FeatureExtractor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        # Frequency pooling
        x = x.mean(dim=-2)
        x = x.transpose(1, 2)

        frame_preds = self.classifier(x)  # (Batch, Time, Classes)

        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        preds = torch.sum(frame_preds * weights, dim=1)
        return preds


def build_model(config: ModelConfig) -> ModelBase:
    if config.model == "cnn":
        return CNNBaseline(
            backbone=config.backbone,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    elif config.model == "sed":
        return SEDModel(
            backbone=config.backbone,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    else:
        raise ValueError(config.model)
