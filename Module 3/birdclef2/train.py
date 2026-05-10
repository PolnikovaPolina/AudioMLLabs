from dotenv import load_dotenv

from debug import debugger

from metrics import (
    competition_metric,
    calculate_macro_f1,
    calculate_map,
    calculate_top_k_accuracy,
)

load_dotenv()

import os
import argparse
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn

from config import (
    ALL_CLASSES,
    TRAIN_CSV,
    TAXONOMY_CSV,
    AUDIO_DIR,
    DATA_DIR,
    FOLDS_CSV,
    SEED,
    Config,
)
from utils import seed_everything
from soundscapes_evaluator import (
    TrainSoundscapesEpochEvaluator,
    TrainSoundscapesEvaluator,
)
from datasets import BirdDataset
from models import build_model


import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch import Trainer

from sklearn.model_selection import StratifiedGroupKFold

from loss import FocalLoss

from torchaudio.transforms import SpecAugment


def prepare_dataframes(debug: bool = False) -> tuple[pd.DataFrame, list]:
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    all_classes = sorted(taxonomy_df["primary_label"].unique().tolist())
    class2idx = {c: i for i, c in enumerate(all_classes)}

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filepath"] = train_df["filename"].apply(
        lambda x: os.path.join(AUDIO_DIR, x)
    )
    train_df["label_idx"] = train_df["primary_label"].map(class2idx)

    # Secondary labels
    def parse_secondary(val):
        try:
            labels = ast.literal_eval(val)
            return [class2idx[l] for l in labels if l in class2idx]
        except Exception:
            return []

    col = "secondary_labels" if "secondary_labels" in train_df.columns else None
    train_df["secondary_idx"] = (
        train_df[col].apply(parse_secondary)
        if col
        else [[] for _ in range(len(train_df))]
    )

    # Завантажуємо готові фолди або будуємо нові
    if os.path.exists(FOLDS_CSV):
        folds_df = pd.read_csv(FOLDS_CSV)[["filename", "fold"]]
        train_df = train_df.merge(folds_df, on="filename")
        print(f"Фолди завантажено з {FOLDS_CSV}")
    else:
        # TODO: refactor
        # import soundfile as sf

        # train_df["duration"] = train_df["filepath"].map(lambda p: sf.info(p).duration)
        # train_df = train_df.loc[train_df["duration"] > 2.5]

        train_df = _build_group_folds(train_df)
        # Зберігаємо на майбутнє
        train_df[["filename", "fold"]].to_csv(FOLDS_CSV, index=False)
        print(f"Фолди збережено в {FOLDS_CSV}")

    if debug:
        train_df = train_df.groupby("primary_label").head(1).head(200)
        print(f"[DEBUG] Використовуємо {len(train_df)} зразків")

    print(f"Всього зразків: {len(train_df)} | Класів: {len(all_classes)}")
    return train_df, all_classes


def _build_group_folds(df: pd.DataFrame) -> pd.DataFrame:
    # Визначаємо групу: author > XC-ID (перша частина filename) > filename
    if "author" in df.columns and df["author"].notna().mean() > 0.5:
        groups = df["author"].fillna("unknown").values
        print("GroupKFold: групуємо по 'author'")
    else:
        # XC-ID = перша частина шляху типу 'XC12345/file.ogg'
        groups = df["filename"].apply(lambda x: x.split("/")[0]).values
        print("GroupKFold: групуємо по XC-ID (перша частина filename)")

    N_FOLDS = 5
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df = df.copy()
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["primary_label"], groups)):
        df.loc[df.index[val_idx], "fold"] = fold

    # Логуємо розподіл
    print("\n--- StratifiedGroupKFold розподіл ---")
    for f, cnt in df.groupby("fold").size().items():
        print(f"  Fold {f}: {cnt} зразків ({cnt / len(df) * 100:.1f}%)")
    print(
        f"  Унікальних класів per fold: {df.groupby('fold')['primary_label'].nunique().to_dict()}"
    )
    print()

    return df


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_limit: int | None = None,
    ):
        super().__init__()

        train_df, _ = prepare_dataframes()
        fold = 0

        # import soundfile as sf
        # from config import DURATION

        # train_df = train_df[
        #     train_df["filepath"].apply(lambda filepath: os.path.exists(filepath))
        # ]
        # train_df["duration"] = train_df["filepath"].apply(
        #     lambda filepath: sf.info(filepath).duration
        # )
        # train_df = train_df[
        #     (train_df["duration"] <= DURATION) & (train_df["duration"] > 2)
        # ]

        train_fold = train_df[train_df["fold"] != fold].copy()
        val_fold = train_df[train_df["fold"] == fold].copy()
        print(f"  Train: {len(train_fold)}  |  Val: {len(val_fold)}")

        if train_limit is not None:
            train_fold = train_fold.iloc[:train_limit]

        self.train_primary_labels = train_fold["primary_label"].unique().tolist()
        self.train_ds = BirdDataset(train_fold, augment=True)
        self.val_ds = BirdDataset(val_fold, augment=False)
        self.batch_size = batch_size
        # self.pseudolabels_ds = PseudolabelsDataset()

        self.soundscapes_evaluator = TrainSoundscapesEvaluator(DATA_DIR)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_audio_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=16,
            shuffle=False,
            pin_memory=True,
        )
        val_soundscapes_dl = DataLoader(
            self.soundscapes_evaluator.get_dataset(),
            batch_size=self.batch_size,
            num_workers=16,
            shuffle=False,
            pin_memory=True,
        )
        return [val_audio_dl, val_soundscapes_dl]


def mixup_batch(x, y, alpha=0.8):
    if alpha > 0:
        weight = np.random.beta(alpha, alpha)
    else:
        weight = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = weight * x + (1 - weight) * x[index, :]
    mixed_y = weight * y + (1 - weight) * y[index, :]
    return mixed_x, mixed_y


class TrainingModule(L.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = Config.model_validate(kwargs, extra="ignore")

        self.spec_augment = SpecAugment(
            n_time_masks=3, time_mask_param=20, n_freq_masks=3, freq_mask_param=10
        )
        self.model = build_model(config.model)
        self.feature_extractor = self.model.make_feature_extractor()

        if config.loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif config.loss == "focal":
            self.criterion = FocalLoss(alpha=self.hparams.focal_loss_alpha, gamma=2.0)
        else:
            raise ValueError(config.loss)

        self.all_preds = []
        self.all_labels = []

        self.soundscapes_eval_epoch: TrainSoundscapesEpochEvaluator | None = None

    def setup(self, stage: str) -> None:
        if self.soundscapes_eval_epoch is None:
            datamodule: DataModule = self.trainer.datamodule
            self.soundscapes_eval_epoch = (
                datamodule.soundscapes_evaluator.get_epoch_evaluator()
            )

    def _extract_features(self, x):
        with torch.autocast(device_type=self.device.type, enabled=False):
            x = self.feature_extractor(x.to(torch.float32))

        assert x.isfinite().all()
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # smoothing_factor = 0.1train_fold['duration']
        # y = y * (1 - smoothing_factor) + smoothing_factor / y.size(1)

        if (
            self.hparams.mixup is not None
            and np.random.rand() < self.hparams.mixup["prob"]
        ):
            x, y = mixup_batch(x, y, self.hparams.mixup["alpha"])

        mel = self._extract_features(x)
        if np.random.rand() < self.hparams.specaugment_prob:
            mel = self.spec_augment(mel)

        logits = self.model(mel)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss)
        return loss

    def on_before_optimizer_step(self, optimizer):
        from lightning.pytorch.utilities import grad_norm

        norms = grad_norm(self.model, norm_type=2)
        self.log("grad_2.0_norm_total", norms["grad_2.0_norm_total"])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch

            logits = self.model(self._extract_features(x))
            loss = self.criterion(logits, y)
            self.log(
                "val/loss", loss, add_dataloader_idx=False, on_step=False, on_epoch=True
            )

            probs = torch.sigmoid(logits).cpu().numpy()

            self.all_preds.append(probs)
            self.all_labels.append(y.cpu().numpy())
        elif dataloader_idx == 1:
            x, row_ids = batch
            logits = self.model(self._extract_features(x))
            probs = torch.sigmoid(logits).cpu().numpy()
            assert self.soundscapes_eval_epoch is not None

            for row_id, prob in zip(row_ids, probs):
                self.soundscapes_eval_epoch.add_preds(row_id, prob.tolist())

    def on_validation_epoch_end(self) -> None:
        preds = np.vstack(self.all_preds)
        labels = np.vstack(self.all_labels)
        self.all_preds.clear()
        self.all_labels.clear()

        self.log("val/aucrog", competition_metric(labels, preds))
        self.log("val/mAP", calculate_map(labels, preds))
        self.log(
            "val/f1",
            calculate_macro_f1(labels, preds, threshold=self.hparams.pred_threshold),
        )
        self.log("val/topk", calculate_top_k_accuracy(labels, preds, k=self.hparams.k))

        assert self.soundscapes_eval_epoch is not None

        soundscapes_metrics = self.soundscapes_eval_epoch.compute_metrics()
        soundscapes_trained_metrics = self.soundscapes_eval_epoch.compute_metrics(
            exclude_labels=[
                cl
                for cl in ALL_CLASSES
                if cl not in self.trainer.datamodule.train_primary_labels
            ]
        )
        self.soundscapes_eval_epoch.reset()

        def log_soundscapes_metrics(setname, metrics):
            def map_metric_name(name):
                prefix = "val/" + setname

                if name == "aucroc_macro":
                    return prefix

                return prefix + "/" + name

            self.log_dict({map_metric_name(k): v for k, v in metrics.items()})

        log_soundscapes_metrics("soundscapes", soundscapes_metrics)
        log_soundscapes_metrics("soundscapes_trained", soundscapes_trained_metrics)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_pct,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def run_training(
    run_name: str, config: Config, logging: bool = False, debug: bool = False
):
    if debug:
        debugger.enable()

    data_module = DataModule(
        batch_size=config.batch_size, train_limit=config.train_limit
    )

    training_module = TrainingModule(**config.model_dump())

    if not debug:
        early_stopping = EarlyStopping(
            monitor="val/soundscapes", mode="max", patience=5, verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",
            filename=run_name + "-{epoch}-{val/loss:.3f}-{val/soundscapes:.3f}",
            monitor="val/soundscapes",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=2,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    else:
        callbacks = []

    if logging:
        logger = WandbLogger(project="birdclef", name=run_name)
    else:
        logger = None

    trainer_args = {}

    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        precision="16-mixed",
        gradient_clip_val=config.grad_clip,
        **trainer_args,
    )

    trainer.fit(training_module, data_module)

    debugger.on_exit()


def load_config(config_path: str) -> Config:
    import yaml

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data, extra="forbid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    args = parser.parse_args()

    seed_everything()

    config = load_config(args.config)

    run_training(run_name=args.name, config=config, logging=args.log, debug=args.debug)
