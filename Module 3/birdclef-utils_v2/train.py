import os
import sys
import argparse
import gc
import ast
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    TRAIN_CSV, TAXONOMY_CSV, AUDIO_DIR, FOLDS_CSV, OUTPUT_DIR,
    BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, SEED, N_FOLDS,
    MODEL_CONFIGS, WARMUP_EPOCHS, LR_HEAD, LR_BACKBONE,
)
from utils import (
    seed_everything, competition_metric,
    calculate_map, calculate_macro_f1, calculate_top_k_accuracy,
)
from datasets import BirdDataset, BirdDatasetWithGeo, BirdDatasetAT
from models import build_model


# ─── Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss для multi-label класифікації.
    Знижує вагу добре класифікованих прикладів → фокусується на складних.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ─── MixUp ────────────────────────────────────────────────────────────────
def mixup_data(mel, label, alpha: float = 0.5, geo=None):
    """
    MixUp augmentation.
    Якщо передано geo — змішуємо і його.
    Повертає (mixed_mel, mixed_label) або (mixed_mel, mixed_geo, mixed_label).
    """
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(mel.size(0), device=mel.device)
    m_mel = lam * mel + (1 - lam) * mel[idx]
    m_lbl = lam * label + (1 - lam) * label[idx]
    if geo is not None:
        m_geo = lam * geo + (1 - lam) * geo[idx]
        return m_mel, m_geo, m_lbl
    return m_mel, m_lbl


# ─── Підготовка даних ─────────────────────────────────────────────────────
def prepare_dataframes(debug: bool = False) -> tuple[pd.DataFrame, list]:
    """
    Завантажує метадані, будує фолди через StratifiedGroupKFold.

    Чому GroupKFold:
        Один автор може мати десятки записів з однієї локації.
        Звичайний StratifiedKFold розкидає їх між train і val →
        модель «бачила» схожий акустичний фон і отримує завищений AUC.
        GroupKFold гарантує, що всі записи одного автора потрапляють
        лише в один split.

    Група = колонка 'author' (якщо є) або перша частина filename до '/'.
    """
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    all_classes = sorted(taxonomy_df["primary_label"].unique().tolist())
    class2idx   = {c: i for i, c in enumerate(all_classes)}

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filepath"]  = train_df["filename"].apply(lambda x: os.path.join(AUDIO_DIR, x))
    train_df["label_idx"] = train_df["primary_label"].map(class2idx)

    # Secondary labels
    def parse_secondary(val):
        try:
            labels = ast.literal_eval(val)
            return [class2idx[l] for l in labels if l in class2idx]
        except Exception:
            return []

    col = "secondary_labels" if "secondary_labels" in train_df.columns else None
    train_df["secondary_idx"] = train_df[col].apply(parse_secondary) if col else [[] for _ in range(len(train_df))]

    # Завантажуємо готові фолди або будуємо нові
    if os.path.exists(FOLDS_CSV):
        folds_df = pd.read_csv(FOLDS_CSV)[["filename", "fold"]]
        train_df = train_df.merge(folds_df, on="filename", how="left")
        print(f"Фолди завантажено з {FOLDS_CSV}")
    else:
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
    """
    Будує фолди через StratifiedGroupKFold (sklearn >= 0.24).

    Стратифікація по primary_label гарантує рівномірний розподіл класів.
    Групування по author (або XC-ID) не дає одному автору потрапити
    одночасно в train і val.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    # Визначаємо групу: author > XC-ID (перша частина filename) > filename
    if "author" in df.columns and df["author"].notna().mean() > 0.5:
        groups = df["author"].fillna("unknown").values
        print("GroupKFold: групуємо по 'author'")
    else:
        # XC-ID = перша частина шляху типу 'XC12345/file.ogg'
        groups = df["filename"].apply(lambda x: x.split("/")[0]).values
        print("GroupKFold: групуємо по XC-ID (перша частина filename)")

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df     = df.copy()
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, df["primary_label"], groups)
    ):
        df.loc[df.index[val_idx], "fold"] = fold

    # Логуємо розподіл
    print("\n--- StratifiedGroupKFold розподіл ---")
    for f, cnt in df.groupby("fold").size().items():
        print(f"  Fold {f}: {cnt} зразків ({cnt / len(df) * 100:.1f}%)")
    print(f"  Унікальних класів per fold: {df.groupby('fold')['primary_label'].nunique().to_dict()}")
    print()

    return df


# ─── DataLoaders ──────────────────────────────────────────────────────────
_DATASET_MAP = {1: BirdDataset, 2: BirdDatasetWithGeo, 3: BirdDatasetAT}

def get_loaders(model_id: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                batch_size: int) -> tuple[DataLoader, DataLoader]:
    DS = _DATASET_MAP[model_id]
    train_ld = DataLoader(DS(train_df, augment=True),  batch_size=batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(DS(val_df,   augment=False), batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    return train_ld, val_ld


# ─── Train / Validate ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, model_id: int) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        if model_id == 2:
            mel, geo, label = batch
            mel, geo, label = mel.to(device), geo.to(device), label.to(device)
            if np.random.rand() < 0.5:
                mel, geo, label = mixup_data(mel, label, alpha=0.5, geo=geo)
            logits = model(mel, geo)
        else:
            mel, label = batch
            mel, label = mel.to(device), label.to(device)
            if np.random.rand() < 0.5:
                mel, label = mixup_data(mel, label, alpha=0.5)
            logits = model(mel)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * mel.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device,
                   model_id: int) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  val", leave=False):
        if model_id == 2:
            mel, geo, label = batch
            mel, geo, label = mel.to(device), geo.to(device), label.to(device)
            logits = model(mel, geo)
        else:
            mel, label = batch
            mel, label = mel.to(device), label.to(device)
            logits = model(mel)

        total_loss += criterion(logits, label).item() * mel.size(0)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(label.cpu().numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    return (
        total_loss / len(loader.dataset),
        competition_metric(labels, preds),
        calculate_map(labels, preds),
        calculate_macro_f1(labels, preds, threshold=0.5),
        calculate_top_k_accuracy(labels, preds, k=3),
    )


def _log(epoch, total, tr_loss, vl_loss, vl_auc, vl_map, vl_f1, vl_top3,
         flag: str = "", prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Ep {epoch:02d}/{total} | "
          f"Tr: {tr_loss:.4f} | Vl: {vl_loss:.4f} | "
          f"AUC: {vl_auc:.4f} | mAP: {vl_map:.4f} | "
          f"F1: {vl_f1:.4f} | Top3: {vl_top3:.4f}{flag}")


# ─── Головний тренувальний цикл ───────────────────────────────────────────
def run_training(model_id: int, fold: int, epochs: int,
                 batch_size: int, lr: float, debug: bool) -> float:

    cfg    = MODEL_CONFIGS[model_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything()

    print(f"\n{'='*70}")
    print(f"  Підхід {model_id}: {cfg['description']}")
    print(f"  Fold: {fold}  |  Epochs: {epochs}  |  LR: {lr}  |  Device: {device}")
    print(f"{'='*70}")

    train_df, _ = prepare_dataframes(debug=debug)
    train_fold  = train_df[train_df["fold"] != fold].copy()
    val_fold    = train_df[train_df["fold"] == fold].copy()
    print(f"  Train: {len(train_fold)}  |  Val: {len(val_fold)}")

    train_ld, val_ld = get_loaders(model_id, train_fold, val_fold, batch_size)

    model     = build_model(model_id).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    ckpt_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}_fold{fold}.pt")
    history   = []
    best_auc  = 0.0

    # ── Підхід 3: двоетапне тренування ─────────────────────────────────────
    if model_id == 3:
        # Stage 1: тільки голова
        model.freeze_backbone()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS,
                                      eta_min=LR_HEAD * 0.1)

        for epoch in range(1, WARMUP_EPOCHS + 1):
            tr_l = train_epoch(model, train_ld, optimizer, criterion, device, model_id)
            vl_l, auc, mp, f1, t3 = validate_epoch(model, val_ld, criterion, device, model_id)
            scheduler.step()
            flag = ""
            if auc > best_auc:
                best_auc = auc; torch.save(model.state_dict(), ckpt_path); flag = "  ✅"
            history.append(dict(epoch=epoch, stage=1, train_loss=tr_l, val_loss=vl_l,
                                val_auc=auc, val_map=mp, val_f1=f1, val_top3=t3))
            _log(epoch, WARMUP_EPOCHS, tr_l, vl_l, auc, mp, f1, t3, flag, "S1")

        # Stage 2: весь backbone
        model.unfreeze_backbone()
        optimizer = AdamW([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(),     "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)
        remaining = epochs - WARMUP_EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining,
                                      eta_min=LR_BACKBONE * 0.01)

        for epoch in range(WARMUP_EPOCHS + 1, epochs + 1):
            tr_l = train_epoch(model, train_ld, optimizer, criterion, device, model_id)
            vl_l, auc, mp, f1, t3 = validate_epoch(model, val_ld, criterion, device, model_id)
            scheduler.step()
            flag = ""
            if auc > best_auc:
                best_auc = auc; torch.save(model.state_dict(), ckpt_path); flag = "  ✅"
            history.append(dict(epoch=epoch, stage=2, train_loss=tr_l, val_loss=vl_l,
                                val_auc=auc, val_map=mp, val_f1=f1, val_top3=t3))
            _log(epoch, epochs, tr_l, vl_l, auc, mp, f1, t3, flag, "S2")

    # ── Підходи 1 і 2: стандартний цикл ────────────────────────────────────
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        for epoch in range(1, epochs + 1):
            tr_l = train_epoch(model, train_ld, optimizer, criterion, device, model_id)
            vl_l, auc, mp, f1, t3 = validate_epoch(model, val_ld, criterion, device, model_id)
            scheduler.step()
            flag = ""
            if auc > best_auc:
                best_auc = auc; torch.save(model.state_dict(), ckpt_path); flag = "  ✅"
            history.append(dict(epoch=epoch, stage=1, train_loss=tr_l, val_loss=vl_l,
                                val_auc=auc, val_map=mp, val_f1=f1, val_top3=t3))
            _log(epoch, epochs, tr_l, vl_l, auc, mp, f1, t3, flag)

    # Зберігаємо історію
    hist_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}_fold{fold}_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    print(f"\n  Best Val AUC: {best_auc:.4f}")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  History:     {hist_path}")

    gc.collect()
    torch.cuda.empty_cache()
    return best_auc


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training")
    parser.add_argument("--model",  type=int, required=True, choices=[1, 2, 3],
                        help="1=Baseline, 2=GeoFusion, 3=AudioPretrained")
    parser.add_argument("--fold",   type=int, default=0, choices=range(N_FOLDS))
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    parser.add_argument("--debug",  action="store_true",
                        help="200 зразків, 2 епохи — для перевірки пайплайну")
    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    run_training(
        model_id   = args.model,
        fold       = args.fold,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        debug      = args.debug,
    )
