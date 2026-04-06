import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    SOUNDSCAPES_DIR, SAMPLE_SUB_CSV, TAXONOMY_CSV,
    OUTPUT_DIR, BATCH_SIZE, MODEL_CONFIGS,
)
from utils import seed_everything
from datasets import SoundscapeDataset
from models import build_model


# ─── Завантаження checkpoint ──────────────────────────────────────────────
def load_model(model_id: int, fold: int, device: torch.device) -> torch.nn.Module:
    cfg       = MODEL_CONFIGS[model_id]
    ckpt_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}_fold{fold}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint не знайдено: {ckpt_path}\n"
            f"Спочатку запусти: python train.py --model {model_id} --fold {fold}"
        )

    model = build_model(model_id)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    print(f"  ✅ Завантажено: {ckpt_path}")
    return model


# ─── Inference на одному soundscape ──────────────────────────────────────
@torch.no_grad()
def predict_soundscape(filepath: str,
                        models_configs: list,
                        weights: np.ndarray,
                        device: torch.device,
                        n_tta: int = 1) -> dict:
    """
    Ensemble передбачення для одного soundscape файлу.

    models_configs : [(model, model_id), ...]
    weights        : ваги для weighted average, вже нормалізовані (сума=1)
    n_tta          : кількість TTA проходів (1 = без TTA)

    Повертає: {row_id: np.ndarray(NUM_CLASSES)}
    """
    all_model_preds = []
    row_ids_flat    = None

    for model, model_id in models_configs:
        cfg      = MODEL_CONFIGS[model_id]
        mel_type = "at" if cfg["type"] == "at" else "base"

        dataset = SoundscapeDataset(filepath, mel_type=mel_type)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

        tta_preds_list = []

        for _ in range(n_tta):
            tta_preds   = []
            current_ids = []

            for mel, row_ids in loader:
                mel = mel.to(device)

                if cfg["type"] == "geo":
                    # Координати для test soundscapes невідомі → нулі
                    geo    = torch.zeros(mel.shape[0], 2, device=device)
                    logits = model(mel, geo)
                else:
                    logits = model(mel)

                tta_preds.append(torch.sigmoid(logits).cpu().numpy())
                current_ids.extend(row_ids)

            tta_preds_list.append(np.vstack(tta_preds))

            # row_ids зчитуємо лише один раз
            if row_ids_flat is None:
                row_ids_flat = current_ids

        # Середнє по TTA проходах
        model_pred = np.mean(tta_preds_list, axis=0)   # (n_chunks, NUM_CLASSES)
        all_model_preds.append(model_pred)

    # Weighted average по моделях
    ensemble = np.average(all_model_preds, axis=0, weights=weights)

    return {row_id: ensemble[i] for i, row_id in enumerate(row_ids_flat)}


# ─── Генерація submission.csv ─────────────────────────────────────────────
def create_submission(models_configs: list,
                       weights: np.ndarray,
                       device: torch.device,
                       output_path: str,
                       n_tta: int = 1):

    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    all_classes = sorted(taxonomy_df["primary_label"].unique().tolist())
    sample_sub  = pd.read_csv(SAMPLE_SUB_CSV)

    soundscape_files = sorted([
        f for f in os.listdir(SOUNDSCAPES_DIR) if f.endswith(".ogg")
    ])
    print(f"\nЗнайдено soundscapes: {len(soundscape_files)}")
    print(f"TTA проходів: {n_tta}")

    all_predictions = {}
    for fname in tqdm(soundscape_files, desc="Inference"):
        filepath = os.path.join(SOUNDSCAPES_DIR, fname)
        preds    = predict_soundscape(filepath, models_configs, weights, device, n_tta)
        all_predictions.update(preds)

    records = []
    for row_id in sample_sub["row_id"]:
        probs = all_predictions.get(row_id, np.zeros(len(all_classes)))
        rec   = {"row_id": row_id}
        rec.update({cls: float(probs[i]) for i, cls in enumerate(all_classes)})
        records.append(rec)

    submission_df = pd.DataFrame(records)
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission збережено: {output_path}")
    print(f"   Рядків: {len(submission_df)} | Колонок: {len(submission_df.columns)}")
    return submission_df


# ─── Таблиця результатів з history CSV ───────────────────────────────────
def print_results_table(model_ids: list, fold: int):
    print(f"\n{'─'*60}")
    print(f"  Val results (fold {fold})")
    print(f"{'─'*60}")
    rows = []
    for mid in model_ids:
        cfg       = MODEL_CONFIGS[mid]
        hist_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}_fold{fold}_history.csv")
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            best = hist.loc[hist["val_auc"].idxmax()]
            rows.append({
                "Model":      cfg["name"],
                "Best AUC":   f"{best['val_auc']:.4f}",
                "mAP":        f"{best.get('val_map', 0):.4f}",
                "F1":         f"{best.get('val_f1', 0):.4f}",
                "Top3":       f"{best.get('val_top3', 0):.4f}",
                "Best Epoch": int(best["epoch"]),
            })
        else:
            rows.append({"Model": cfg["name"], "Best AUC": "—",
                         "mAP": "—", "F1": "—", "Top3": "—", "Best Epoch": "—"})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"{'─'*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Inference")
    parser.add_argument("--models",  type=int,   nargs="+", default=[1, 2, 3],
                        help="Список model_id (1 2 3)")
    parser.add_argument("--folds",   type=int,   nargs="+", default=[0, 0, 0],
                        help="Відповідні фолди checkpoint (1:1 до --models)")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Ваги weighted ensemble (default: рівні)")
    parser.add_argument("--tta",     type=int,   default=1,
                        help="Кількість TTA проходів (default=1, без TTA)")
    parser.add_argument("--output",  type=str,
                        default=os.path.join(OUTPUT_DIR, "submission.csv"))
    args = parser.parse_args()

    if len(args.models) != len(args.folds):
        parser.error("--models і --folds мають однакову кількість елементів")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything()

    print(f"Device:  {device}")
    print(f"Моделі:  {args.models}")
    print(f"Фолди:   {args.folds}")
    print(f"TTA:     {args.tta}")

    print("\nЗавантаження checkpoint-ів:")
    models_configs = [
        (load_model(mid, fold, device), mid)
        for mid, fold in zip(args.models, args.folds)
    ]

    weights = np.array(args.weights) if args.weights else np.ones(len(args.models))
    weights = weights / weights.sum()
    print(f"\nВаги ensemble: { {mid: round(w, 3) for mid, w in zip(args.models, weights)} }")

    print_results_table(args.models, args.folds[0])

    create_submission(
        models_configs=models_configs,
        weights=weights,
        device=device,
        output_path=args.output,
        n_tta=args.tta,
    )
