from dotenv import load_dotenv

load_dotenv()

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import SOUNDSCAPES_DIR, TAXONOMY_CSV, OUTPUT_DIR
from utils import seed_everything
from datasets import SoundscapeDataset
from models import ModelBase
from train import TrainingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint: str) -> ModelBase:
    training_module = TrainingModule.load_from_checkpoint(checkpoint)
    training_module.eval()
    return training_module.model


@torch.no_grad()
def predict_soundscape(
    filepath: str,
    model: ModelBase,
) -> dict:
    row_ids = None
    feature_extractor = model.make_feature_extractor().to(device)

    dataset = SoundscapeDataset(filepath)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    model_pred = []
    row_ids_flat = []

    for wf, row_ids in loader:
        wf = wf.to(device)
        logits = model(feature_extractor(wf))

        model_pred.append(torch.sigmoid(logits).cpu().numpy())
        row_ids_flat.extend(row_ids)

    model_pred = np.vstack(model_pred)

    return {row_id: model_pred[i] for i, row_id in enumerate(row_ids_flat)}


def create_submission(
    model: ModelBase,
    output_path: str,
):
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    all_classes = sorted(taxonomy_df["primary_label"].unique().tolist())
    sample_file = pd.read_csv("solution.csv")

    soundscape_files = sorted(
        [f for f in os.listdir(SOUNDSCAPES_DIR) if f.endswith(".ogg")]
    )
    np.random.seed(42)
    soundscape_files = np.random.choice(soundscape_files, size=100, replace=False)

    sample_files = []
    for f in sample_file["row_id"]:
        sample_files.append(f[: f.rfind("_")])
    sample_files = [v + ".ogg" for v in set(sample_files)]
    soundscape_files = sample_files

    print(f"\nЗнайдено soundscapes: {len(soundscape_files)}")

    all_predictions = {}
    for fname in tqdm(soundscape_files, desc="Inference"):
        filepath = os.path.join(SOUNDSCAPES_DIR, fname)
        preds = predict_soundscape(filepath, model)
        all_predictions.update(preds)

    records = []
    for row_id in sample_file["row_id"]:
        probs = all_predictions.get(row_id, np.zeros(len(all_classes)))
        rec = {"row_id": row_id}
        rec.update({cls: float(probs[i]) for i, cls in enumerate(all_classes)})
        records.append(rec)

    submission_df = pd.DataFrame(records)
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission збережено: {output_path}")
    print(f"   Рядків: {len(submission_df)} | Колонок: {len(submission_df.columns)}")
    return submission_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Inference")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint file", required=True
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(OUTPUT_DIR, "submission.csv")
    )
    args = parser.parse_args()

    seed_everything()

    model = load_model(args.checkpoint)
    model.to(device)

    submission = create_submission(
        model=model,
        output_path=args.output,
    )
