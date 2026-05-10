import pandas as pd

from datasets import SoundscapeDataset

from evaluate import (
    read_soundscapes_labels,
    competition_metric,
    roc_auc_score_micro,
    cmap,
)

from torch.utils.data import DataLoader
from tqdm import tqdm


class BirdClassifier:
    def classify(self, batch):
        raise NotImplementedError


class TrainSoundscapesMetrics:
    def __init__(self, data_dir: str):
        classes = pd.read_csv(f"{data_dir}/taxonomy.csv")["primary_label"].tolist()

        self.soundscapes_labels = read_soundscapes_labels(
            f"{data_dir}/train_soundscapes_labels.csv", classes
        )

        self.all_classes = classes
        self.row_ids = set(self.soundscapes_labels.index)

    def compute_metrics(
        self, preds: dict[str, list[str]], exclude_labels: list[str] | None = None
    ) -> dict[str, float]:
        if not all(row_id in preds for row_id in self.soundscapes_labels.index):
            return {}

        submission = pd.DataFrame(
            [[row_id] + preds[row_id] for row_id in self.soundscapes_labels.index],
            columns=["row_id"] + self.all_classes,
        ).set_index("row_id")
        solution = self.soundscapes_labels

        if exclude_labels is not None:
            submission = submission.drop(columns=exclude_labels)
            solution = solution.drop(columns=exclude_labels)

        return {
            "aucroc_macro": competition_metric(solution, submission),
            "aucroc_micro": roc_auc_score_micro(solution, submission),
            "cmap": cmap(solution, submission),
        }


class TrainSoundscapesEvaluator:
    def __init__(self, data_dir: str):
        val_soundscapes_files = (
            pd.read_csv(f"{data_dir}/train_soundscapes_labels.csv")["filename"]
            .unique()
            .tolist()
        )
        self.val_soundscapes_ds = SoundscapeDataset(
            [f"{data_dir}/train_soundscapes/{fname}" for fname in val_soundscapes_files]
        )

        self.metrics = TrainSoundscapesMetrics(data_dir)

    def get_dataset(self):
        return self.val_soundscapes_ds

    def get_epoch_evaluator(self) -> "TrainSoundscapesEpochEvaluator":
        return TrainSoundscapesEpochEvaluator(self.metrics)

    def evaluate(
        self,
        model: BirdClassifier,
        batch_size: int = 1,
        num_workers: int = 0,
        progress_bar: bool = True,
    ) -> dict[str, float]:
        epoch = self.get_epoch_evaluator()

        dl = DataLoader(
            self.val_soundscapes_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        if progress_bar:
            dl = tqdm(dl)

        for batch in dl:
            wave, row_ids = batch

            preds = model.classify(wave)

            for row_id, pred in zip(row_ids, preds):
                epoch.add_preds(row_id, pred.tolist())

        return epoch.compute_metrics()


class TrainSoundscapesEpochEvaluator:
    def __init__(self, metrics: TrainSoundscapesMetrics):
        self.metrics = metrics

        self.preds = {}

    def add_preds(self, row_id: str, probs: list[str]):
        self.preds[row_id] = probs

    def compute_metrics(
        self, exclude_labels: list[str] | None = None
    ) -> dict[str, float]:
        return self.metrics.compute_metrics(self.preds, exclude_labels=exclude_labels)

    def reset(self):
        self.preds.clear()
