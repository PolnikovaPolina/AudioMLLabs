from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import numpy as np


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def calculate_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Macro F1-Score."""
    y_bin = (y_pred >= threshold).astype(int)
    f1s = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            f1s.append(f1_score(y_true[:, i], y_bin[:, i], zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def calculate_top_k_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 3
) -> float:
    """Top-K Accuracy."""
    correct = 0
    total = y_true.shape[0]
    for i in range(total):
        top_k = np.argsort(y_pred[i])[-k:]
        true_idx = np.where(y_true[i] == 1)[0]
        if len(set(top_k).intersection(set(true_idx))) > 0:
            correct += 1
    return correct / total if total > 0 else 0.0
