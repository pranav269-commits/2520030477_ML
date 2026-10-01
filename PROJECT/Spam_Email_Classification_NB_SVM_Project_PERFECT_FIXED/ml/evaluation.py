from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Optional[Iterable[float]] = None,
) -> dict:
    """Return classification metrics used throughout the project.

    Spam is encoded as the positive class (1), ham as the negative class (0).
    """
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        roc_auc = _rounded(roc_auc_score(y_true, np.asarray(list(y_score), dtype=float)))

    return {
        "accuracy": _rounded(accuracy_score(y_true, y_pred)),
        "precision": _rounded(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _rounded(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _rounded(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": _rounded(specificity),
        "roc_auc": roc_auc,
        "fpr": _rounded(fpr),
        "fnr": _rounded(fnr),
        "confusion_matrix": cm.astype(int).tolist(),
    }
