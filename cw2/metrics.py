"""Section 3.2 metrics with a fixed class order.

Undefined values: precision of a class that is never predicted (TP+FP=0) is set to 0, and so is
its F1 (zero_division=0). Every class in the fixed order is included in the macro averages; the
manifest audit guarantees every class has validation and test support, so recall is always defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate(y_true, y_pred, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    assert (s > 0).all(), "a class has no evaluation support"
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(r.mean()),
        "per_class": pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f, "support": s}),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def summary(m: dict) -> str:
    return f"macro-F1 {m['macro_f1']:.4f} | acc {m['accuracy']:.4f} | bal-acc {m['balanced_accuracy']:.4f}"
