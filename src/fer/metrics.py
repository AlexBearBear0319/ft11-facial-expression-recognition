"""Metrics shared by every baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS


def summarize_predictions(
    actual: list[int], predicted: list[int], average_loss: float
) -> dict[str, Any]:
    class_ids = list(range(len(LABELS)))
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=class_ids, average="macro", zero_division=0)),
        "per_class": classification_report(
            actual,
            predicted,
            labels=class_ids,
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=class_ids).tolist(),
    }


def majority_reference(train_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_rows or not evaluation_rows:
        raise ValueError("The majority reference needs train and evaluation records.")
    counts = np.bincount([int(row["class_id"]) for row in train_rows], minlength=len(LABELS))
    majority_class = int(counts.argmax())
    actual = [int(row["class_id"]) for row in evaluation_rows]
    predicted = [majority_class] * len(actual)
    result = {
        "model_id": "majority_reference",
        "predicted_class": LABELS[majority_class],
        "train_class_counts": {label: int(counts[index]) for index, label in enumerate(LABELS)},
        **summarize_predictions(actual, predicted, average_loss=0.0),
    }
    result["loss"] = None
    return result
