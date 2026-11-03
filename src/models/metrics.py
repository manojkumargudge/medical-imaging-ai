"""Multi-label metrics: per-label AUROC, F1-optimal thresholds and subgroup AUROC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.data.eda import AGE_BIN_NAMES, AGE_BINS, MAX_PLAUSIBLE_AGE

Array = NDArray[np.floating[Any]]
INVALID_AGE = f"invalid (age > {MAX_PLAUSIBLE_AGE})"


def _check(targets: Array, scores: Array, labels: Sequence[str]) -> None:
    if targets.shape != scores.shape or targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError(f"targets {targets.shape} and scores {scores.shape} for {len(labels)}")
    if targets.shape[0] == 0:
        raise ValueError("no rows")


def label_aurocs(targets: Array, scores: Array, labels: Sequence[str]) -> list[dict[str, Any]]:
    """AUROC per label with its positive and negative counts; None when one class is missing."""
    _check(targets, scores, labels)
    rows = []
    for index, label in enumerate(labels):
        column = targets[:, index]
        positives = int(column.sum())
        negatives = int(len(column) - positives)
        auroc = None
        if positives and negatives:
            auroc = float(roc_auc_score(column, scores[:, index]))
        rows.append(
            {"label": label, "auroc": auroc, "positives": positives, "negatives": negatives}
        )
    return rows


def mean_auroc(rows: Sequence[dict[str, Any]], min_positives: int = 1) -> float | None:
    values = [
        row["auroc"]
        for row in rows
        if row["auroc"] is not None and row["positives"] >= min_positives
    ]
    return float(np.mean(values)) if values else None


def best_f1_threshold(targets: Array, scores: Array) -> dict[str, float] | None:
    """The score threshold (predict positive when score >= threshold) with the highest F1."""
    if targets.sum() == 0 or targets.sum() == len(targets):
        return None
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    best = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def threshold_metrics(targets: Array, scores: Array, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    actual = targets.astype(bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "sensitivity": recall,
        "specificity": ratio(tn, tn + fp),
        "f1": f1,
    }


def age_band(age: int) -> str:
    if age > MAX_PLAUSIBLE_AGE or age < 0:
        return INVALID_AGE
    for upper, name in zip(AGE_BINS[1:], AGE_BIN_NAMES, strict=True):
        if age < upper:
            return name
    raise AssertionError(age)


def subgroup_aurocs(
    targets: Array,
    scores: Array,
    groups: Sequence[str],
    labels: Sequence[str],
    min_positives: int,
    excluded: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Per-label AUROC inside each group. Groups in `excluded` are only counted, not scored."""
    _check(targets, scores, labels)
    names = np.asarray(groups)
    if len(names) != len(targets):
        raise ValueError(f"{len(names)} group values for {len(targets)} rows")
    result: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        mask = names == group
        if group in excluded:
            result[group] = {"images": int(mask.sum()), "excluded": True}
            continue
        rows = label_aurocs(targets[mask], scores[mask], labels)
        result[group] = {
            "images": int(mask.sum()),
            "mean_auroc": mean_auroc(rows, min_positives),
            "labels_in_mean": sum(
                row["auroc"] is not None and row["positives"] >= min_positives for row in rows
            ),
            "labels": rows,
        }
    return result
