"""Multi-label metrics: per-label AUROC, thresholds, subgroup AUROC and bootstrap intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
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


def sensitivity_threshold(
    targets: Array, scores: Array, target: float, min_positives: int
) -> dict[str, float] | None:
    """The highest threshold (predict positive when score >= threshold) whose sensitivity is at
    least `target`, so specificity is as high as that sensitivity allows. None with fewer than
    `min_positives` positives or no negatives: such a label gets a score only."""
    positives = np.sort(scores[targets.astype(bool)])[::-1]
    if len(positives) < max(min_positives, 1) or len(positives) == len(targets):
        return None
    # Exact arithmetic: 0.8 * 30 is 24.000000000000004 in floating point.
    needed = math.ceil(Fraction(str(target)) * len(positives))
    threshold = float(positives[needed - 1])
    at = threshold_metrics(targets, scores, threshold)
    return {
        "threshold": threshold,
        "sensitivity": at["sensitivity"],
        "specificity": at["specificity"],
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


def common_labels(
    targets: Array,
    groups: Sequence[str],
    labels: Sequence[str],
    min_positives: int,
    excluded: Sequence[str] = (),
) -> list[str]:
    """Labels with at least `min_positives` positives and one negative in every scored group,
    so that every group's mean AUROC is taken over the same labels."""
    names = np.asarray(groups)
    if len(names) != len(targets):
        raise ValueError(f"{len(names)} group values for {len(targets)} rows")
    masks = [names == group for group in sorted(set(groups)) if group not in excluded]
    if not masks:
        return []
    return [
        label
        for index, label in enumerate(labels)
        if all(min_positives <= targets[mask, index].sum() < mask.sum() for mask in masks)
    ]


class _RankedScores:
    """One label's scores ranked once; a bootstrap resample is then a weighted count, which is
    the Mann-Whitney form of AUROC (ties count one half)."""

    def __init__(self, targets: Array, scores: Array) -> None:
        _, blocks = np.unique(scores, return_inverse=True)
        self.size = int(blocks.max()) + 1 if len(blocks) else 0
        positive = targets.astype(bool)
        self.positive_rows, self.negative_rows = np.flatnonzero(positive), np.flatnonzero(~positive)
        self.positive_blocks, self.negative_blocks = blocks[positive], blocks[~positive]

    def auroc(self, weights: Array) -> float | None:
        pos = np.bincount(self.positive_blocks, weights[self.positive_rows], self.size)
        neg = np.bincount(self.negative_blocks, weights[self.negative_rows], self.size)
        total_pos, total_neg = pos.sum(), neg.sum()
        if total_pos == 0 or total_neg == 0:
            return None
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + 0.5 * neg)) / (total_pos * total_neg))


def _interval(values: Array, level: float) -> dict[str, Any] | None:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return None
    low, high = np.percentile(valid, [50 * (1 - level), 50 * (1 + level)])
    return {"low": float(low), "high": float(high), "valid_resamples": len(valid)}


def bootstrap_aurocs(
    targets: Array,
    scores: Array,
    patients: Sequence[str],
    labels: Sequence[str],
    mean_labels: Sequence[str],
    resamples: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, Any]:
    """Percentile intervals for per-label AUROC and for the mean AUROC over `mean_labels`.

    Patients are resampled with replacement, each with all of its images, because images of one
    patient are not independent. A resample in which a label has only one class gives no AUROC
    for that label (and no mean, if the label is in `mean_labels`); the number of usable
    resamples is reported with each interval.
    """
    _check(targets, scores, labels)
    if len(patients) != len(targets):
        raise ValueError(f"{len(patients)} patient IDs for {len(targets)} rows")
    _, patient_rows = np.unique(np.asarray(patients), return_inverse=True)
    count = int(patient_rows.max()) + 1
    ranked = [_RankedScores(targets[:, index], scores[:, index]) for index in range(len(labels))]
    values = np.full((resamples, len(labels)), np.nan)
    rng = np.random.default_rng(seed)
    for resample in range(resamples):
        draws = np.bincount(rng.integers(0, count, count), minlength=count)
        weights = draws[patient_rows].astype(float)
        for index, scores_for_label in enumerate(ranked):
            auroc = scores_for_label.auroc(weights)
            if auroc is not None:
                values[resample, index] = auroc
    columns = [list(labels).index(label) for label in mean_labels]
    return {
        "labels": {label: _interval(values[:, index], level) for index, label in enumerate(labels)},
        "mean": _interval(values[:, columns].mean(axis=1), level) if columns else None,
    }
