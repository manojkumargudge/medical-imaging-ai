from typing import Any

import numpy as np
import pytest

from src.models import metrics

LABELS = ("A", "B")


def test_label_aurocs_with_counts() -> None:
    targets = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
    scores = np.array([[0.9, 0.1], [0.2, 0.2], [0.4, 0.3], [0.5, 0.4]])
    rows = metrics.label_aurocs(targets, scores, LABELS)
    assert rows[0] == {"label": "A", "auroc": 0.75, "positives": 2, "negatives": 2}
    assert rows[1]["auroc"] is None and rows[1]["positives"] == 0
    assert metrics.mean_auroc(rows) == 0.75


def test_mean_auroc_skips_labels_with_few_positives() -> None:
    rows: list[dict[str, Any]] = [
        {"label": "A", "auroc": 0.8, "positives": 20},
        {"label": "B", "auroc": 0.4, "positives": 3},
        {"label": "C", "auroc": None, "positives": 0},
    ]
    assert metrics.mean_auroc(rows) == pytest.approx(0.6)
    assert metrics.mean_auroc(rows, min_positives=10) == pytest.approx(0.8)
    assert metrics.mean_auroc(rows[2:]) is None


def test_best_f1_threshold() -> None:
    targets = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.3, 0.35, 0.8, 0.7, 0.9])
    best = metrics.best_f1_threshold(targets, scores)
    assert best is not None
    assert best["threshold"] == pytest.approx(0.35)
    assert best["f1"] == pytest.approx(2 * 0.75 * 1.0 / 1.75)
    at = metrics.threshold_metrics(targets, scores, best["threshold"])
    assert (at["tp"], at["fp"], at["fn"], at["tn"]) == (3, 1, 0, 2)
    assert at["f1"] == pytest.approx(best["f1"])
    assert at["specificity"] == pytest.approx(2 / 3)


def test_no_threshold_without_both_classes() -> None:
    assert metrics.best_f1_threshold(np.zeros(4), np.linspace(0, 1, 4)) is None
    assert metrics.best_f1_threshold(np.ones(4), np.linspace(0, 1, 4)) is None


@pytest.mark.parametrize(
    ("age", "band"),
    [(0, "0-17"), (17, "0-17"), (18, "18-39"), (79, "60-79"), (100, "80-100")],
)
def test_age_bands(age: int, band: str) -> None:
    assert metrics.age_band(age) == band


@pytest.mark.parametrize("age", [101, 148, 414])
def test_age_above_100_is_invalid(age: int) -> None:
    assert metrics.age_band(age) == metrics.INVALID_AGE


def test_subgroups_count_but_do_not_score_excluded_groups() -> None:
    targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=float)
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.3], [0.6, 0.2]])
    groups = ["x", "x", "x", "x", metrics.INVALID_AGE]
    result = metrics.subgroup_aurocs(targets, scores, groups, LABELS, 1, (metrics.INVALID_AGE,))
    assert result[metrics.INVALID_AGE] == {"images": 1, "excluded": True}
    assert result["x"]["images"] == 4
    assert result["x"]["mean_auroc"] == 1.0
    assert result["x"]["labels_in_mean"] == 2


def test_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError):
        metrics.label_aurocs(np.zeros((3, 2)), np.zeros((3, 3)), LABELS)
