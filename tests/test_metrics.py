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


def test_sensitivity_threshold_is_the_highest_reaching_the_target() -> None:
    rng = np.random.default_rng(1)
    targets = np.r_[np.ones(30), np.zeros(50)]
    scores = rng.random(80)
    chosen = metrics.sensitivity_threshold(targets, scores, 0.8, 30)
    assert chosen is not None
    # 0.8 * 30 = 24 positives must be at or above the threshold.
    positives = np.sort(scores[:30])[::-1]
    assert chosen["threshold"] == positives[23]
    assert chosen["sensitivity"] == pytest.approx(24 / 30)
    at = metrics.threshold_metrics(targets, scores, chosen["threshold"])
    assert at["sensitivity"] == chosen["sensitivity"]
    assert at["specificity"] == chosen["specificity"]
    # Any higher threshold drops below the target.
    higher = metrics.threshold_metrics(targets, scores, float(np.nextafter(positives[23], 2)))
    assert higher["sensitivity"] < 0.8


def test_sensitivity_threshold_rounds_the_needed_positives_up() -> None:
    targets = np.r_[np.ones(31), np.zeros(5)]
    scores = np.r_[np.linspace(1, 0.1, 31), np.full(5, 0.05)]
    chosen = metrics.sensitivity_threshold(targets, scores, 0.8, 30)
    assert chosen is not None
    assert chosen["sensitivity"] == pytest.approx(25 / 31)  # 24.8 rounds up to 25


@pytest.mark.parametrize(("positives", "negatives"), [(29, 50), (30, 0)])
def test_sensitivity_threshold_needs_enough_positives_and_a_negative(
    positives: int, negatives: int
) -> None:
    targets = np.r_[np.ones(positives), np.zeros(negatives)]
    scores = np.linspace(0, 1, len(targets))
    assert metrics.sensitivity_threshold(targets, scores, 0.8, 30) is None


def test_common_labels_need_enough_positives_in_every_scored_group() -> None:
    groups = ["F"] * 20 + ["M"] * 20 + ["bad"] * 2
    targets = np.zeros((42, 3))
    targets[:10, 0] = targets[20:30, 0] = 1  # A: 10 in F and 10 in M
    targets[:10, 1] = targets[20:29, 1] = 1  # B: only 9 in M
    targets[:40, 2] = 1  # C: no negatives
    assert metrics.common_labels(targets, groups, ("A", "B", "C"), 10, ("bad",)) == ["A"]
    assert metrics.common_labels(targets, groups, ("A", "B", "C"), 10) == []


def test_weighted_auroc_matches_sklearn_with_ties_and_repeats() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    targets = (rng.random(200) < 0.3).astype(float)
    scores = np.round(rng.random(200), 1)  # many ties
    ranked = metrics._RankedScores(targets, scores)
    assert ranked.auroc(np.ones(200)) == pytest.approx(roc_auc_score(targets, scores))
    weights = rng.integers(0, 4, 200)
    repeated = np.repeat(np.arange(200), weights)
    assert ranked.auroc(weights.astype(float)) == pytest.approx(
        roc_auc_score(targets[repeated], scores[repeated])
    )


def bootstrap_data(patients: int = 60, per_patient: int = 2) -> tuple[Any, Any, list[str]]:
    rng = np.random.default_rng(3)
    targets = (rng.random((patients, 2)) < 0.4).astype(float)
    scores = np.clip(targets * 0.3 + rng.random((patients, 2)), 0, None)
    ids = [f"p{index}" for index in range(patients)]
    rows = np.repeat(np.arange(patients), per_patient)
    return targets[rows], scores[rows], [ids[row] for row in rows]


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    targets, scores, patients = bootstrap_data()
    first = metrics.bootstrap_aurocs(targets, scores, patients, LABELS, LABELS, 200, 42)
    again = metrics.bootstrap_aurocs(targets, scores, patients, LABELS, LABELS, 200, 42)
    other = metrics.bootstrap_aurocs(targets, scores, patients, LABELS, LABELS, 200, 7)
    assert first == again and first != other
    rows = metrics.label_aurocs(targets, scores, LABELS)
    for row in rows:
        interval = first["labels"][row["label"]]
        assert interval["low"] < row["auroc"] < interval["high"]
        assert interval["valid_resamples"] == 200
    assert first["mean"]["low"] < metrics.mean_auroc(rows) < first["mean"]["high"]


def test_bootstrap_resamples_patients_not_images() -> None:
    # Each patient's images are copies. Resampling whole patients gives the same intervals
    # whether a patient has one image or three; resampling images would not.
    one = metrics.bootstrap_aurocs(*bootstrap_data(per_patient=1), LABELS, LABELS, 100, 0)
    three = metrics.bootstrap_aurocs(*bootstrap_data(per_patient=3), LABELS, LABELS, 100, 0)
    for label in LABELS:
        for key in ("low", "high"):
            assert one["labels"][label][key] == pytest.approx(three["labels"][label][key])


def test_bootstrap_skips_resamples_with_one_class() -> None:
    targets = np.array([[1, 1], [0, 1], [0, 0], [0, 1], [0, 0]], dtype=float)
    scores = np.array([[0.9, 0.8], [0.1, 0.7], [0.2, 0.1], [0.3, 0.6], [0.4, 0.2]])
    result = metrics.bootstrap_aurocs(targets, scores, list("abcde"), LABELS, ["A"], 300, 0)
    # Label A has one positive image: about a third of the resamples miss it.
    assert 0 < result["labels"]["A"]["valid_resamples"] < 300
    assert result["mean"]["valid_resamples"] == result["labels"]["A"]["valid_resamples"]
