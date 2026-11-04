"""The viewer's thresholds must be the val-chosen values in reports/thresholds_v2.json."""

import json
from pathlib import Path

import yaml

from src.data.eda import FINDINGS

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER = REPO_ROOT / "configs" / "viewer_thresholds.yaml"
REPORT = REPO_ROOT / "reports" / "thresholds_v2.json"


def test_viewer_thresholds_match_the_v2_report() -> None:
    viewer = yaml.safe_load(VIEWER.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    labels, sets = viewer["labels"], report["sets"]
    print(f"viewer labels {len(labels)}, report labels {len(sets['best_f1']['labels'])}")
    assert labels and sets["best_f1"]["labels"]
    assert report["selected_on"] == "val"
    assert REPO_ROOT / viewer["source"] == REPORT
    assert sorted(labels) == sorted(FINDINGS)

    for label, entry in labels.items():
        if entry["set"] == "score_only":
            assert entry["threshold"] is None and not entry["low_reliability"]
            continue
        chosen = sets[entry["set"]]["labels"][label]
        assert chosen is not None and entry["threshold"] == chosen["threshold"], label
        # Set A is only a fallback here, so every set A label is flagged.
        assert entry["low_reliability"] == (entry["set"] == "best_f1"), label

    # Labels without a set B threshold (too few val positives) cannot use set B.
    for label, chosen in sets["sensitivity_80"]["labels"].items():
        if chosen is None:
            assert labels[label]["set"] != "sensitivity_80", label
