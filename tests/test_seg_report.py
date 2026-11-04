"""docs/results_seg.md is generated from reports/segmentation_metrics.json."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models import seg_report

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "reports" / "segmentation_metrics.json"
RESULTS = REPO_ROOT / "docs" / "results_seg.md"


def tiny_report() -> dict[str, Any]:
    cases: dict[str, list[dict[str, Any]]] = {
        "val": [{"case": "a", "dice": 0.9}, {"case": "b", "dice": 0.7}],
        "test": [{"case": "c", "dice": 0.8}],
    }
    return {
        "notice": "For education only, not for clinical use.",
        "dataset": "D",
        "checkpoint_epoch": 10,
        "metric": "Dice",
        "splits": {
            split: {"cases": rows, "mean_dice": sum(r["dice"] for r in rows) / len(rows)}
            for split, rows in cases.items()
        },
    }


def test_summary_and_per_case_rows() -> None:
    text = seg_report.render_markdown(tiny_report())
    assert "For education only, not for clinical use." in text
    assert "| val | 2 | 0.8000 | 0.7000 | 0.9000 |" in text
    assert "| test | 1 | 0.8000 | 0.8000 | 0.8000 |" in text
    assert "| a | 0.9000 |" in text and "| c | 0.8000 |" in text
    assert "CI" not in text


def test_empty_split_fails() -> None:
    report = tiny_report()
    report["splits"]["test"]["cases"] = []
    with pytest.raises(ValueError, match="no test cases"):
        seg_report.render_markdown(report)


def test_committed_results_match_the_committed_metrics() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    expected = seg_report.render_markdown(report)
    committed = RESULTS.read_text(encoding="utf-8")
    print(f"generated {len(expected.splitlines())} lines, committed {len(committed.splitlines())}")
    assert expected and committed and committed == expected
