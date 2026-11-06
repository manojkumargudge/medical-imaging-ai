"""Worklist logic without I/O: processing status, review status, times.

Processing status of a study:
- receiving: Orthanc has not marked it stable yet (instances may still arrive)
- no_model: stable, but no series has a modality any model reads (DX, CR, CT)
- processing: stable, and a model that should run has no result row yet
- failed: the newest row of some model has status "error"
- done: the newest row of every model that should run has status "ok"
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

# The worker's model names (src/service/analyzers.py; tests/test_api.py checks they match).
CLASSIFIER = "densenet121-chestxray14"
SEGMENTATION = "monai-unet-spleen"
MODEL_FOR_MODALITY = {"DX": CLASSIFIER, "CR": CLASSIFIER, "CT": SEGMENTATION}

SCORE_NOTE = (
    "Scores are not calibrated probabilities: a threshold is a cut-off on the model's score, "
    "not a probability of disease."
)
NOTICE = "For education only. Not for clinical use."


def expected_models(modalities: Iterable[str]) -> set[str]:
    return {MODEL_FOR_MODALITY[m] for m in modalities if m in MODEL_FOR_MODALITY}


def processing_status(
    is_stable: bool, modalities: Iterable[str], newest_status_by_model: Mapping[str, str]
) -> str:
    if not is_stable:
        return "receiving"
    expected = expected_models(modalities)
    if not expected:
        return "no_model"
    if any(newest_status_by_model.get(model) == "error" for model in expected):
        return "failed"
    if all(newest_status_by_model.get(model) == "ok" for model in expected):
        return "done"
    return "processing"


def review_status(decision: str | None) -> str:
    return {None: "none", "approve": "approved", "reject": "rejected"}[decision]


def orthanc_time(value: str) -> datetime:
    """Orthanc's LastUpdate ("20261106T050824"), in the container's time zone, which is UTC."""
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def result_summary(rows: Iterable[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> str | None:
    """One line from the newest result of each model.

    DX: how many labels with a threshold (13; Hernia is score only) score at or above it, with
    the thresholds in configs/viewer_thresholds.yaml. CT: the spleen volume from the mask.
    """
    parts = []
    for row in sorted(rows, key=lambda r: str(r["model"])):
        try:
            if row["status"] != "ok":
                parts.append("Failed")
            elif row["model"] == CLASSIFIER:
                scores = row["results"]["instances"][0]["labels"]
                cut = {
                    k: v["threshold"] for k, v in thresholds.items() if v["threshold"] is not None
                }
                above = sum(1 for label, t in cut.items() if scores[label]["score"] >= t)
                parts.append(f"{above} of {len(cut)} above threshold")
            elif row["model"] == SEGMENTATION:
                volume = sum(series["spleen_ml"] for series in row["results"]["series"])
                parts.append(f"Spleen {volume:.1f} mL")
        except (KeyError, IndexError, TypeError):  # a malformed row must not break the worklist
            parts.append("Result not readable")
    return "; ".join(parts) or None


def description(study: Mapping[str, Any], series: Iterable[Mapping[str, Any]]) -> str:
    """StudyDescription, else the first SeriesDescription. An upload's is "Upload <id>"; the
    uploaded file's name is never shown."""
    found = str(study.get("MainDicomTags", {}).get("StudyDescription") or "")
    for one in series:
        found = found or str(one.get("MainDicomTags", {}).get("SeriesDescription") or "")
    return found
