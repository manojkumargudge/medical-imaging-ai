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
