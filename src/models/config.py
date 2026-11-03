"""Classifier configuration, read from configs/classifier.yaml and checked."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.data.eda import FINDINGS

ARCHITECTURES = ("densenet121",)
OPTIMIZERS = ("adamw",)
LOSSES = ("bce_with_logits",)
SELECTION_METRICS = ("val_mean_auroc",)


@dataclass(frozen=True)
class AugmentConfig:
    rotation_degrees: float
    translate: float
    brightness: float
    contrast: float


@dataclass(frozen=True)
class ClassifierConfig:
    processed_dir: Path
    manifest: Path
    num_workers: int
    architecture: str
    weights: str | None
    labels: tuple[str, ...]
    loss: str
    batch_size: int
    optimizer: str
    lr: float
    weight_decay: float
    max_epochs: int
    amp: bool
    seed: int
    selection_metric: str
    checkpoint_dir: Path
    augment: AugmentConfig
    subgroup_min_positives: int

    def fingerprint(self) -> str:
        """Hash of the settings a resumed run must share with the checkpoint it resumes from.

        Paths, worker count and max_epochs are left out: they can change between sessions.
        """
        fields = asdict(self)
        for key in ("processed_dir", "manifest", "num_workers", "checkpoint_dir", "max_epochs"):
            del fields[key]
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Plain types only, so it fits in a weights_only checkpoint."""
        return json.loads(json.dumps(asdict(self), default=str))  # type: ignore[no-any-return]


def _check(config: ClassifierConfig) -> ClassifierConfig:
    choices = {
        "architecture": ARCHITECTURES,
        "optimizer": OPTIMIZERS,
        "loss": LOSSES,
        "selection_metric": SELECTION_METRICS,
    }
    for name, allowed in choices.items():
        if getattr(config, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {getattr(config, name)!r}")
    if config.labels != FINDINGS:
        raise ValueError(f"labels must be the 14 NIH findings in order {FINDINGS}")
    positive = {"batch_size": config.batch_size, "lr": config.lr, "max_epochs": config.max_epochs}
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    augment = config.augment
    if not (0 <= augment.rotation_degrees <= 15 and 0 <= augment.translate <= 0.15):
        raise ValueError(f"rotation and shift must stay small, got {augment}")
    if not (0 <= augment.brightness <= 0.3 and 0 <= augment.contrast <= 0.3):
        raise ValueError(f"brightness and contrast changes must stay small, got {augment}")
    return config


def load_config(path: Path, **overrides: Any) -> ClassifierConfig:
    """Read the YAML config; keyword overrides (e.g. checkpoint_dir in tests) are checked too."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data, model, train = raw["data"], raw["model"], raw["train"]
    config = ClassifierConfig(
        processed_dir=Path(data["processed_dir"]),
        manifest=Path(data["manifest"]),
        num_workers=int(data["num_workers"]),
        architecture=model["architecture"],
        weights=model["weights"],
        labels=tuple(model["labels"]),
        loss=train["loss"],
        batch_size=int(train["batch_size"]),
        optimizer=train["optimizer"],
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        max_epochs=int(train["max_epochs"]),
        amp=bool(train["amp"]),
        seed=int(train["seed"]),
        selection_metric=train["selection_metric"],
        checkpoint_dir=Path(train["checkpoint_dir"]),
        augment=AugmentConfig(**{key: float(value) for key, value in raw["augment"].items()}),
        subgroup_min_positives=int(raw["evaluation"]["subgroup_min_positives"]),
    )
    return _check(replace(config, **overrides))
