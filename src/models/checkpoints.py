"""Checkpoint files shared by the classifier and the segmentation model.

Checkpoints are `epoch_NNN.pt` (with `epoch_NNN.json` metrics) in one folder. They are written under
a unique temporary name and renamed, and an existing file is never overwritten, so a resumed run
can only add files.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_CHECKPOINT = re.compile(r"^epoch_(\d{3})\.pt$")


def checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"epoch_{epoch:03d}.pt"


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    found = sorted(
        path for path in checkpoint_dir.glob("epoch_*.pt") if _CHECKPOINT.match(path.name)
    )
    return found[-1] if found else None


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


def best_epoch(history: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """The epoch with the highest value of metric; the earliest one on a tie."""
    scored = [entry for entry in history if entry[metric] is not None]
    if not scored:
        raise ValueError(f"no epoch has a {metric}")
    return dict(max(scored, key=lambda entry: (entry[metric], -entry["epoch"])))


def save_new(path: Path, write: Callable[[Path], object]) -> None:
    """Write to a unique temporary name, then rename; refuses to replace an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.partial")
    write(temporary)
    if path.exists():
        raise FileExistsError(f"{path} appeared while saving; {temporary} kept")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
