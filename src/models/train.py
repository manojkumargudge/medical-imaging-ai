"""Train the classifier with one checkpoint per epoch; a rerun resumes from the latest checkpoint.

Checkpoints (`epoch_001.pt`, ...) and per-epoch metrics (`epoch_001.json`) go to the config's
checkpoint_dir, on Drive when run on Colab. They are written under a temporary name and renamed,
and an existing file is never overwritten. The best epoch is the one with the highest mean AUROC
over the 14 labels on val.

Run: python -m src.models.train --config configs/classifier.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.classifier import build_loss, build_model
from src.models.config import ClassifierConfig, load_config
from src.models.data import make_datasets
from src.models.metrics import Array, label_aurocs, mean_auroc

Batch = tuple[torch.Tensor, torch.Tensor]
Log = Callable[[str], None]
_CHECKPOINT = re.compile(r"^epoch_(\d{3})\.pt$")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def best_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The epoch with the highest val mean AUROC; the earliest one on a tie."""
    scored = [entry for entry in history if entry["val_mean_auroc"] is not None]
    if not scored:
        raise ValueError("no epoch has a val mean AUROC")
    return dict(max(scored, key=lambda entry: (entry["val_mean_auroc"], -entry["epoch"])))


def _save_new(path: Path, write: Callable[[Path], object]) -> None:
    """Write to a unique temporary name, then rename; refuses to replace an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; not overwriting")
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.partial")
    write(temporary)
    if path.exists():
        raise FileExistsError(f"{path} appeared while saving; {temporary} kept")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def predict(
    model: nn.Module, loader: DataLoader[Batch], device: torch.device, amp: bool
) -> tuple[Array, Array]:
    """Targets and sigmoid scores for every row of the loader, in order."""
    model.eval()
    targets, scores = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(scores)


def _loader(
    dataset: Dataset[Batch], config: ClassifierConfig, shuffle: bool, seed: int
) -> DataLoader[Batch]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )


def train(
    config: ClassifierConfig,
    datasets: Mapping[str, Dataset[Batch]] | None = None,
    device: torch.device | None = None,
    log: Log = print,
) -> dict[str, Any]:
    """Train up to config.max_epochs, resuming after the latest checkpoint. Returns the history."""
    device = device or select_device()
    use_amp = config.amp and device.type == "cuda"
    if config.amp and not use_amp:
        log("AMP needs a CUDA GPU; training in float32 on CPU")
    datasets = datasets if datasets is not None else make_datasets(config, ("train", "val"))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    model = build_model(len(config.labels), config.weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = build_loss()
    history: list[dict[str, Any]] = []
    start = 1

    latest = latest_checkpoint(config.checkpoint_dir)
    if latest is not None:
        state = load_checkpoint(latest)
        if state["fingerprint"] != config.fingerprint():
            raise ValueError(
                f"{latest} was trained with another config ({state['fingerprint']} vs "
                f"{config.fingerprint()}); use a new checkpoint_dir"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state["history"])
        start = int(state["epoch"]) + 1
        log(f"resuming from {latest.name}: epoch {start} of {config.max_epochs}")
    if start > config.max_epochs:
        log(f"all {config.max_epochs} epochs are done")

    val_loader = _loader(datasets["val"], config, shuffle=False, seed=config.seed)
    for epoch in range(start, config.max_epochs + 1):
        started = time.monotonic()
        # Seeded per epoch, so a resumed run sees the same order and augmentation draws.
        torch.manual_seed(config.seed + epoch)
        train_loader = _loader(datasets["train"], config, shuffle=True, seed=config.seed + epoch)
        model.train()
        total_loss, rows = 0.0, 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            rows += len(labels)
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}/{len(train_loader)} loss {total_loss / rows:.4f}")

        targets, scores = predict(model, val_loader, device, use_amp)
        per_label = label_aurocs(targets, scores, config.labels)
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / rows,
            "train_images": rows,
            "val_images": len(targets),
            "val_mean_auroc": mean_auroc(per_label),
            "val_auroc": {row["label"]: row["auroc"] for row in per_label},
            "seconds": round(time.monotonic() - started, 1),
        }
        history.append(entry)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "fingerprint": config.fingerprint(),
            "config": config.to_dict(),
            "versions": {
                "torch": str(torch.__version__),
                "torchvision": torchvision.__version__,
            },
        }
        path = checkpoint_path(config.checkpoint_dir, epoch)
        _save_new(path, partial(torch.save, state))
        summary = json.dumps(entry, indent=2) + "\n"
        _save_new(path.with_suffix(".json"), partial(_write_text, text=summary))
        mean = entry["val_mean_auroc"]
        log(
            f"epoch {epoch}: train loss {entry['train_loss']:.4f}, val mean AUROC "
            f"{'n/a' if mean is None else f'{mean:.4f}'} ({entry['seconds']} s), saved {path.name}"
        )

    if history:
        best = best_epoch(history)
        log(f"best epoch {best['epoch']}: val mean AUROC {best['val_mean_auroc']:.4f}")
    return {"history": history, "best": best_epoch(history) if history else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, help="override the config's folder")
    args = parser.parse_args()
    overrides = {"checkpoint_dir": args.checkpoint_dir} if args.checkpoint_dir else {}
    train(load_config(args.config, **overrides))


if __name__ == "__main__":
    main()
