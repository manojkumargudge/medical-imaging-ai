"""3D spleen segmentation: MONAI UNet (spatial_dims=3) on MSD Task09_Spleen.

- Preprocessing: reorient to RAS, resample to 1.5 mm, CT soft-tissue window to [0, 1], crop to the
  body (foreground).
- Train split only: random 96^3 crops (half centred on spleen) and light augmentation (small
  rotation and scaling, intensity shift). Val and test get no random transform.
- Dice + cross-entropy loss; validation by sliding-window inference over whole volumes; the best
  checkpoint has the highest mean val Dice.
- A checkpoint every val_interval epochs to checkpoint_dir (Drive on Colab), never overwritten; a
  rerun resumes from the latest one.

Run: python -m src.models.segmentation train --config configs/segmentation.yaml
     python -m src.models.segmentation evaluate --config configs/segmentation.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from monai.data import CacheDataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch import nn

from src.data.preprocess import check_off_drive
from src.data.spleen import read_split
from src.models.checkpoints import (
    best_epoch,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    save_new,
    write_text,
)

KEYS = ("image", "label")
Log = Callable[[str], None]


@dataclass(frozen=True)
class SegConfig:
    task_dir: Path
    split: Path
    cache_rate: float
    num_workers: int
    spatial_dims: int
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    num_res_units: int
    norm: str
    pixdim: tuple[float, float, float]
    hu_window: tuple[float, float]
    roi_size: tuple[int, int, int]
    samples_per_volume: int
    positive_fraction: float
    affine_prob: float
    rotate_degrees: float
    scale: float
    intensity_shift: float
    intensity_prob: float
    loss: str
    batch_size: int
    optimizer: str
    lr: float
    max_epochs: int
    val_interval: int
    amp: bool
    seed: int
    selection_metric: str
    checkpoint_dir: Path
    sw_batch_size: int
    overlap: float

    def fingerprint(self) -> str:
        """Settings a resumed run must share with its checkpoint (not paths, workers, epochs)."""
        fields = asdict(self)
        for key in ("task_dir", "split", "num_workers", "cache_rate", "checkpoint_dir"):
            del fields[key]
        del fields["max_epochs"]
        return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[
            :16
        ]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))  # type: ignore[no-any-return]


def _check(config: SegConfig) -> SegConfig:
    if config.spatial_dims != 3:
        raise ValueError("the segmentation model is 3D: spatial_dims must be 3")
    if len(config.strides) != len(config.channels) - 1:
        raise ValueError("strides must have one entry fewer than channels")
    choices = {
        "loss": ("dice_ce",),
        "optimizer": ("adam",),
        "norm": ("batch", "instance"),
        "selection_metric": ("val_mean_dice",),
    }
    for name, allowed in choices.items():
        if getattr(config, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {getattr(config, name)!r}")
    low, high = config.hu_window
    if not low < high:
        raise ValueError(f"hu_window must be (low, high), got {config.hu_window}")
    if not (0 <= config.rotate_degrees <= 15 and 0 <= config.scale <= 0.15):
        raise ValueError("rotation and scaling must stay light")
    if not 0 < config.positive_fraction < 1:
        raise ValueError("positive_fraction must be between 0 and 1")
    if config.val_interval < 1 or config.max_epochs < 1:
        raise ValueError("max_epochs and val_interval must be positive")
    return config


def load_seg_config(path: Path, **overrides: Any) -> SegConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data, model, pre, aug, train, infer = (
        raw[key] for key in ("data", "model", "preprocess", "augment", "train", "inference")
    )
    config = SegConfig(
        task_dir=Path(data["task_dir"]),
        split=Path(data["split"]),
        cache_rate=float(data["cache_rate"]),
        num_workers=int(data["num_workers"]),
        spatial_dims=int(model["spatial_dims"]),
        channels=tuple(int(value) for value in model["channels"]),
        strides=tuple(int(value) for value in model["strides"]),
        num_res_units=int(model["num_res_units"]),
        norm=str(model["norm"]),
        pixdim=tuple(float(value) for value in pre["pixdim"]),  # type: ignore[arg-type]
        hu_window=tuple(float(value) for value in pre["hu_window"]),  # type: ignore[arg-type]
        roi_size=tuple(int(value) for value in pre["roi_size"]),  # type: ignore[arg-type]
        samples_per_volume=int(aug["samples_per_volume"]),
        positive_fraction=float(aug["positive_fraction"]),
        affine_prob=float(aug["affine_prob"]),
        rotate_degrees=float(aug["rotate_degrees"]),
        scale=float(aug["scale"]),
        intensity_shift=float(aug["intensity_shift"]),
        intensity_prob=float(aug["intensity_prob"]),
        loss=str(train["loss"]),
        batch_size=int(train["batch_size"]),
        optimizer=str(train["optimizer"]),
        lr=float(train["lr"]),
        max_epochs=int(train["max_epochs"]),
        val_interval=int(train["val_interval"]),
        amp=bool(train["amp"]),
        seed=int(train["seed"]),
        selection_metric=str(train["selection_metric"]),
        checkpoint_dir=Path(train["checkpoint_dir"]),
        sw_batch_size=int(infer["sw_batch_size"]),
        overlap=float(infer["overlap"]),
    )
    return _check(replace(config, **overrides))


def build_unet(config: SegConfig) -> nn.Module:
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=1,
        out_channels=2,  # background, spleen
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        norm=config.norm,
    )


def build_loss() -> nn.Module:
    return DiceCELoss(to_onehot_y=True, softmax=True)


def build_transforms(config: SegConfig, train: bool) -> Compose:
    """Deterministic preprocessing for every split; random crops and augmentation for train only."""
    low, high = config.hu_window
    steps: list[Any] = [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        # Explicit labels: MONAI is changing the default axis labels of Orientationd.
        Orientationd(keys=KEYS, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(keys=KEYS, pixdim=config.pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),
        EnsureTyped(keys=KEYS, track_meta=not train),
    ]
    if train:
        positive = config.positive_fraction
        radians = math.radians(config.rotate_degrees)
        steps += [
            RandCropByPosNegLabeld(
                keys=KEYS,
                label_key="label",
                spatial_size=config.roi_size,
                pos=positive,
                neg=1 - positive,
                num_samples=config.samples_per_volume,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            RandAffined(
                keys=KEYS,
                prob=config.affine_prob,
                rotate_range=(radians, radians, radians),
                scale_range=(config.scale,) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandShiftIntensityd(
                keys="image", offsets=config.intensity_shift, prob=config.intensity_prob
            ),
        ]
    return Compose(steps)


def data_dicts(config: SegConfig, split: str) -> list[dict[str, str]]:
    rows = read_split(config.split)[split]
    return [
        {
            "image": str(config.task_dir / row.image),
            "label": str(config.task_dir / row.label),
            "case": row.case,
        }
        for row in rows
    ]


def make_loaders(config: SegConfig, splits: Sequence[str]) -> dict[str, DataLoader]:
    """Data on the VM disk only; train shuffles crops, val/test load whole volumes one by one."""
    check_off_drive(config.task_dir, "task_dir")
    loaders = {}
    for split in splits:
        train = split == "train"
        dataset = CacheDataset(
            data=data_dicts(config, split),
            transform=build_transforms(config, train),
            cache_rate=config.cache_rate,
            num_workers=config.num_workers,
            progress=False,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size if train else 1,
            shuffle=train,
            num_workers=config.num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def case_dice(
    model: nn.Module, loader: DataLoader, config: SegConfig, device: torch.device, amp: bool
) -> list[dict[str, Any]]:
    """Dice of the spleen for every case, from sliding-window inference over the whole volume."""
    model.eval()
    to_pred = AsDiscrete(argmax=True, to_onehot=2)
    to_label = AsDiscrete(to_onehot=2)
    metric = DiceMetric(include_background=False, reduction="none")
    results = []
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for batch in loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            logits = sliding_window_inference(
                images, config.roi_size, config.sw_batch_size, model, overlap=config.overlap
            )
            assert isinstance(logits, torch.Tensor)
            predictions = [to_pred(item) for item in decollate_batch(logits.float())]
            targets = [to_label(item) for item in decollate_batch(labels)]
            scores = metric(
                y_pred=cast(list[torch.Tensor], predictions), y=cast(list[torch.Tensor], targets)
            )
            for case, score in zip(batch["case"], scores, strict=True):
                results.append({"case": case, "dice": float(score[0])})
    return results


def mean_dice(results: Sequence[Mapping[str, Any]]) -> float | None:
    values = [row["dice"] for row in results if not math.isnan(row["dice"])]
    return float(np.mean(values)) if values else None


def train_segmentation(
    config: SegConfig,
    loaders: Mapping[str, DataLoader] | None = None,
    device: torch.device | None = None,
    log: Log = print,
) -> dict[str, Any]:
    """Train up to max_epochs, resuming after the latest checkpoint; returns the history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp and device.type == "cuda"
    if config.amp and not use_amp:
        log("AMP needs a CUDA GPU; training in float32")
    loaders = loaders if loaders is not None else make_loaders(config, ("train", "val"))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    model = build_unet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = build_loss()
    history: list[dict[str, Any]] = []
    start = 1

    latest = latest_checkpoint(config.checkpoint_dir)
    if latest is not None:
        state = load_checkpoint(latest)
        if state["fingerprint"] != config.fingerprint():
            raise ValueError(f"{latest} was trained with another config; use a new checkpoint_dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state["history"])
        start = int(state["epoch"]) + 1
        log(f"resuming from {latest.name}: epoch {start} of {config.max_epochs}")

    losses: list[float] = []
    for epoch in range(start, config.max_epochs + 1):
        started = time.monotonic()
        # Seeded per epoch, so a resumed run draws the same crops and augmentation.
        torch.manual_seed(config.seed + epoch)
        np.random.seed(config.seed + epoch)
        cast(Any, loaders["train"].dataset).transform.set_random_state(seed=config.seed + epoch)
        model.train()
        total, steps = 0.0, 0
        for batch in loaders["train"]:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
        losses.append(total / steps)
        if epoch % config.val_interval and epoch != config.max_epochs:
            continue

        cases = case_dice(model, loaders["val"], config, device, use_amp)
        entry = {
            "epoch": epoch,
            "train_loss": losses[-1],
            "train_losses_since_last": losses,
            "val_cases": len(cases),
            "val_mean_dice": mean_dice(cases),
            "val_dice": cases,
            "seconds": round(time.monotonic() - started, 1),
        }
        losses = []
        history.append(entry)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "fingerprint": config.fingerprint(),
            "config": config.to_dict(),
            "versions": {"torch": str(torch.__version__)},
        }
        path = checkpoint_path(config.checkpoint_dir, epoch)
        save_new(path, partial(torch.save, state))
        summary = json.dumps(entry, indent=2) + "\n"
        save_new(path.with_suffix(".json"), partial(write_text, text=summary))
        dice = entry["val_mean_dice"]
        log(
            f"epoch {epoch}: train loss {entry['train_loss']:.4f}, val mean Dice "
            f"{'n/a' if dice is None else f'{dice:.4f}'}, saved {path.name}"
        )

    best = best_epoch(history, "val_mean_dice") if history else None
    if best is not None:
        log(f"best epoch {best['epoch']}: val mean Dice {best['val_mean_dice']:.4f}")
    return {"history": history, "best": best}


def evaluate_segmentation(
    config: SegConfig,
    splits: Sequence[str] = ("val", "test"),
    loaders: Mapping[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Per-case Dice of the best checkpoint on val and test."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latest = latest_checkpoint(config.checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint in {config.checkpoint_dir}")
    history = load_checkpoint(latest)["history"]
    best = best_epoch(history, "val_mean_dice")
    state = load_checkpoint(checkpoint_path(config.checkpoint_dir, best["epoch"]))
    if state["fingerprint"] != config.fingerprint():
        raise ValueError("the checkpoint was trained with another config")
    model = build_unet(config)
    model.load_state_dict(state["model"])
    model.to(device)
    loaders = loaders if loaders is not None else make_loaders(config, splits)
    amp = config.amp and device.type == "cuda"
    report: dict[str, Any] = {
        "notice": "For education only, not for clinical use.",
        "dataset": "Medical Segmentation Decathlon Task09_Spleen (labelled cases)",
        "checkpoint_epoch": best["epoch"],
        "metric": f"Dice of the spleen label, sliding-window inference at {config.pixdim} mm",
        "splits": {},
    }
    for split in splits:
        cases = case_dice(model, loaders[split], config, device, amp)
        print(f"{split}: {len(cases)} cases, mean Dice {mean_dice(cases)}")
        report["splits"][split] = {"cases": cases, "mean_dice": mean_dice(cases)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmentation.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, help="override the config's folder")
    parser.add_argument("--out", type=Path, default=Path("reports/segmentation_metrics.json"))
    args = parser.parse_args()
    overrides = {"checkpoint_dir": args.checkpoint_dir} if args.checkpoint_dir else {}
    config = load_seg_config(args.config, **overrides)
    if args.command == "train":
        train_segmentation(config)
    else:
        report = evaluate_segmentation(config)
        save_new(args.out, partial(write_text, text=json.dumps(report, indent=2) + "\n"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
