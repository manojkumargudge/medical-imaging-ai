"""Training data: the preprocess manifest, a dataset that reads the shards, and the transforms.

Augmentation (small rotation and shift, brightness and contrast) is applied to the train split
only. Val and test get a deterministic pipeline: scale to [0, 1], copy to 3 channels, normalize
with the mean and std that the preprocess stage recorded.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.data.eda import NO_FINDING
from src.data.preprocess import MANIFEST_COLUMNS, PREPROCESS_INFO, check_off_drive
from src.data.split import SPLITS
from src.models.config import AugmentConfig, ClassifierConfig

Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class Sample:
    image: str
    patient: str
    split: str
    shard: str
    age: int
    sex: str
    targets: tuple[int, ...]


def read_manifest(path: Path, labels: Sequence[str]) -> dict[str, list[Sample]]:
    """Samples per split. Prints counts; fails on an empty split or a patient in two splits."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {MANIFEST_COLUMNS}")
        rows = list(reader)
    known = {*labels, NO_FINDING}
    samples: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    for row in rows:
        names = set(row["labels"].split("|"))
        if names - known:
            raise ValueError(f"{row['image']}: unknown labels {sorted(names - known)}")
        samples[row["split"]].append(
            Sample(
                image=row["image"],
                patient=row["patient"],
                split=row["split"],
                shard=row["shard"],
                age=int(row["age"]),
                sex=row["sex"],
                targets=tuple(int(label in names) for label in labels),
            )
        )
    check_patient_level(samples)
    return samples


def check_patient_level(samples: Mapping[str, Sequence[Sample]]) -> None:
    patients = {split: {sample.patient for sample in samples[split]} for split in SPLITS}
    for split in SPLITS:
        print(f"{split}: {len(samples[split])} images, {len(patients[split])} patients")
        if not samples[split]:
            raise ValueError(f"split {split} is empty")
    for first, second in combinations(SPLITS, 2):
        overlap = patients[first] & patients[second]
        print(f"patient overlap {first}/{second}: {len(overlap)}")
        if overlap:
            raise ValueError(f"{len(overlap)} patients are in both {first} and {second}")


def read_preprocess_info(processed_dir: Path) -> dict[str, Any]:
    """image_size, mean and std recorded by the preprocess stage."""
    info: dict[str, Any] = json.loads((processed_dir / PREPROCESS_INFO).read_text("utf-8"))
    return info


def to_three_channels(image: torch.Tensor) -> torch.Tensor:
    # ImageNet weights expect RGB; the X-ray is grayscale, so the channel is copied.
    return image.expand(3, -1, -1)


def build_transform(
    train: bool, mean: Sequence[float], std: Sequence[float], augment: AugmentConfig
) -> v2.Compose:
    steps: list[Any] = [v2.ToImage()]
    if train:
        steps.append(
            v2.RandomAffine(
                degrees=augment.rotation_degrees, translate=(augment.translate, augment.translate)
            )
        )
    steps.append(v2.ToDtype(torch.float32, scale=True))
    if train:
        steps.append(v2.ColorJitter(brightness=augment.brightness, contrast=augment.contrast))
    steps += [v2.Lambda(to_three_channels), v2.Normalize(mean=list(mean), std=list(std))]
    return v2.Compose(steps)


def transform_for_split(
    split: str, mean: Sequence[float], std: Sequence[float], augment: AugmentConfig
) -> v2.Compose:
    """Augmentation for train only; val and test are never augmented."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return build_transform(split == "train", mean, std, augment)


class ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Images read straight from the preprocess shards (stored zips), opened lazily per process."""

    def __init__(
        self, samples: Sequence[Sample], shards_dir: Path, transform: Transform, image_size: int
    ) -> None:
        if not samples:
            raise ValueError("no samples")
        self.samples = list(samples)
        self.shards_dir = shards_dir
        self.transform = transform
        self.image_size = image_size
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self.samples)

    def _shard(self, name: str) -> zipfile.ZipFile:
        # A forked DataLoader worker must not share the parent's file handles (shared offsets).
        if os.getpid() != self._pid:
            self._zips, self._pid = {}, os.getpid()
        if name not in self._zips:
            self._zips[name] = zipfile.ZipFile(self.shards_dir / name)
        return self._zips[name]

    def load_image(self, index: int) -> Image.Image:
        sample = self.samples[index]
        with Image.open(io.BytesIO(self._shard(sample.shard).read(sample.image))) as image:
            image.load()
            if image.mode != "L" or image.size != (self.image_size, self.image_size):
                raise ValueError(
                    f"{sample.image}: {image.mode} {image.size}, expected L {self.image_size}px"
                )
            return image.copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.transform(self.load_image(index)).as_subclass(torch.Tensor)
        targets = torch.tensor(self.samples[index].targets, dtype=torch.float32)
        return pixels, targets

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_zips": {}}


def make_datasets(
    config: ClassifierConfig, splits: Sequence[str] = SPLITS
) -> dict[str, ShardDataset]:
    # Training reads from the VM disk only (after `dvc pull`), never from Drive.
    check_off_drive(config.processed_dir, "processed_dir")
    check_off_drive(config.manifest, "manifest")
    info = read_preprocess_info(config.processed_dir)
    samples = read_manifest(config.manifest, config.labels)
    return {
        split: ShardDataset(
            samples[split],
            config.processed_dir,
            cast(Transform, transform_for_split(split, info["mean"], info["std"], config.augment)),
            int(info["image_size"]),
        )
        for split in splits
    }
