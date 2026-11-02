"""Resize images to grayscale and store them as size-limited .npz shards.

Pixels are stored as horizontal byte differences, the same idea as a PNG filter: it is lossless and
compresses much better than raw pixels. Read shards with `read_shard`, which undoes it.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

IMAGE_SIZE = 224
SHARD_SIZE = 5_000
MAX_TOTAL_BYTES = 4_000_000_000
ENCODING = "delta-horizontal-v1"

_ALLOWED_MODES = {"L", "RGB", "RGBA"}

Images = NDArray[np.uint8]


class ShardSizeLimitError(RuntimeError):
    """The shards are projected to reach the size limit."""


@dataclass(frozen=True)
class ShardReport:
    images: int
    shards: int
    total_bytes: int


def preprocess_png(data: bytes, size: int = IMAGE_SIZE) -> Images:
    """Decode an image and return it as a size x size uint8 grayscale array."""
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in _ALLOWED_MODES:
            raise ValueError(f"unexpected image mode {image.mode}")
        resized = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)


def delta_encode(images: Images) -> Images:
    """Replace each pixel by its difference from the left neighbour (wraps at 256)."""
    encoded = images.copy()
    encoded[..., 1:] = images[..., 1:] - images[..., :-1]
    return encoded


def delta_decode(encoded: Images) -> Images:
    """Undo delta_encode."""
    return np.cumsum(encoded, axis=-1, dtype=np.uint8)


def read_shard(path: Path) -> tuple[Images, list[str]]:
    """Return a shard's images (N x H x W, uint8) and image names."""
    with np.load(path, allow_pickle=False) as shard:
        encoding = str(shard["encoding"])
        if encoding != ENCODING:
            raise ValueError(f"{path}: unknown encoding {encoding}")
        return delta_decode(shard["images"]), [str(name) for name in shard["names"]]


def plan_shards(manifest: pd.DataFrame, shard_size: int = SHARD_SIZE) -> dict[str, list[str]]:
    """Map shard file names to image names, per split, in manifest order."""
    if manifest.empty:
        raise ValueError("manifest has no rows")
    plan: dict[str, list[str]] = {}
    for split, rows in manifest.groupby("split", sort=True):
        images = rows["image"].astype(str).tolist()
        for index, start in enumerate(range(0, len(images), shard_size)):
            plan[f"{split}-{index:04d}.npz"] = images[start : start + shard_size]
    return plan


def write_shards(
    manifest: pd.DataFrame,
    read_image: Callable[[str], bytes],
    out_dir: Path,
    work_dir: Path,
    size: int = IMAGE_SIZE,
    shard_size: int = SHARD_SIZE,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> ShardReport:
    """Write all shards, keeping finished ones, so an interrupted run can be repeated.

    Each shard is built in work_dir (local disk) and then moved into out_dir (Drive). After every
    shard the final size is projected from the bytes per image so far; the run stops if the
    projection reaches max_total_bytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_shards(manifest, shard_size)
    total_images = sum(len(names) for names in plan.values())
    done_images = 0
    total_bytes = 0

    for number, (name, images) in enumerate(plan.items(), start=1):
        target = out_dir / name
        if target.exists():
            _, stored_names = read_shard(target)
            if stored_names != images:
                raise ValueError(f"{target} holds different images than the manifest plans")
        else:
            pixels = np.stack([preprocess_png(read_image(image), size) for image in images])
            local = work_dir / name
            np.savez_compressed(
                local,
                images=delta_encode(pixels),
                names=np.array(images),
                encoding=np.array(ENCODING),
            )
            partial = out_dir / f"{name}.partial"
            shutil.copyfile(local, partial)
            os.replace(partial, target)
            local.unlink()

        done_images += len(images)
        total_bytes += target.stat().st_size
        projected = round(total_bytes / done_images * total_images)
        print(
            f"shard {number}/{len(plan)} {name}: {done_images}/{total_images} images, "
            f"{total_bytes / 1e9:.2f} GB, projected {projected / 1e9:.2f} GB"
        )
        if projected >= max_total_bytes:
            raise ShardSizeLimitError(
                f"projected {projected / 1e9:.2f} GB reaches the "
                f"{max_total_bytes / 1e9:.2f} GB limit after {done_images} images"
            )

    return ShardReport(images=done_images, shards=len(plan), total_bytes=total_bytes)


class ZipImages:
    """Read images straight from the dataset zip, so nothing has to be extracted."""

    def __init__(self, path: Path) -> None:
        self._zip = zipfile.ZipFile(path)
        self._members: dict[str, str] = {}
        for member in self._zip.namelist():
            if member.lower().endswith(".png"):
                base = PurePosixPath(member).name
                if base in self._members:
                    raise ValueError(f"{path}: image {base} appears twice")
                self._members[base] = member

    @property
    def names(self) -> set[str]:
        return set(self._members)

    def read(self, image: str) -> bytes:
        return self._zip.read(self._members[image])

    def extract_file(self, base_name: str, out_dir: Path) -> Path:
        """Copy one non-image file, found by base name anywhere in the zip, to out_dir."""
        matches = [m for m in self._zip.namelist() if PurePosixPath(m).name == base_name]
        if len(matches) != 1:
            raise ValueError(f"expected one {base_name} in the zip, found {len(matches)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / base_name
        target.write_bytes(self._zip.read(matches[0]))
        return target

    def close(self) -> None:
        self._zip.close()
