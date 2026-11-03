"""Resize images to 224px grayscale PNG and store them in zip shards with a checksum manifest.

Used by notebooks/colab_preprocess.ipynb. Resumable: a shard is recorded in the manifest only after
it is complete in the output folder, and a rerun skips recorded shards after checking their sha256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

IMAGE_SIZE = 224
SHARD_SIZE = 2_000
SHARDS_MANIFEST = "shards.csv"
IMAGES_MANIFEST = "images.csv"

_ALLOWED_MODES = {"L", "RGB", "RGBA"}
# Fixed timestamp, so the same images always give the same shard bytes and sha256.
_ZIP_DATE = (2020, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ShardRecord:
    shard: str
    images: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ImageRecord:
    image: str
    shard: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Report:
    shards: int
    images: int
    total_bytes: int
    written: int
    skipped: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_png(data: bytes, size: int = IMAGE_SIZE) -> bytes:
    """Decode an image and return it as a size x size 8-bit grayscale PNG."""
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in _ALLOWED_MODES:
            raise ValueError(f"unexpected image mode {image.mode}")
        resized = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def plan_shards(images: Sequence[str], shard_size: int = SHARD_SIZE) -> dict[str, list[str]]:
    """Map shard names to sorted image names."""
    if not images:
        raise ValueError("no images to shard")
    if len(set(images)) != len(images):
        raise ValueError("duplicate image names")
    ordered = sorted(images)
    return {
        f"shard-{index:05d}.zip": ordered[start : start + shard_size]
        for index, start in enumerate(range(0, len(ordered), shard_size))
    }


def _build_shard(
    path: Path, images: Sequence[str], read_image: Callable[[str], bytes], size: int
) -> list[tuple[str, int, str]]:
    entries = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for image in images:
            png = resize_png(read_image(image), size)
            archive.writestr(zipfile.ZipInfo(image, date_time=_ZIP_DATE), png)
            entries.append((image, len(png), hashlib.sha256(png).hexdigest()))
    return entries


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_shard_records(path: Path) -> list[ShardRecord]:
    return [
        ShardRecord(row["shard"], int(row["images"]), int(row["bytes"]), row["sha256"])
        for row in _read_rows(path)
    ]


def _read_image_records(path: Path) -> list[ImageRecord]:
    return [
        ImageRecord(row["image"], row["shard"], int(row["bytes"]), row["sha256"])
        for row in _read_rows(path)
    ]


def _write_records(path: Path, records: Sequence[object], columns: Sequence[str]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([getattr(record, column) for column in columns] for record in records)
    os.replace(partial, path)


def write_shards(
    images: Sequence[str],
    read_image: Callable[[str], bytes],
    out_dir: Path,
    work_dir: Path,
    size: int = IMAGE_SIZE,
    shard_size: int = SHARD_SIZE,
) -> Report:
    """Write every planned shard that is not already recorded, then return totals.

    Each shard is built in work_dir (local disk), copied into out_dir (Drive) under a temporary
    name, renamed, and only then added to the manifests. A rerun with a different image list or
    shard size fails instead of mixing two plans.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_shards(images, shard_size)
    shards_path = out_dir / SHARDS_MANIFEST
    images_path = out_dir / IMAGES_MANIFEST
    shard_records = {record.shard: record for record in _read_shard_records(shards_path)}
    image_records = _read_image_records(images_path)

    unknown = set(shard_records) - set(plan)
    if unknown:
        raise ValueError(f"manifest lists shards that are not in the plan: {sorted(unknown)[:3]}")
    planned_images = {name: shard for shard, names in plan.items() for name in names}
    for image_record in image_records:
        if planned_images.get(image_record.image) != image_record.shard:
            raise ValueError(f"manifest has {image_record.image} in another shard than the plan")

    written = skipped = 0
    for number, (shard, names) in enumerate(plan.items(), start=1):
        target = out_dir / shard
        recorded = shard_records.get(shard)
        if recorded is not None:
            if not target.exists():
                raise FileNotFoundError(f"{target} is in the manifest but missing")
            if recorded.images != len(names) or sha256_file(target) != recorded.sha256:
                raise ValueError(f"{target} does not match its manifest entry")
            skipped += 1
            continue

        local = work_dir / shard
        entries = _build_shard(local, names, read_image, size)
        digest = sha256_file(local)
        if target.exists() and sha256_file(target) != digest:
            # A finished copy from an interrupted run is fine only if it is byte-identical.
            raise ValueError(
                f"{target} exists, is not in the manifest and differs; not overwriting"
            )
        partial = out_dir / f"{shard}.partial"
        shutil.copyfile(local, partial)
        os.replace(partial, target)
        local.unlink()

        shard_records[shard] = ShardRecord(shard, len(names), target.stat().st_size, digest)
        image_records += [ImageRecord(name, shard, n, sha) for name, n, sha in entries]
        _write_records(images_path, image_records, ("image", "shard", "bytes", "sha256"))
        _write_records(
            shards_path,
            [shard_records[name] for name in plan if name in shard_records],
            ("shard", "images", "bytes", "sha256"),
        )
        written += 1
        done = sum(record.images for record in shard_records.values())
        total = sum(record.bytes for record in shard_records.values())
        print(f"shard {number}/{len(plan)} {shard}: {done} images, {total / 1e9:.2f} GB so far")

    return Report(
        shards=len(shard_records),
        images=sum(record.images for record in shard_records.values()),
        total_bytes=sum(record.bytes for record in shard_records.values()),
        written=written,
        skipped=skipped,
    )


def verify_shards(out_dir: Path) -> Report:
    """Recheck every shard and image against the manifests; fail on any mismatch."""
    shard_records = _read_shard_records(out_dir / SHARDS_MANIFEST)
    image_records = _read_image_records(out_dir / IMAGES_MANIFEST)
    if not shard_records or not image_records:
        raise ValueError(f"{out_dir}: empty manifest")
    expected = {(record.shard, record.image): record.sha256 for record in image_records}
    seen = 0
    for shard_record in shard_records:
        path = out_dir / shard_record.shard
        if sha256_file(path) != shard_record.sha256:
            raise ValueError(f"{path}: sha256 differs from the manifest")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != shard_record.images:
                raise ValueError(
                    f"{path}: {len(names)} images, manifest says {shard_record.images}"
                )
            for name in names:
                digest = hashlib.sha256(archive.read(name)).hexdigest()
                if expected.get((shard_record.shard, name)) != digest:
                    raise ValueError(f"{path}/{name}: sha256 differs from the manifest")
                seen += 1
    if seen != len(image_records):
        raise ValueError(f"{seen} images in shards, {len(image_records)} in the manifest")
    return Report(
        shards=len(shard_records),
        images=seen,
        total_bytes=sum(record.bytes for record in shard_records),
        written=0,
        skipped=len(shard_records),
    )


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
    def names(self) -> list[str]:
        return sorted(self._members)

    def read(self, image: str) -> bytes:
        return self._zip.read(self._members[image])

    def close(self) -> None:
        self._zip.close()
