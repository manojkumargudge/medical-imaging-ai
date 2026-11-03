"""Resize images to 224px grayscale PNG and store them in zip shards with a checksum manifest.

DVC stage `preprocess`, run on Colab by notebooks/colab_preprocess.ipynb. Everything is written to
the Colab VM disk, never to Drive: Drive holds only the DVC remote, filled by `dvc push`. Resumable
within a VM session: a shard is recorded in the manifest only after it is complete, and a rerun
skips recorded shards after checking their sha256.

Pixels are stored as 8-bit grayscale, not as normalized floats: float32 RGB tensors would need about
67 GB instead of a few GB. The ImageNet mean and std are stage parameters, written to
`preprocess.json` next to the shards, and the training loader applies them (src/models/data.py).

Run: NIH_ZIP=/content/raw/data.zip dvc repro preprocess
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image

from src.data import split

IMAGE_SIZE = 224
SHARD_SIZE = 2_000
SHARDS_MANIFEST = "shards.csv"
IMAGES_MANIFEST = "images.csv"
PREPROCESS_INFO = "preprocess.json"
# Google Drive as mounted on Colab. Stage outputs and training data must not live under it.
DRIVE_MOUNT = "/content/drive"
# The training manifest: one row per image, with the split columns plus where the image is stored.
MANIFEST_COLUMNS = (*split.COLUMNS, "split", "shard", "sha256")

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


def check_off_drive(path: Path, what: str) -> None:
    """Fail if path is on the mounted Google Drive (slow, and its space is kept for the remote)."""
    # resolve() follows symlinks into Drive; the drive letter is dropped so tests behave the same
    # on Windows.
    posix = path.resolve().as_posix()
    if len(posix) > 1 and posix[1] == ":":
        posix = posix[2:]
    if posix == DRIVE_MOUNT or posix.startswith(DRIVE_MOUNT + "/"):
        raise ValueError(f"{what} {posix} is on Google Drive; use the VM disk")


def read_params(path: Path) -> dict[str, Any]:
    """The `preprocess` section of params.yaml, checked."""
    params = yaml.safe_load(path.read_text(encoding="utf-8"))["preprocess"]
    for key in ("mean", "std"):
        values = params[key]
        if len(values) != 3 or not all(isinstance(value, float) for value in values):
            raise ValueError(f"preprocess.{key} must be three floats, got {values!r}")
    if not all(value > 0 for value in params["std"]):
        raise ValueError("preprocess.std must be positive")
    return {
        "image_size": int(params["image_size"]),
        "shard_size": int(params["shard_size"]),
        "mean": [float(value) for value in params["mean"]],
        "std": [float(value) for value in params["std"]],
    }


def _write_new_text(path: Path, text: str) -> None:
    """Write a file; an existing one is accepted only if identical, never replaced."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} exists with other content; not overwriting")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, path)


def copy_shards(source: Path, target: Path) -> Report:
    """Copy verified shards and their manifests into an empty folder, then verify the copy."""
    verified = verify_shards(source)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"{target} is not empty; not overwriting")
    target.mkdir(parents=True, exist_ok=True)
    shards = [record.shard for record in _read_shard_records(source / SHARDS_MANIFEST)]
    for name in (*shards, SHARDS_MANIFEST, IMAGES_MANIFEST):
        partial = target / f"{name}.partial"
        shutil.copyfile(source / name, partial)
        os.replace(partial, target / name)
    copied = verify_shards(target)
    print(f"copied {source} -> {target}: {verified.images} -> {copied.images} images")
    if not (verified.images > 0 and copied.images == verified.images):
        raise ValueError(f"copy has {copied.images} images, {verified.images} expected")
    return copied


def build_manifest(splits: split.Splits, images_manifest: Path) -> list[tuple[object, ...]]:
    """Join the split rows with the shard of each image; fails on any image without a shard."""
    stored = {record.image: record for record in _read_image_records(images_manifest)}
    rows: list[tuple[object, ...]] = []
    for name in split.SPLITS:
        for entry in splits[name]:
            record = stored.get(entry.image)
            if record is None:
                raise ValueError(f"{entry.image} ({name}) is not in {images_manifest}")
            rows.append((*astuple(entry), name, record.shard, record.sha256))
    print(f"manifest rows {len(rows)}, stored images {len(stored)}")
    if not rows or len(rows) != len(stored):
        raise ValueError(f"{len(rows)} split rows but {len(stored)} stored images")
    return rows


def run_stage(
    zip_path: Path,
    splits_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    work_dir: Path,
    params: dict[str, Any],
    resume_dir: Path | None = None,
) -> Report:
    """Resize every split image from the dataset zip into verified shards plus a manifest.

    With resume_dir (a VM folder outside the DVC outputs), the shards are built in a subfolder named
    after the image and shard size, so an interrupted `dvc repro` (which deletes its outputs first)
    restarts from the last finished shard; they are copied into out_dir once complete and
    verified. No path may be on Google Drive.
    """
    for path, what in (
        (out_dir, "out_dir"),
        (manifest_path, "manifest"),
        (work_dir, "work_dir"),
        *([(resume_dir, "resume_dir")] if resume_dir is not None else []),
    ):
        check_off_drive(path, what)
    splits = split.read_splits(splits_dir)
    split.check_patient_level(splits)
    wanted = sorted(entry.image for name in split.SPLITS for entry in splits[name])

    images = ZipImages(zip_path)
    try:
        in_zip = set(images.names)
        print(f"images in zip {len(in_zip)}, images in splits {len(wanted)}")
        if not in_zip or not wanted:
            raise ValueError("no images in the zip or the splits")
        missing = sorted(set(wanted) - in_zip)
        if missing:
            raise ValueError(f"{len(missing)} split images are not in the zip, e.g. {missing[:3]}")
        if len(in_zip) != len(wanted):
            raise ValueError(f"the zip has {len(in_zip) - len(wanted)} images that are in no split")
        build_dir = out_dir
        if resume_dir is not None:
            # Keyed by the params, so shards of another size are never reused.
            build_dir = resume_dir / f"{params['image_size']}px-{params['shard_size']}"
        write_shards(
            wanted,
            images.read,
            build_dir,
            work_dir,
            size=params["image_size"],
            shard_size=params["shard_size"],
        )
    finally:
        images.close()

    report = copy_shards(build_dir, out_dir) if build_dir != out_dir else verify_shards(out_dir)
    rows = build_manifest(splits, out_dir / IMAGES_MANIFEST)
    lines = io.StringIO()
    writer = csv.writer(lines, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    writer.writerows(rows)
    _write_new_text(manifest_path, lines.getvalue())

    info = {
        "image_size": params["image_size"],
        "stored_as": "8-bit grayscale PNG; scale to [0, 1], copy to 3 channels, then normalize",
        "mean": params["mean"],
        "std": params["std"],
        "images": report.images,
        "shards": report.shards,
        "source_zip": zip_path.name,
        "source_zip_bytes": zip_path.stat().st_size,
    }
    _write_new_text(out_dir / PREPROCESS_INFO, json.dumps(info, indent=2) + "\n")
    print(f"{report.images} images in {report.shards} shards, {report.total_bytes / 1e9:.2f} GB")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="dataset zip ($NIH_ZIP in dvc.yaml)")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--resume-dir", default="", help="optional VM build folder that survives a failed run"
    )
    args = parser.parse_args()
    if not args.zip:
        parser.error("--zip is empty: set NIH_ZIP to the dataset zip")
    run_stage(
        Path(args.zip),
        args.splits_dir,
        args.out_dir,
        args.manifest,
        args.work_dir,
        read_params(args.params),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
    )


if __name__ == "__main__":
    main()
