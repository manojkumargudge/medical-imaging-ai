"""MSD Task09_Spleen: download, case-level train/val/test split, and EDA of the volumes.

Only the 41 labelled volumes (`training` in dataset.json) are used; the 20 unlabelled test volumes
of the challenge have no ground truth. MSD gives no patient IDs, so each case (one CT volume) is
treated as one patient: a case is never in two splits.

Run:
    python -m src.data.spleen download --root data/raw
    python -m src.data.spleen split --task-dir data/raw/Task09_Spleen
        --out data/seg_splits/spleen.csv
    python -m src.data.spleen eda --task-dir data/raw/Task09_Spleen
        --split data/seg_splits/spleen.csv --out docs/eda.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

from src.data.split import SPLITS

TASK = "Task09_Spleen"
# The copy that MONAI's DecathlonDataset downloads, with its md5.
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"
EXPECTED_CASES = 41
_CASE = re.compile(r"^spleen_(\d+)\.nii\.gz$")


@dataclass(frozen=True)
class Case:
    case: str
    image: str  # relative to the task folder, e.g. imagesTr/spleen_2.nii.gz
    label: str


@dataclass(frozen=True)
class SplitCase:
    case: str
    image: str
    label: str
    split: str


COLUMNS = tuple(field.name for field in fields(SplitCase))


def download(root: Path) -> Path:
    """Download and extract the task (about 1.5 GB) into root; the md5 is checked."""
    from monai.apps.utils import download_and_extract

    task_dir = root / TASK
    if not (task_dir / "dataset.json").exists():
        # MONAI 1.6 defaults to sha256; the published hash for this file is md5.
        download_and_extract(
            URL, str(root / f"{TASK}.tar"), str(root), hash_val=MD5, hash_type="md5"
        )
    return task_dir


def load_cases(task_dir: Path, expected: int | None = EXPECTED_CASES) -> list[Case]:
    """The labelled cases listed in dataset.json, sorted by case number."""
    meta = json.loads((task_dir / "dataset.json").read_text(encoding="utf-8"))
    cases = []
    for entry in meta["training"]:
        image = PurePosixPath(entry["image"]).relative_to(".").as_posix()
        label = PurePosixPath(entry["label"]).relative_to(".").as_posix()
        name = PurePosixPath(image).name
        match = _CASE.match(name)
        if match is None or PurePosixPath(label).name != name:
            raise ValueError(f"unexpected entry {entry}")
        cases.append(Case(case=f"spleen_{int(match.group(1))}", image=image, label=label))
    names = [case.case for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("duplicate cases in dataset.json")
    if expected is not None and len(cases) != expected:
        raise ValueError(f"{len(cases)} labelled cases, expected {expected}")
    return sorted(cases, key=lambda case: int(case.case.split("_")[1]))


def split_cases(
    cases: Sequence[Case], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[SplitCase]]:
    """Seeded shuffle of the sorted cases; test and val get round(n * fraction) cases each."""
    n_test = round(len(cases) * test_fraction)
    n_val = round(len(cases) * val_fraction)
    if min(n_test, n_val) < 1 or n_test + n_val >= len(cases):
        raise ValueError(f"cannot split {len(cases)} cases into {n_val} val and {n_test} test")
    order = sorted(cases, key=lambda case: case.case)
    random.Random(seed).shuffle(order)
    assigned = {
        "test": order[:n_test],
        "val": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }
    return {
        split: [
            SplitCase(case.case, case.image, case.label, split)
            for case in sorted(assigned[split], key=lambda case: int(case.case.split("_")[1]))
        ]
        for split in SPLITS
    }


def check_no_overlap(splits: Mapping[str, Sequence[SplitCase]]) -> None:
    """Print counts; fail on an empty split or a case or file in two splits."""
    for split in SPLITS:
        print(f"{split}: {len(splits[split])} cases")
        if not splits[split]:
            raise ValueError(f"split {split} is empty")
    print(f"total: {sum(len(splits[split]) for split in SPLITS)} cases")
    for first, second in combinations(SPLITS, 2):
        for attribute in ("case", "image", "label"):
            overlap = {getattr(row, attribute) for row in splits[first]} & {
                getattr(row, attribute) for row in splits[second]
            }
            if attribute == "case":
                print(f"case overlap {first}/{second}: {len(overlap)}")
            if overlap:
                raise ValueError(f"{attribute} in both {first} and {second}: {sorted(overlap)}")


def write_split(splits: Mapping[str, Sequence[SplitCase]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(astuple(row) for split in SPLITS for row in splits[split])


def read_split(path: Path) -> dict[str, list[SplitCase]]:
    splits: dict[str, list[SplitCase]] = {split: [] for split in SPLITS}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {COLUMNS}")
        for row in reader:
            splits[row["split"]].append(SplitCase(**row))
    check_no_overlap(splits)
    return splits


def build_split(
    task_dir: Path, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[SplitCase]]:
    splits = split_cases(load_cases(task_dir), val_fraction, test_fraction, seed)
    check_no_overlap(splits)
    return splits


# EDA


def volume_stats(task_dir: Path, row: SplitCase) -> dict[str, Any]:
    """Shape, spacing, orientation, HU range and spleen volume of one case."""
    import nibabel as nib
    import numpy as np

    image = nib.load(task_dir / row.image)
    label = nib.load(task_dir / row.label)
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-3):
        raise ValueError(f"{row.case}: image and label grids differ")
    spacing = [float(value) for value in image.header.get_zooms()[:3]]
    pixels = np.asarray(image.dataobj)
    mask = np.asarray(label.dataobj)
    labels = sorted(int(value) for value in np.unique(mask))
    if not set(labels) <= {0, 1}:
        raise ValueError(f"{row.case}: label values {labels}, expected 0 and 1")
    voxel_ml = float(np.prod(spacing)) / 1000
    return {
        "case": row.case,
        "split": row.split,
        "shape": [int(value) for value in image.shape[:3]],
        "spacing": spacing,
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "hu_min": float(pixels.min()),
        "hu_max": float(pixels.max()),
        "spleen_ml": round(float(mask.sum()) * voxel_ml, 1),
        "spleen_slices": int(np.any(mask > 0, axis=(0, 1)).sum()),
    }


def _summary(values: Sequence[float], digits: int = 2) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.{digits}f}"
        # Trailing zeros only after a decimal point: 500 must stay 500.
        return text.rstrip("0").rstrip(".") if "." in text else text

    return f"{fmt(min(values))} / {fmt(statistics.median(values))} / {fmt(max(values))}"


def render_eda(stats: Sequence[Mapping[str, Any]], resample_mm: float = 1.5) -> str:
    """The spleen section of docs/eda.md: per-split min / median / max of each quantity."""
    groups = {split: [row for row in stats if row["split"] == split] for split in SPLITS}
    groups["all"] = list(stats)
    header = "| | " + " | ".join(groups) + " |"
    rule = "|---|" + "---|" * len(groups)

    def line(name: str, value: Any) -> str:
        return f"| {name} | " + " | ".join(value(rows) for rows in groups.values()) + " |"

    axes = ("x", "y", "z")
    in_plane = [r["spacing"][i] for r in stats for i in (0, 1)]
    slices = [r["spacing"][2] for r in stats]
    lines = [
        "# MSD Task09_Spleen: EDA of the split",
        "",
        "**For education only, not for clinical use.**",
        "",
        "Generated by `python -m src.data.spleen eda` from the NIfTI files and "
        "`data/seg_splits/spleen.csv`; do not edit by hand.",
        "",
        f"Data: the {len(stats)} labelled portal-venous CT volumes of the Medical Segmentation "
        "Decathlon Task09 (Memorial Sloan Kettering), label 1 = spleen. The split is by case, "
        "seeded (`params.yaml`, `seg_split`); MSD gives no patient IDs, so one case is treated as "
        "one patient. No case is in two splits.",
        "",
        "Cells are min / median / max over the volumes.",
        "",
        header,
        rule,
        line("volumes", lambda rows: str(len(rows))),
        *(
            line(f"size {axis} (voxels)", lambda rows, i=i: _summary([r["shape"][i] for r in rows]))
            for i, axis in enumerate(axes)
        ),
        *(
            line(
                f"spacing {axis} (mm)", lambda rows, i=i: _summary([r["spacing"][i] for r in rows])
            )
            for i, axis in enumerate(axes)
        ),
        *(
            line(
                f"extent {axis} (mm)",
                lambda rows, i=i: _summary(
                    [r["shape"][i] * r["spacing"][i] for r in rows], digits=0
                ),
            )
            for i, axis in enumerate(axes)
        ),
        *(
            line(
                f"size {axis} at {resample_mm:g} mm",
                lambda rows, i=i: _summary(
                    [round(r["shape"][i] * r["spacing"][i] / resample_mm) for r in rows], digits=0
                ),
            )
            for i, axis in enumerate(axes)
        ),
        line("spleen volume (ml)", lambda rows: _summary([r["spleen_ml"] for r in rows], 0)),
        line("slices with spleen", lambda rows: _summary([r["spleen_slices"] for r in rows], 0)),
        line("HU min", lambda rows: _summary([r["hu_min"] for r in rows], 0)),
        line("HU max", lambda rows: _summary([r["hu_max"] for r in rows], 0)),
        "",
        "Orientation codes (nibabel): "
        + ", ".join(
            f"{code} {sum(r['orientation'] == code for r in stats)}"
            for code in sorted({r["orientation"] for r in stats})
        )
        + ".",
        "",
        f"In-plane spacing ranges {min(in_plane):.2f} to {max(in_plane):.2f} mm and slice "
        f"spacing {min(slices):g} to {max(slices):g} mm, so "
        f"the model resamples every volume to {resample_mm:g} mm (src/models/segmentation.py) "
        "before cropping 96^3 patches; most volumes are upsampled along z.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.data.report import write_section

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download")
    get.add_argument("--root", type=Path, default=Path("data/raw"))
    make = commands.add_parser("split")
    make.add_argument("--task-dir", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--val-fraction", type=float, default=0.15)
    make.add_argument("--test-fraction", type=float, default=0.15)
    make.add_argument("--seed", type=int, default=42)
    eda = commands.add_parser("eda")
    eda.add_argument("--task-dir", type=Path, required=True)
    eda.add_argument("--split", type=Path, required=True)
    eda.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "download":
        print(f"task folder {download(args.root)}")
    elif args.command == "split":
        splits = build_split(args.task_dir, args.val_fraction, args.test_fraction, args.seed)
        write_split(splits, args.out)
        print(f"wrote {args.out}")
    else:
        rows = [row for split in SPLITS for row in read_split(args.split)[split]]
        stats = [volume_stats(args.task_dir, row) for row in rows]
        write_section(args.out, "spleen", render_eda(stats))
        print(f"wrote the spleen section of {args.out}")


if __name__ == "__main__":
    main()
