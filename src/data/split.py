"""Patient-level train/val/test split of NIH ChestX-ray14.

The test split is the official test_list.txt. Validation is carved from the official
train_val_list.txt by patient, so no patient is in two splits. Standard library only, so the DVC
stage runs with any Python 3.11 on the path.

Run: python -m src.data.split --data-entry data/raw/Data_Entry_2017.csv
         --train-val-list data/raw/train_val_list.txt --test-list data/raw/test_list.txt
         --out-dir data/splits
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from itertools import combinations
from pathlib import Path

SPLITS = ("train", "val", "test")
EXPECTED_IMAGES = 112_120

_SOURCE_COLUMNS = {
    "Image Index": "image",
    "Patient ID": "patient",
    "Finding Labels": "labels",
    "Patient Age": "age",
    "Patient Gender": "sex",
    "View Position": "view",
}


@dataclass(frozen=True)
class Entry:
    """One image row. `age` is kept as recorded; the EDA reports implausible values."""

    image: str
    patient: str
    labels: str
    age: int
    sex: str
    view: str


COLUMNS = tuple(field.name for field in fields(Entry))
Splits = Mapping[str, Sequence[Entry]]


def load_entries(path: Path) -> list[Entry]:
    """Read Data_Entry_2017.csv and validate the columns the split needs."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(_SOURCE_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")

    entries = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        values = {target: (row[source] or "").strip() for source, target in _SOURCE_COLUMNS.items()}
        empty = sorted(name for name, value in values.items() if not value)
        if empty:
            raise ValueError(f"{path}:{line}: empty {empty}")
        if values["image"] in seen:
            raise ValueError(f"{path}:{line}: duplicate image {values['image']}")
        seen.add(values["image"])
        if values["sex"] not in {"M", "F"}:
            raise ValueError(f"{path}:{line}: unexpected sex {values['sex']!r}")
        if not values["age"].isdigit():
            raise ValueError(f"{path}:{line}: age is not a whole number: {values['age']!r}")
        entries.append(
            Entry(
                image=values["image"],
                patient=values["patient"],
                labels=values["labels"],
                age=int(values["age"]),
                sex=values["sex"],
                view=values["view"],
            )
        )
    return entries


def read_image_list(path: Path) -> list[str]:
    """Read an official split list (one image name per line)."""
    images = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    images = [image for image in images if image]
    if not images:
        raise ValueError(f"{path}: empty image list")
    if len(set(images)) != len(images):
        raise ValueError(f"{path}: duplicate image names")
    return images


def split_by_patient(
    entries: Sequence[Entry],
    train_val_images: Iterable[str],
    test_images: Iterable[str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Entry]]:
    """Keep the official test list; move a seeded share of train_val patients to val."""
    train_val = set(train_val_images)
    test = set(test_images)
    known = {entry.image for entry in entries}
    if train_val & test:
        raise ValueError(f"{len(train_val & test)} images are in both official lists")
    if known != train_val | test:
        raise ValueError(
            f"official lists do not match the CSV: {len(known - train_val - test)} images only in "
            f"the CSV, {len((train_val | test) - known)} only in the lists"
        )
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    # Sorted first, so the result depends only on the seed and not on the row order.
    patients = sorted({entry.patient for entry in entries if entry.image in train_val})
    n_val = round(len(patients) * val_fraction)
    if not 0 < n_val < len(patients):
        raise ValueError(f"cannot carve val: {len(patients)} patients, fraction {val_fraction}")
    val_patients = set(random.Random(seed).sample(patients, n_val))

    splits: dict[str, list[Entry]] = {split: [] for split in SPLITS}
    for entry in sorted(entries, key=lambda entry: entry.image):
        if entry.image in test:
            splits["test"].append(entry)
        elif entry.patient in val_patients:
            splits["val"].append(entry)
        else:
            splits["train"].append(entry)
    return splits


def split_counts(splits: Splits) -> dict[str, tuple[int, int]]:
    """(rows, patients) per split."""
    return {
        split: (len(splits[split]), len({entry.patient for entry in splits[split]}))
        for split in SPLITS
    }


def check_patient_level(splits: Splits) -> None:
    """Print row and patient counts; fail on an empty split or a patient in two splits."""
    counts = split_counts(splits)
    for split, (rows, n_patients) in counts.items():
        print(f"{split}: {rows} rows, {n_patients} patients")
    print(
        f"total: {sum(rows for rows, _ in counts.values())} rows, "
        f"{len({entry.patient for split in SPLITS for entry in splits[split]})} patients"
    )
    empty = [split for split, (rows, _) in counts.items() if rows == 0]
    if empty:
        raise ValueError(f"empty splits: {empty}")
    patients = {split: {entry.patient for entry in splits[split]} for split in SPLITS}
    for first, second in combinations(SPLITS, 2):
        overlap = patients[first] & patients[second]
        print(f"patient overlap {first}/{second}: {len(overlap)}")
        if overlap:
            raise ValueError(f"{len(overlap)} patients are in both {first} and {second}")


def write_splits(splits: Splits, out_dir: Path) -> list[Path]:
    """Write one CSV per split (columns COLUMNS) and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in SPLITS:
        path = out_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(astuple(entry) for entry in splits[split])
        paths.append(path)
    return paths


def read_splits(splits_dir: Path) -> dict[str, list[Entry]]:
    """Read the CSVs written by write_splits."""
    splits: dict[str, list[Entry]] = {}
    for split in SPLITS:
        with (splits_dir / f"{split}.csv").open(newline="", encoding="utf-8") as file:
            splits[split] = [
                Entry(**{**row, "age": int(row["age"])})  # type: ignore[arg-type]
                for row in csv.DictReader(file)
            ]
    return splits


def build_splits(
    data_entry: Path,
    train_val_list: Path,
    test_list: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    expected_images: int | None = EXPECTED_IMAGES,
) -> dict[str, list[Entry]]:
    """Load, split and check. Raises before anything is written if a check fails."""
    entries = load_entries(data_entry)
    if expected_images is not None and len(entries) != expected_images:
        raise ValueError(f"{data_entry}: {len(entries)} rows, expected {expected_images}")
    splits = split_by_patient(
        entries,
        read_image_list(train_val_list),
        read_image_list(test_list),
        val_fraction=val_fraction,
        seed=seed,
    )
    check_patient_level(splits)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-entry", type=Path, required=True)
    parser.add_argument("--train-val-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    args = parser.parse_args()

    splits = build_splits(
        args.data_entry,
        args.train_val_list,
        args.test_list,
        val_fraction=args.val_fraction,
        seed=args.seed,
        expected_images=args.expected_images,
    )
    for path in write_splits(splits, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
