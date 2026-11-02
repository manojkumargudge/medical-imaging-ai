"""NIH ChestX-ray14: labels, patient-level splits, manifest and EDA tables.

Run: python -m src.data.nih --data-entry Data_Entry_2017.csv --train-val-list train_val_list.txt
         --test-list test_list.txt --out data/manifests/nih_manifest.csv
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)
NO_FINDING = "No Finding"
LABEL_COLUMNS = (*FINDINGS, NO_FINDING)
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ("image", "patient", "split", "labels", "age", "sex", *LABEL_COLUMNS)

EXPECTED_IMAGES = 112_120
MAX_VALID_AGE = 120

_REQUIRED_COLUMNS = {
    "Image Index": "image",
    "Finding Labels": "labels",
    "Patient ID": "patient",
    "Patient Age": "age",
    "Patient Gender": "sex",
}
_AGE_PATTERN = re.compile(r"^(\d+)Y?$")


def parse_labels(labels: pd.Series) -> pd.DataFrame:
    """Turn pipe-separated labels into 14 multi-hot findings + No Finding."""
    positions = {name: i for i, name in enumerate(LABEL_COLUMNS)}
    matrix = np.zeros((len(labels), len(LABEL_COLUMNS)), dtype=np.int8)
    for row, raw in enumerate(labels.astype(str)):
        names = {name.strip() for name in raw.split("|")}
        unknown = names - positions.keys()
        if unknown:
            raise ValueError(f"unknown labels {sorted(unknown)} in row {row}: {raw!r}")
        if NO_FINDING in names and len(names) > 1:
            raise ValueError(f"{NO_FINDING!r} combined with findings in row {row}: {raw!r}")
        for name in names:
            matrix[row, positions[name]] = 1
    return pd.DataFrame(matrix, columns=list(LABEL_COLUMNS), index=labels.index)


def parse_age(raw: str) -> int | None:
    """Return age in years, or None if it is not a plausible age in years."""
    match = _AGE_PATTERN.match(raw.strip())
    if match is None:
        return None
    age = int(match.group(1))
    return age if age <= MAX_VALID_AGE else None


def load_data_entry(path: Path) -> pd.DataFrame:
    """Read Data_Entry_2017.csv into columns image, labels, patient, age, sex."""
    raw = pd.read_csv(path, dtype=str)
    missing = sorted(set(_REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    entries = raw[list(_REQUIRED_COLUMNS)].rename(columns=_REQUIRED_COLUMNS)
    if entries.empty:
        raise ValueError(f"{path}: no rows")
    if entries[["image", "labels", "patient", "sex"]].isna().any(axis=None):
        raise ValueError(f"{path}: empty image, labels, patient or sex values")
    duplicates = entries["image"][entries["image"].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"{path}: duplicate images, first {duplicates.iloc[0]}")
    bad_sex = sorted(set(entries["sex"]) - {"M", "F"})
    if bad_sex:
        raise ValueError(f"{path}: unexpected sex values {bad_sex}")
    entries["age"] = pd.array(
        [parse_age(value) for value in entries["age"].fillna("")], dtype="Int64"
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


def assign_splits(
    entries: pd.DataFrame,
    train_val_images: list[str],
    test_images: list[str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Use the official test list; carve val from the official train_val list by patient."""
    train_val = set(train_val_images)
    test = set(test_images)
    known = set(entries["image"])
    if train_val & test:
        raise ValueError(f"{len(train_val & test)} images are in both official lists")
    if known != train_val | test:
        raise ValueError(
            f"official lists do not match the CSV: {len(known - train_val - test)} images only in "
            f"the CSV, {len((train_val | test) - known)} only in the lists"
        )
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    is_test = entries["image"].isin(test)
    # Sorted first, so the carve depends only on the seed and not on the row order.
    patients = np.array(sorted(set(entries.loc[~is_test, "patient"])))
    n_val = round(len(patients) * val_fraction)
    if not 0 < n_val < len(patients):
        raise ValueError(f"cannot carve val: {len(patients)} patients, fraction {val_fraction}")
    val_patients = set(np.random.default_rng(seed).permutation(patients)[:n_val].tolist())

    splits = pd.Series("train", index=entries.index, name="split")
    splits[entries["patient"].isin(val_patients) & ~is_test] = "val"
    splits[is_test] = "test"
    return splits


def split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Rows and patients per split, in SPLITS order."""
    grouped = manifest.groupby("split")["patient"]
    counts = pd.DataFrame({"rows": grouped.size(), "patients": grouped.nunique()})
    return counts.reindex(list(SPLITS), fill_value=0)


def assert_patient_level_splits(manifest: pd.DataFrame) -> None:
    """Print row and patient counts; fail on an empty split or a patient in two splits."""
    counts = split_counts(manifest)
    print(counts.to_string())
    print(f"total rows {len(manifest)}, total patients {manifest['patient'].nunique()}")
    empty = [split for split in SPLITS if counts.loc[split, "rows"] == 0]
    if empty:
        raise ValueError(f"empty splits: {empty}")
    patients = {split: set(manifest.loc[manifest["split"] == split, "patient"]) for split in SPLITS}
    for first, second in combinations(SPLITS, 2):
        overlap = patients[first] & patients[second]
        print(f"patient overlap {first}/{second}: {len(overlap)}")
        if overlap:
            raise ValueError(f"{len(overlap)} patients are in both {first} and {second}")


def build_manifest(
    data_entry: Path,
    train_val_list: Path,
    test_list: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    expected_images: int | None = EXPECTED_IMAGES,
) -> pd.DataFrame:
    """Build the checked manifest: one row per image, sorted by image name."""
    entries = load_data_entry(data_entry)
    if expected_images is not None and len(entries) != expected_images:
        raise ValueError(f"{data_entry}: {len(entries)} rows, expected {expected_images}")
    splits = assign_splits(
        entries,
        read_image_list(train_val_list),
        read_image_list(test_list),
        val_fraction=val_fraction,
        seed=seed,
    )
    manifest = pd.concat([entries, splits, parse_labels(entries["labels"])], axis=1)
    manifest = manifest[list(MANIFEST_COLUMNS)].sort_values("image", ignore_index=True)
    assert_patient_level_splits(manifest)
    return manifest


def label_prevalence(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count and percent of images with each label, per split and overall."""
    columns: dict[str, pd.Series] = {}
    for split in (*SPLITS, "all"):
        rows = manifest if split == "all" else manifest[manifest["split"] == split]
        counts = rows[list(LABEL_COLUMNS)].sum()
        columns[f"{split}_count"] = counts
        columns[f"{split}_percent"] = (100 * counts / len(rows)).round(2)
    return pd.DataFrame(columns).rename_axis("label")


def age_sex_by_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Age summary and sex share per split, counted per image."""
    rows = []
    for split in (*SPLITS, "all"):
        part = manifest if split == "all" else manifest[manifest["split"] == split]
        age = part["age"].dropna().astype(float)
        rows.append(
            {
                "split": split,
                "images": len(part),
                "age_missing": int(part["age"].isna().sum()),
                "age_mean": round(float(age.mean()), 2),
                "age_median": float(age.median()),
                "age_min": float(age.min()),
                "age_max": float(age.max()),
                "male_percent": round(100 * float((part["sex"] == "M").mean()), 2),
                "female_percent": round(100 * float((part["sex"] == "F").mean()), 2),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def images_per_patient(manifest: pd.DataFrame) -> pd.DataFrame:
    """Distribution of images per patient, per split and overall."""
    rows = []
    for split in (*SPLITS, "all"):
        part = manifest if split == "all" else manifest[manifest["split"] == split]
        counts = part.groupby("patient").size()
        rows.append(
            {
                "split": split,
                "patients": len(counts),
                "mean": round(float(counts.mean()), 2),
                "median": float(counts.median()),
                "p95": float(counts.quantile(0.95)),
                "max": int(counts.max()),
                "single_image_percent": round(100 * float((counts == 1).mean()), 2),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def write_eda_tables(manifest: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write the EDA tables as CSV files and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "label_prevalence.csv": label_prevalence(manifest),
        "age_sex_by_split.csv": age_sex_by_split(manifest),
        "images_per_patient.csv": images_per_patient(manifest),
    }
    paths = []
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-entry", type=Path, required=True)
    parser.add_argument("--train-val-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    args = parser.parse_args()

    manifest = build_manifest(
        args.data_entry,
        args.train_val_list,
        args.test_list,
        val_fraction=args.val_fraction,
        seed=args.seed,
        expected_images=args.expected_images,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(manifest)} rows")


if __name__ == "__main__":
    main()
