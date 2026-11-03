"""EDA of the NIH ChestX-ray14 splits, written as a Markdown report.

Run: python -m src.data.eda --splits-dir data/splits --out docs/eda.md (replaces the NIH section)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.report import write_section
from src.data.split import COLUMNS, SPLITS

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
LABELS = (*FINDINGS, NO_FINDING)
# Ages above this are treated as recording errors.
MAX_PLAUSIBLE_AGE = 100
AGE_BINS = [0, 18, 40, 60, 80, MAX_PLAUSIBLE_AGE + 1]
AGE_BIN_NAMES = ["0-17", "18-39", "40-59", "60-79", f"80-{MAX_PLAUSIBLE_AGE}"]
GROUPS = (*SPLITS, "all")


def load_splits(splits_dir: Path) -> pd.DataFrame:
    """All split CSVs in one frame with a `split` column; fails on a missing or empty split."""
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(splits_dir / f"{split}.csv", dtype={"patient": str, "image": str})
        if list(frame.columns) != list(COLUMNS):
            raise ValueError(f"{split}.csv: columns {list(frame.columns)}, expected {COLUMNS}")
        if frame.empty:
            raise ValueError(f"{split}.csv has no rows")
        frames.append(frame.assign(split=split))
    return pd.concat(frames, ignore_index=True)


def multi_hot(labels: pd.Series) -> pd.DataFrame:
    """One 0/1 column per label in LABELS; fails on unknown labels."""
    names = labels.str.split("|")
    unknown = set(names.explode()) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    return pd.DataFrame(
        {label: names.map(lambda row, label=label: int(label in row)) for label in LABELS},
        index=labels.index,
    )


def _groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {group: frame if group == "all" else frame[frame["split"] == group] for group in GROUPS}


def split_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {
        group: {
            "images": len(part),
            "patients": part["patient"].nunique(),
            "images %": round(100 * len(part) / len(frame), 2),
        }
        for group, part in _groups(frame).items()
    }
    return pd.DataFrame(rows).T.astype({"images": int, "patients": int})


def label_prevalence(frame: pd.DataFrame) -> pd.DataFrame:
    """Percent of images with each label, per split, sorted by the overall percent."""
    hot = multi_hot(frame["labels"])
    table = pd.DataFrame(
        {
            f"{group} %": (100 * hot.loc[part.index].mean()).round(2)
            for group, part in _groups(frame).items()
        }
    )
    table.insert(0, "all count", hot.sum())
    return table.sort_values("all %", ascending=False).rename_axis("label")


def no_finding_share(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of images labelled only No Finding, and images with 2+ findings."""
    hot = multi_hot(frame["labels"])
    findings = hot[list(FINDINGS)].sum(axis=1)
    rows = {
        group: {
            "No Finding %": round(100 * float(hot.loc[part.index, NO_FINDING].mean()), 2),
            "1 finding %": round(100 * float((findings.loc[part.index] == 1).mean()), 2),
            "2+ findings %": round(100 * float((findings.loc[part.index] >= 2).mean()), 2),
        }
        for group, part in _groups(frame).items()
    }
    return pd.DataFrame(rows).T


def images_per_patient(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for group, part in _groups(frame).items():
        counts = part.groupby("patient").size()
        rows[group] = {
            "patients": len(counts),
            "mean": round(float(counts.mean()), 2),
            "median": float(counts.median()),
            "p90": float(counts.quantile(0.9)),
            "max": int(counts.max()),
            "1 image %": round(100 * float((counts == 1).mean()), 2),
        }
    return pd.DataFrame(rows).T.astype({"patients": int, "max": int})


def age_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Age summary per split, excluding outliers, plus share of images per age band."""
    rows = {}
    for group, part in _groups(frame).items():
        ages = part.loc[part["age"] <= MAX_PLAUSIBLE_AGE, "age"]
        bands = pd.cut(ages, AGE_BINS, right=False, labels=AGE_BIN_NAMES)
        shares = (100 * bands.value_counts(normalize=True)).round(2)
        rows[group] = {
            "mean": round(float(ages.mean()), 1),
            "median": float(ages.median()),
            "q1": float(ages.quantile(0.25)),
            "q3": float(ages.quantile(0.75)),
            "min": int(ages.min()),
            "max": int(ages.max()),
            **{f"{name} %": float(shares[name]) for name in AGE_BIN_NAMES},
        }
    return pd.DataFrame(rows).T.astype({"min": int, "max": int})


def sex_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {
        group: {
            "male %": round(100 * float((part["sex"] == "M").mean()), 2),
            "female %": round(100 * float((part["sex"] == "F").mean()), 2),
            "male patients": part.loc[part["sex"] == "M", "patient"].nunique(),
            "female patients": part.loc[part["sex"] == "F", "patient"].nunique(),
        }
        for group, part in _groups(frame).items()
    }
    return pd.DataFrame(rows).T.astype({"male patients": int, "female patients": int})


def age_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Images with an age above MAX_PLAUSIBLE_AGE, sorted by age."""
    columns = ["image", "patient", "age", "split"]
    outliers = frame.loc[frame["age"] > MAX_PLAUSIBLE_AGE, columns]
    return outliers.sort_values(["age", "image"], ascending=[False, True]).set_index("image")


def markdown_table(table: pd.DataFrame) -> str:
    header = [str(table.index.name or ""), *map(str, table.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in table.iterrows():
        cells = [str(index), *(_format(value) for value in row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}" if value.is_integer() else f"{value:.2f}".rstrip("0")
    return str(value)


def split_difference_note(frame: pd.DataFrame) -> str:
    """One sentence comparing the test split with train; metrics on test inherit the gap."""
    share = no_finding_share(frame)
    per_patient = images_per_patient(frame)
    return (
        f"The official test split differs from train: {share.loc['test', 'No Finding %']:g}% "
        f"No Finding images against {share.loc['train', 'No Finding %']:g}%, and "
        f"{per_patient.loc['test', 'mean']:g} images per patient against "
        f"{per_patient.loc['train', 'mean']:g}. Results on test are not directly comparable with "
        "results on val."
    )


def render_report(frame: pd.DataFrame) -> str:
    outliers = age_outliers(frame)
    outlier_patients = outliers["patient"].nunique()
    sections = [
        "# NIH ChestX-ray14: EDA of the splits",
        "",
        "**For education only, not for clinical use.**",
        "",
        "Generated by `python -m src.data.eda` from `data/splits/*.csv`; do not edit by hand.",
        "",
        "**Labels are NLP-mined, not radiologist-verified.** The 14 findings and No Finding were "
        "extracted from radiology reports by text mining. Expect label noise, and treat every "
        "metric on these labels as agreement with the mined labels, not with a diagnosis.",
        "",
        "Splits: the official `test_list.txt` is the test split; val is about 10% of the "
        "`train_val_list.txt` patients (seed 42). No patient is in two splits. All tables count "
        "images unless they say patients.",
        "",
        "## Split sizes",
        "",
        markdown_table(split_sizes(frame)),
        "",
        split_difference_note(frame),
        "",
        "## Label prevalence per split",
        "",
        "Percent of images with each label. An image can have several findings.",
        "",
        markdown_table(label_prevalence(frame)),
        "",
        "## No Finding share",
        "",
        markdown_table(no_finding_share(frame)),
        "",
        "## Images per patient",
        "",
        markdown_table(images_per_patient(frame)),
        "",
        "## Age per split",
        "",
        f"Ages above {MAX_PLAUSIBLE_AGE} are left out here and listed under Age outliers.",
        "",
        markdown_table(age_distribution(frame)),
        "",
        "## Sex per split",
        "",
        markdown_table(sex_distribution(frame)),
        "",
        "## Age outliers",
        "",
        f"{len(outliers)} images from {outlier_patients} patients have an age above "
        f"{MAX_PLAUSIBLE_AGE}, which is almost certainly a recording error. They are kept in the "
        "splits as recorded.",
        "",
    ]
    if not outliers.empty:
        sections += [markdown_table(outliers), ""]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = load_splits(args.splits_dir)
    print(f"loaded {len(frame)} rows, {frame['patient'].nunique()} patients")
    # Only the NIH block is replaced; the spleen section (src/data/spleen.py) is kept.
    write_section(args.out, "nih", render_report(frame))
    print(f"wrote the NIH section of {args.out}")


if __name__ == "__main__":
    main()
