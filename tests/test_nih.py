from pathlib import Path

import pandas as pd
import pytest

from src.data import nih

FIXTURES = Path(__file__).parent / "fixtures" / "nih"
DATA_ENTRY = FIXTURES / "Data_Entry_2017.csv"
TRAIN_VAL = FIXTURES / "train_val_list.txt"
TEST = FIXTURES / "test_list.txt"


def build(**kwargs: object) -> pd.DataFrame:
    return nih.build_manifest(
        DATA_ENTRY,
        TRAIN_VAL,
        TEST,
        val_fraction=0.2,
        expected_images=13,
        **kwargs,  # type: ignore[arg-type]
    )


def test_label_columns_are_14_findings_plus_no_finding() -> None:
    assert len(nih.FINDINGS) == 14
    assert nih.LABEL_COLUMNS[-1] == "No Finding"


def test_parse_labels_multi_hot() -> None:
    parsed = nih.parse_labels(pd.Series(["Mass|Nodule", "No Finding"]))
    assert list(parsed.columns) == list(nih.LABEL_COLUMNS)
    assert parsed.iloc[0].sum() == 2
    assert parsed.loc[0, "Mass"] == 1
    assert parsed.loc[0, "Nodule"] == 1
    assert parsed.iloc[1].sum() == 1
    assert parsed.loc[1, "No Finding"] == 1


@pytest.mark.parametrize("raw", ["Mass|Unknown", "", "No Finding|Mass"])
def test_parse_labels_rejects_bad_values(raw: str) -> None:
    with pytest.raises(ValueError):
        nih.parse_labels(pd.Series([raw]))


@pytest.mark.parametrize(
    ("raw", "expected"), [("58", 58), ("081Y", 81), ("412", None), ("5M", None), ("", None)]
)
def test_parse_age(raw: str, expected: int | None) -> None:
    assert nih.parse_age(raw) == expected


def test_manifest_columns_and_values() -> None:
    manifest = build()
    assert list(manifest.columns) == list(nih.MANIFEST_COLUMNS)
    assert len(manifest) == 13
    assert manifest["image"].is_monotonic_increasing
    row = manifest.set_index("image").loc["00000001_001.png"]
    assert row["patient"] == "1"
    assert row["labels"] == "Cardiomegaly|Emphysema"
    assert row["sex"] == "M"
    assert row["age"] == 58
    assert row["Cardiomegaly"] == 1
    assert row["Emphysema"] == 1
    assert row["No Finding"] == 0


def test_implausible_age_becomes_missing() -> None:
    manifest = build().set_index("image")
    assert pd.isna(manifest.loc["00000005_000.png", "age"])
    assert manifest.loc["00000003_000.png", "age"] == 81


def test_official_test_list_is_kept_as_test() -> None:
    manifest = build()
    test_images = set(manifest.loc[manifest["split"] == "test", "image"])
    assert test_images == set(nih.read_image_list(TEST))


def test_val_is_carved_by_patient_from_train_val() -> None:
    manifest = build()
    # 9 train_val patients at 20% round to 2.
    assert manifest.loc[manifest["split"] == "val", "patient"].nunique() == 2
    per_patient = manifest.groupby("patient")["split"].nunique()
    assert per_patient.max() == 1


def test_split_is_reproducible_and_seed_dependent() -> None:
    first = build(seed=42)["split"].tolist()
    assert first == build(seed=42)["split"].tolist()
    assert any(build(seed=seed)["split"].tolist() != first for seed in range(1, 10))


def test_split_does_not_depend_on_row_order() -> None:
    entries = nih.load_data_entry(DATA_ENTRY)
    train_val = nih.read_image_list(TRAIN_VAL)
    test = nih.read_image_list(TEST)
    forward = nih.assign_splits(entries, train_val, test, val_fraction=0.2)
    shuffled = entries.iloc[::-1]
    backward = nih.assign_splits(shuffled, train_val, test, val_fraction=0.2)
    assert forward.sort_index().tolist() == backward.sort_index().tolist()


def test_counts_are_printed(capsys: pytest.CaptureFixture[str]) -> None:
    build()
    out = capsys.readouterr().out
    assert "total rows 13, total patients 11" in out
    assert "patient overlap train/test: 0" in out


def test_patient_overlap_fails() -> None:
    manifest = build()
    leaked = manifest.copy()
    # Patient 1 has two images; moving one of them puts the patient in two splits.
    leaked.loc[leaked["image"] == "00000001_000.png", "split"] = "train"
    leaked.loc[leaked["image"] == "00000001_001.png", "split"] = "test"
    with pytest.raises(ValueError, match="patients are in both"):
        nih.assert_patient_level_splits(leaked)


def test_official_lists_sharing_a_patient_fail(tmp_path: Path) -> None:
    # 00000010_000 stays in train_val while 00000010_001 is in test.
    train_val = tmp_path / "train_val.txt"
    train_val.write_text(TRAIN_VAL.read_text() + "00000010_000.png\n")
    test = tmp_path / "test.txt"
    test.write_text("00000010_001.png\n00000011_000.png\n")
    with pytest.raises(ValueError, match="patients are in both"):
        nih.build_manifest(
            DATA_ENTRY, train_val, test, val_fraction=0.2, expected_images=13, seed=42
        )


def test_empty_split_fails() -> None:
    manifest = build()
    with pytest.raises(ValueError, match="empty splits"):
        nih.assert_patient_level_splits(manifest[manifest["split"] != "val"])


def test_wrong_image_count_fails() -> None:
    with pytest.raises(ValueError, match="expected 112120"):
        nih.build_manifest(DATA_ENTRY, TRAIN_VAL, TEST)


def test_list_not_matching_csv_fails(tmp_path: Path) -> None:
    short = tmp_path / "test.txt"
    short.write_text("00000011_000.png\n")
    with pytest.raises(ValueError, match="do not match the CSV"):
        nih.build_manifest(DATA_ENTRY, TRAIN_VAL, short, expected_images=13)


def test_image_in_both_lists_fails(tmp_path: Path) -> None:
    both = tmp_path / "test.txt"
    both.write_text(TEST.read_text() + "00000001_000.png\n")
    with pytest.raises(ValueError, match="both official lists"):
        nih.build_manifest(DATA_ENTRY, TRAIN_VAL, both, expected_images=13)


def test_eda_tables(tmp_path: Path) -> None:
    manifest = build()
    paths = nih.write_eda_tables(manifest, tmp_path)
    assert sorted(path.name for path in paths) == [
        "age_sex_by_split.csv",
        "images_per_patient.csv",
        "label_prevalence.csv",
    ]

    prevalence = nih.label_prevalence(manifest)
    assert prevalence.loc["No Finding", "all_count"] == 3
    assert prevalence.loc["Fibrosis", "all_count"] == 2
    assert prevalence.loc["No Finding", "all_percent"] == round(100 * 3 / 13, 2)

    demographics = nih.age_sex_by_split(manifest)
    assert demographics.loc["all", "images"] == 13
    assert demographics.loc["all", "age_missing"] == 1
    assert demographics.loc["all", "age_max"] == 82

    per_patient = nih.images_per_patient(manifest)
    assert per_patient.loc["all", "patients"] == 11
    assert per_patient.loc["all", "max"] == 2


def test_cli_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "manifests" / "nih_manifest.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "nih",
            "--data-entry",
            str(DATA_ENTRY),
            "--train-val-list",
            str(TRAIN_VAL),
            "--test-list",
            str(TEST),
            "--out",
            str(out),
            "--val-fraction",
            "0.2",
            "--expected-images",
            "13",
        ],
    )
    nih.main()
    written = pd.read_csv(out, dtype={"patient": str})
    assert list(written.columns) == list(nih.MANIFEST_COLUMNS)
    assert len(written) == 13
    assert pd.isna(written.set_index("image").loc["00000005_000.png", "age"])
    assert written["split"].value_counts().to_dict().keys() == {"train", "val", "test"}
