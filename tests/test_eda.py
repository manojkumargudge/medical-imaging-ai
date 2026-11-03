from pathlib import Path

import pandas as pd
import pytest

from src.data import eda, split

FIXTURES = Path(__file__).parent / "fixtures" / "nih"


@pytest.fixture
def splits_dir(tmp_path: Path) -> Path:
    splits = split.build_splits(
        FIXTURES / "Data_Entry_2017.csv",
        FIXTURES / "train_val_list.txt",
        FIXTURES / "test_list.txt",
        val_fraction=0.2,
        expected_images=13,
    )
    split.write_splits(splits, tmp_path)
    return tmp_path


@pytest.fixture
def frame(splits_dir: Path) -> pd.DataFrame:
    return eda.load_splits(splits_dir)


def test_labels_are_14_findings_plus_no_finding() -> None:
    assert len(eda.FINDINGS) == 14
    assert eda.LABELS[-1] == "No Finding"


def test_multi_hot_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        eda.multi_hot(pd.Series(["Mass|Unknown"]))


def test_load_fails_on_empty_split(splits_dir: Path) -> None:
    (splits_dir / "val.csv").write_text(",".join(split.COLUMNS) + "\n")
    with pytest.raises(ValueError, match="val.csv has no rows"):
        eda.load_splits(splits_dir)


def test_split_sizes(frame: pd.DataFrame) -> None:
    sizes = eda.split_sizes(frame)
    assert sizes.loc["all", "images"] == 13
    assert sizes.loc["all", "patients"] == 11
    assert sizes.loc["test", "images"] == 3


def test_label_prevalence(frame: pd.DataFrame) -> None:
    prevalence = eda.label_prevalence(frame)
    assert prevalence.loc["No Finding", "all count"] == 3
    assert prevalence.loc["Fibrosis", "all count"] == 2
    assert prevalence.loc["Fibrosis", "test %"] == round(100 * 2 / 3, 2)
    assert prevalence.loc["Fibrosis", "train %"] == 0


def test_no_finding_share(frame: pd.DataFrame) -> None:
    share = eda.no_finding_share(frame)
    assert share.loc["all", "No Finding %"] == round(100 * 3 / 13, 2)
    # 00000001_001, 00000004_000, 00000007_000, 00000009_000, 00000010_001.
    assert share.loc["all", "2+ findings %"] == round(100 * 5 / 13, 2)


def test_images_per_patient(frame: pd.DataFrame) -> None:
    per_patient = eda.images_per_patient(frame)
    assert per_patient.loc["all", "patients"] == 11
    assert per_patient.loc["all", "max"] == 2
    assert per_patient.loc["all", "1 image %"] == round(100 * 9 / 11, 2)


def test_age_distribution_leaves_out_outliers(frame: pd.DataFrame) -> None:
    ages = eda.age_distribution(frame)
    assert ages.loc["all", "max"] == 82
    assert ages.loc["all", "min"] == 19


def test_sex_distribution(frame: pd.DataFrame) -> None:
    sexes = eda.sex_distribution(frame)
    assert sexes.loc["all", "male %"] + sexes.loc["all", "female %"] == pytest.approx(100)
    assert sexes.loc["all", "male patients"] + sexes.loc["all", "female patients"] == 11


def test_age_outliers(frame: pd.DataFrame) -> None:
    outliers = eda.age_outliers(frame)
    assert list(outliers.index) == ["00000005_000.png"]
    assert outliers.loc["00000005_000.png", "age"] == 412


def test_report_has_every_section_and_the_label_note(frame: pd.DataFrame) -> None:
    report = eda.render_report(frame)
    assert "**Labels are NLP-mined, not radiologist-verified.**" in report
    assert "For education only, not for clinical use." in report
    for heading in [
        "## Split sizes",
        "## Label prevalence per split",
        "## No Finding share",
        "## Images per patient",
        "## Age per split",
        "## Sex per split",
        "## Age outliers",
    ]:
        assert heading in report
    assert "1 images from 1 patients have an age above 100" in report
    assert "The official test split differs from train" in report
    assert "| 00000005_000.png | 5 | 412 |" in report


def test_cli_writes_report(
    splits_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "docs" / "eda.md"
    monkeypatch.setattr("sys.argv", ["eda", "--splits-dir", str(splits_dir), "--out", str(out)])
    eda.main()
    assert out.read_text(encoding="utf-8").startswith("# NIH ChestX-ray14")
