import json
from pathlib import Path

import pytest
from conftest import make_spleen_task

from src.data import report, spleen

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = REPO_ROOT / "data" / "seg_splits" / "spleen.csv"


def write_dataset_json(path: Path, cases: int) -> Path:
    training = [
        {"image": f"./imagesTr/spleen_{n}.nii.gz", "label": f"./labelsTr/spleen_{n}.nii.gz"}
        for n in range(1, cases + 1)
    ]
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset.json").write_text(json.dumps({"training": training}))
    return path


def test_41_cases_give_29_6_6(tmp_path: Path) -> None:
    splits = spleen.build_split(write_dataset_json(tmp_path, 41), 0.15, 0.15, 42)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 29, "val": 6, "test": 6}


def test_split_has_no_overlap_and_covers_every_case(tmp_path: Path) -> None:
    cases = spleen.load_cases(write_dataset_json(tmp_path, 41))
    splits = spleen.split_cases(cases, 0.15, 0.15, 42)
    names = {name: {row.case for row in rows} for name, rows in splits.items()}
    for name, members in names.items():
        print(f"{name}: {len(members)} cases")
        assert members
    assert not names["train"] & names["val"]
    assert not names["train"] & names["test"]
    assert not names["val"] & names["test"]
    assert set().union(*names.values()) == {case.case for case in cases}


def test_split_is_seeded(tmp_path: Path) -> None:
    cases = spleen.load_cases(write_dataset_json(tmp_path, 41))
    first = spleen.split_cases(cases, 0.15, 0.15, 42)
    assert first == spleen.split_cases(list(reversed(cases)), 0.15, 0.15, 42)
    assert first != spleen.split_cases(cases, 0.15, 0.15, 7)


def test_overlap_is_rejected(tmp_path: Path) -> None:
    splits = spleen.split_cases(spleen.load_cases(write_dataset_json(tmp_path, 41)), 0.15, 0.15, 42)
    splits["test"].append(splits["train"][0])
    with pytest.raises(ValueError, match="in both train and test"):
        spleen.check_no_overlap(splits)


def test_wrong_case_count_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 41"):
        spleen.load_cases(write_dataset_json(tmp_path, 40))


def test_split_csv_round_trip(tmp_path: Path) -> None:
    splits = spleen.build_split(write_dataset_json(tmp_path, 41), 0.15, 0.15, 42)
    spleen.write_split(splits, tmp_path / "split.csv")
    assert spleen.read_split(tmp_path / "split.csv") == splits


def test_volume_stats_and_eda(tmp_path: Path) -> None:
    task = make_spleen_task(tmp_path, cases=10)
    splits = spleen.split_cases(spleen.load_cases(task, expected=10), 0.2, 0.2, 42)
    rows = [row for name in ("train", "val", "test") for row in splits[name]]
    stats = [spleen.volume_stats(task, row) for row in rows]
    assert stats[0]["shape"] == [24, 20, 16]
    assert stats[0]["spacing"] == pytest.approx([0.8, 0.8, 2.5])
    assert stats[0]["orientation"] == "LPS"
    assert all(row["spleen_ml"] > 0 for row in stats)
    text = spleen.render_eda(stats)
    assert "| volumes | 6 | 2 | 2 | 10 |" in text
    assert "spacing z (mm) | 2.5 / 2.5 / 2.5" in text


@pytest.mark.real_data
@pytest.mark.skipif(not SPLIT_CSV.exists(), reason="run `dvc repro seg_split`")
def test_real_spleen_split_has_no_overlap() -> None:
    splits = spleen.read_split(SPLIT_CSV)  # prints counts, fails on overlap
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 29, "val": 6, "test": 6}


def test_sections_are_replaced_independently(tmp_path: Path) -> None:
    path = tmp_path / "eda.md"
    report.write_section(path, "nih", "# NIH\n\nold\n")
    report.write_section(path, "spleen", "# Spleen\n")
    report.write_section(path, "nih", "# NIH\n\nnew\n")
    text = path.read_text()
    assert text.index("# NIH") < text.index("# Spleen")
    assert "old" not in text and "new" in text
    assert report.read_section(path, "spleen") == "# Spleen\n"


def test_repo_eda_has_both_sections() -> None:
    path = REPO_ROOT / "docs" / "eda.md"
    nih = report.read_section(path, "nih")
    assert nih is not None and nih.startswith("# NIH ChestX-ray14")
    spleen_text = report.read_section(path, "spleen")
    assert spleen_text is not None and spleen_text.startswith("# MSD Task09_Spleen")


def test_summary_keeps_whole_numbers() -> None:
    # A real-data run once printed 500 ml as "5": zeros were stripped from whole numbers.
    assert spleen._summary([500.0, 280.0, 1000.0], digits=0) == "280 / 500 / 1000"
    assert spleen._summary([0.5, 1.25, 2.0]) == "0.5 / 1.25 / 2"
