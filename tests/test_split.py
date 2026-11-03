from pathlib import Path

import pytest

from src.data import split

FIXTURES = Path(__file__).parent / "fixtures" / "nih"
DATA_ENTRY = FIXTURES / "Data_Entry_2017.csv"
TRAIN_VAL = FIXTURES / "train_val_list.txt"
TEST = FIXTURES / "test_list.txt"


def build(seed: int = 42) -> dict[str, list[split.Entry]]:
    return split.build_splits(
        DATA_ENTRY, TRAIN_VAL, TEST, val_fraction=0.2, seed=seed, expected_images=13
    )


def patients(entries: list[split.Entry]) -> set[str]:
    return {entry.patient for entry in entries}


def test_no_patient_in_two_splits() -> None:
    splits = build()
    assert not patients(splits["train"]) & patients(splits["val"])
    assert not patients(splits["train"]) & patients(splits["test"])
    assert not patients(splits["val"]) & patients(splits["test"])


def test_every_split_is_non_empty() -> None:
    splits = build()
    assert all(splits[name] for name in split.SPLITS)


def test_counts_are_printed(capsys: pytest.CaptureFixture[str]) -> None:
    build()
    out = capsys.readouterr().out
    assert "test: 3 rows, 2 patients" in out
    assert "total: 13 rows, 11 patients" in out
    assert "patient overlap train/val: 0" in out


def test_official_test_list_is_the_test_split() -> None:
    splits = build()
    assert [entry.image for entry in splits["test"]] == sorted(split.read_image_list(TEST))


def test_val_is_about_the_requested_share_of_train_val_patients() -> None:
    splits = build()
    # 9 train_val patients at 20% round to 2.
    assert len(patients(splits["val"])) == 2
    assert len(patients(splits["train"])) == 7


def test_split_is_reproducible_and_seed_dependent() -> None:
    first = build(seed=42)
    assert build(seed=42) == first
    assert any(build(seed=seed)["val"] != first["val"] for seed in range(1, 10))


def test_split_does_not_depend_on_row_order() -> None:
    entries = split.load_entries(DATA_ENTRY)
    train_val = split.read_image_list(TRAIN_VAL)
    test = split.read_image_list(TEST)
    forward = split.split_by_patient(entries, train_val, test, val_fraction=0.2)
    backward = split.split_by_patient(entries[::-1], train_val[::-1], test, val_fraction=0.2)
    assert forward == backward


def test_age_is_kept_as_recorded() -> None:
    splits = build()
    ages = {entry.image: entry.age for name in split.SPLITS for entry in splits[name]}
    assert ages["00000005_000.png"] == 412


def test_overlap_check_fails_on_a_shared_patient() -> None:
    splits = build()
    leaked = {name: list(entries) for name, entries in splits.items()}
    leaked["test"].append(leaked["train"][0])
    with pytest.raises(ValueError, match="patients are in both train and test"):
        split.check_patient_level(leaked)


def test_official_lists_sharing_a_patient_fail(tmp_path: Path) -> None:
    # Patient 10 then has one image in train_val and one in test.
    train_val = tmp_path / "train_val.txt"
    train_val.write_text(TRAIN_VAL.read_text() + "00000010_000.png\n")
    test = tmp_path / "test.txt"
    test.write_text("00000010_001.png\n00000011_000.png\n")
    with pytest.raises(ValueError, match="patients are in both"):
        split.build_splits(DATA_ENTRY, train_val, test, val_fraction=0.2, expected_images=13)


def test_empty_split_fails() -> None:
    splits = build()
    with pytest.raises(ValueError, match="empty splits"):
        split.check_patient_level({**splits, "val": []})


def test_wrong_image_count_fails() -> None:
    with pytest.raises(ValueError, match="expected 112120"):
        split.build_splits(DATA_ENTRY, TRAIN_VAL, TEST)


@pytest.mark.parametrize(
    ("extra", "message"),
    [("00000001_000.png\n", "both official lists"), ("", "do not match the CSV")],
)
def test_bad_official_lists_fail(tmp_path: Path, extra: str, message: str) -> None:
    test = tmp_path / "test.txt"
    # Without the extra line this drops two test images, so the lists no longer cover the CSV.
    test.write_text(("00000011_000.png\n" if not extra else TEST.read_text()) + extra)
    with pytest.raises(ValueError, match=message):
        split.build_splits(DATA_ENTRY, TRAIN_VAL, test, expected_images=13)


def test_bad_age_fails(tmp_path: Path) -> None:
    data_entry = tmp_path / "Data_Entry_2017.csv"
    data_entry.write_text(DATA_ENTRY.read_text().replace(",81,F,", ",81.5,F,"))
    with pytest.raises(ValueError, match="age is not a whole number"):
        split.load_entries(data_entry)


def test_cli_writes_csvs_that_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "splits"
    monkeypatch.setattr(
        "sys.argv",
        [
            "split",
            "--data-entry",
            str(DATA_ENTRY),
            "--train-val-list",
            str(TRAIN_VAL),
            "--test-list",
            str(TEST),
            "--out-dir",
            str(out_dir),
            "--val-fraction",
            "0.2",
            "--expected-images",
            "13",
        ],
    )
    split.main()
    assert sorted(path.name for path in out_dir.iterdir()) == ["test.csv", "train.csv", "val.csv"]
    assert (out_dir / "train.csv").read_text().splitlines()[0] == ",".join(split.COLUMNS)
    assert split.read_splits(out_dir) == build()
    assert "wrote" in capsys.readouterr().out
