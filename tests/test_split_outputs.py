"""Checks on the real split CSVs from `dvc repro split`; skipped when they are absent (CI)."""

from pathlib import Path

import pytest

from src.data import split

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not all((SPLITS_DIR / f"{name}.csv").exists() for name in split.SPLITS),
        reason="data/splits/*.csv not built; run `dvc repro split`",
    ),
]


def test_real_splits_are_patient_level(capsys: pytest.CaptureFixture[str]) -> None:
    splits = split.read_splits(SPLITS_DIR)
    split.check_patient_level(splits)
    with capsys.disabled():
        for name, (rows, patients) in split.split_counts(splits).items():
            print(f"\n{name}: {rows} rows, {patients} patients", end="")
        print()
    assert sum(len(splits[name]) for name in split.SPLITS) == split.EXPECTED_IMAGES
