"""The notebook's flow on the fixture: zip -> manifest CSV -> EDA -> shards -> read back."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import nih, shards

FIXTURES = Path(__file__).parent / "fixtures" / "nih"
METADATA = ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")


def make_dataset_zip(path: Path) -> list[str]:
    names = nih.read_image_list(FIXTURES / "train_val_list.txt")
    names += nih.read_image_list(FIXTURES / "test_list.txt")
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as archive:
        for name in METADATA:
            archive.write(FIXTURES / name, name)
        for index, name in enumerate(names):
            pixels = rng.integers(0, 256, (64, 64), dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            archive.writestr(f"images_{index % 2 + 1:03d}/images/{name}", buffer.getvalue())
    return names


def test_notebook_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "data.zip"
    names = make_dataset_zip(zip_path)
    images = shards.ZipImages(zip_path)
    meta = tmp_path / "meta"
    data_entry, train_val, test = (images.extract_file(name, meta) for name in METADATA)

    manifest_path = tmp_path / "nih_manifest.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "nih",
            "--data-entry",
            str(data_entry),
            "--train-val-list",
            str(train_val),
            "--test-list",
            str(test),
            "--out",
            str(manifest_path),
            "--val-fraction",
            "0.2",
            "--expected-images",
            "13",
        ],
    )
    nih.main()

    manifest = pd.read_csv(manifest_path, dtype={"patient": str, "image": str})
    nih.assert_patient_level_splits(manifest)
    assert set(manifest["image"]) == images.names == set(names)

    tables = nih.write_eda_tables(manifest, tmp_path / "reports")
    assert all(path.stat().st_size > 0 for path in tables)

    out = tmp_path / "shards"
    report = shards.write_shards(manifest, images.read, out, tmp_path / "work", size=32)
    images.close()
    assert report.images == len(manifest) == 13

    stored: list[str] = []
    for path in sorted(out.glob("*.npz")):
        split = path.name.split("-")[0]
        pixels, shard_names = shards.read_shard(path)
        assert pixels.shape == (len(shard_names), 32, 32)
        rows = manifest[manifest["image"].isin(shard_names)]
        assert set(rows["split"]) == {split}
        stored += shard_names
    assert sorted(stored) == sorted(names)
