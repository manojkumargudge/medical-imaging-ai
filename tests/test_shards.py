import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import shards


def png_bytes(value: int, mode: str = "L", size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=value if mode == "L" else (value,) * len(mode)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def make_manifest(train: int = 5, test: int = 2) -> pd.DataFrame:
    rows = [{"image": f"tr{i}.png", "split": "train"} for i in range(train)]
    rows += [{"image": f"te{i}.png", "split": "test"} for i in range(test)]
    return pd.DataFrame(rows)


def reader(image: str) -> bytes:
    return png_bytes(sum(image.encode()) % 256)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_preprocess_gives_grayscale_uint8_at_target_size(mode: str) -> None:
    result = shards.preprocess_png(png_bytes(100, mode=mode), size=32)
    assert result.shape == (32, 32)
    assert result.dtype == np.uint8


def test_preprocess_rejects_unexpected_mode() -> None:
    with pytest.raises(ValueError, match="unexpected image mode"):
        shards.preprocess_png(png_bytes(1, mode="1"))


def test_delta_round_trip_is_lossless_including_wraparound() -> None:
    images = np.random.default_rng(0).integers(0, 256, (3, 16, 16), dtype=np.uint8)
    images[0, 0, :2] = [255, 0]
    assert np.array_equal(shards.delta_decode(shards.delta_encode(images)), images)


def test_plan_keeps_splits_apart_and_covers_every_image() -> None:
    manifest = make_manifest(train=5, test=2)
    plan = shards.plan_shards(manifest, shard_size=2)
    assert list(plan) == ["test-0000.npz", "train-0000.npz", "train-0001.npz", "train-0002.npz"]
    assert all(name.startswith("te") for name in plan["test-0000.npz"])
    planned = [image for images in plan.values() for image in images]
    assert sorted(planned) == sorted(manifest["image"])


def test_plan_rejects_empty_manifest() -> None:
    with pytest.raises(ValueError, match="no rows"):
        shards.plan_shards(make_manifest(train=0, test=0))


def test_write_and_read_back(tmp_path: Path) -> None:
    manifest = make_manifest()
    out = tmp_path / "out"
    report = shards.write_shards(manifest, reader, out, tmp_path / "work", size=16, shard_size=2)
    assert report.images == len(manifest) == 7
    assert report.shards == 4
    assert report.total_bytes == sum(path.stat().st_size for path in out.glob("*.npz"))
    assert not list(out.glob("*.partial"))

    stored: dict[str, np.ndarray] = {}
    for path in out.glob("*.npz"):
        images, names = shards.read_shard(path)
        assert images.shape[1:] == (16, 16)
        stored.update(zip(names, images, strict=True))
    assert set(stored) == set(manifest["image"])
    for name, image in stored.items():
        assert np.array_equal(image, shards.preprocess_png(reader(name), size=16))


def test_rerun_keeps_finished_shards(tmp_path: Path) -> None:
    manifest = make_manifest()
    out = tmp_path / "out"
    first = shards.write_shards(manifest, reader, out, tmp_path / "work", size=16, shard_size=2)

    def must_not_read(image: str) -> bytes:
        raise AssertionError(f"re-read {image}")

    second = shards.write_shards(
        manifest, must_not_read, out, tmp_path / "work", size=16, shard_size=2
    )
    assert second == first


def test_rerun_with_other_manifest_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    shards.write_shards(make_manifest(), reader, out, tmp_path / "work", size=16, shard_size=2)
    other = make_manifest()
    other.loc[0, "image"] = "other.png"
    with pytest.raises(ValueError, match="different images"):
        shards.write_shards(other, reader, out, tmp_path / "work", size=16, shard_size=2)


def test_size_limit_stops_after_first_shard(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(shards.ShardSizeLimitError, match="reaches the"):
        shards.write_shards(
            make_manifest(),
            reader,
            out,
            tmp_path / "work",
            size=16,
            shard_size=2,
            max_total_bytes=10,
        )
    assert len(list(out.glob("*.npz"))) == 1


def test_delta_shrinks_a_smooth_image(tmp_path: Path) -> None:
    ramp = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
    np.savez_compressed(tmp_path / "raw.npz", images=ramp)
    np.savez_compressed(tmp_path / "delta.npz", images=shards.delta_encode(ramp))
    assert (tmp_path / "delta.npz").stat().st_size < (tmp_path / "raw.npz").stat().st_size


def test_zip_images_reads_nested_members(tmp_path: Path) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images_001/images/a.png", png_bytes(1))
        archive.writestr("images_002/images/b.png", png_bytes(2))
        archive.writestr("Data_Entry_2017.csv", "x\n1\n")
    images = shards.ZipImages(path)
    assert images.names == {"a.png", "b.png"}
    assert images.read("b.png") == png_bytes(2)
    extracted = images.extract_file("Data_Entry_2017.csv", tmp_path / "meta")
    assert extracted.read_text() == "x\n1\n"
    with pytest.raises(ValueError, match="expected one"):
        images.extract_file("missing.txt", tmp_path / "meta")
    images.close()


def test_zip_images_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images_001/images/a.png", png_bytes(1))
        archive.writestr("images_002/images/a.png", png_bytes(2))
    with pytest.raises(ValueError, match="appears twice"):
        shards.ZipImages(path)
