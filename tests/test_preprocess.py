import csv
import io
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from PIL import Image

from src.data import preprocess

NAMES = [f"img{i}.png" for i in range(5)]


def png_bytes(value: int, mode: str = "L", size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    color = value if mode == "L" else (value,) * len(mode)
    Image.new(mode, size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def reader(image: str) -> bytes:
    return png_bytes(sum(image.encode()) % 256)


def write(
    out: Path,
    work: Path,
    read: Callable[[str], bytes] = reader,
    names: Sequence[str] = tuple(NAMES),
) -> preprocess.Report:
    return preprocess.write_shards(names, read, out, work, size=16, shard_size=2)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_resize_gives_grayscale_png_at_target_size(mode: str) -> None:
    with Image.open(io.BytesIO(preprocess.resize_png(png_bytes(100, mode), size=32))) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        assert image.size == (32, 32)


def test_resize_rejects_unexpected_mode() -> None:
    with pytest.raises(ValueError, match="unexpected image mode"):
        preprocess.resize_png(png_bytes(1, mode="1"))


def test_plan_is_sorted_and_complete() -> None:
    plan = preprocess.plan_shards(NAMES[::-1], shard_size=2)
    assert list(plan) == ["shard-00000.zip", "shard-00001.zip", "shard-00002.zip"]
    assert [name for names in plan.values() for name in names] == sorted(NAMES)


@pytest.mark.parametrize("names", [[], ["a.png", "a.png"]])
def test_plan_rejects_empty_or_duplicate(names: list[str]) -> None:
    with pytest.raises(ValueError):
        preprocess.plan_shards(names)


def test_write_records_checksums_and_verifies(tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = write(out, tmp_path / "work")
    assert (report.shards, report.images, report.written, report.skipped) == (3, 5, 3, 0)
    assert report.total_bytes == sum(path.stat().st_size for path in out.glob("*.zip"))
    assert not list(out.glob("*.partial"))

    with (out / preprocess.IMAGES_MANIFEST).open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["image"] for row in rows] == sorted(NAMES)
    assert all(len(row["sha256"]) == 64 for row in rows)

    verified = preprocess.verify_shards(out)
    assert (verified.shards, verified.images) == (3, 5)


def test_rerun_skips_recorded_shards(tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = write(out, tmp_path / "work")

    def must_not_read(image: str) -> bytes:
        raise AssertionError(f"re-read {image}")

    second = write(out, tmp_path / "work", read=must_not_read)
    assert (second.written, second.skipped) == (0, 3)
    assert (second.images, second.total_bytes) == (first.images, first.total_bytes)


def test_resume_after_a_disconnect(tmp_path: Path) -> None:
    out = tmp_path / "out"
    calls: list[str] = []

    def flaky(image: str) -> bytes:
        calls.append(image)
        if image == "img3.png" and calls.count(image) == 1:
            raise ConnectionError("runtime disconnected")
        return reader(image)

    with pytest.raises(ConnectionError):
        write(out, tmp_path / "work", read=flaky)
    with (out / preprocess.SHARDS_MANIFEST).open(newline="") as file:
        assert [row["shard"] for row in csv.DictReader(file)] == ["shard-00000.zip"]

    report = write(out, tmp_path / "work", read=flaky)
    assert (report.written, report.skipped, report.images) == (2, 1, 5)
    assert preprocess.verify_shards(out).images == 5


def test_identical_unrecorded_shard_is_adopted(tmp_path: Path) -> None:
    # Simulates a crash after the shard reached Drive but before the manifest was written.
    out = tmp_path / "out"
    write(out, tmp_path / "work")
    shards_manifest = out / preprocess.SHARDS_MANIFEST
    lines = shards_manifest.read_text().splitlines()
    shards_manifest.write_text("\n".join(lines[:-1]) + "\n")
    images_manifest = out / preprocess.IMAGES_MANIFEST
    image_lines = images_manifest.read_text().splitlines()
    images_manifest.write_text("\n".join(image_lines[:-1]) + "\n")

    report = write(out, tmp_path / "work")
    assert (report.written, report.images) == (1, 5)
    assert preprocess.verify_shards(out).images == 5


def test_different_unrecorded_shard_is_not_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "shard-00000.zip").write_bytes(b"something else")
    with pytest.raises(ValueError, match="not overwriting"):
        write(out, tmp_path / "work")
    assert (out / "shard-00000.zip").read_bytes() == b"something else"


def test_corrupted_shard_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write(out, tmp_path / "work")
    with (out / "shard-00001.zip").open("ab") as file:
        file.write(b"x")
    with pytest.raises(ValueError, match="does not match its manifest entry"):
        write(out, tmp_path / "work")
    with pytest.raises(ValueError, match="sha256 differs"):
        preprocess.verify_shards(out)


def test_rerun_with_other_images_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write(out, tmp_path / "work")
    with pytest.raises(ValueError, match="another shard than the plan"):
        write(out, tmp_path / "work", names=["a.png", *NAMES])


def test_shards_are_deterministic(tmp_path: Path) -> None:
    write(tmp_path / "a", tmp_path / "work")
    write(tmp_path / "b", tmp_path / "work")
    first = (tmp_path / "a" / preprocess.SHARDS_MANIFEST).read_text()
    assert first == (tmp_path / "b" / preprocess.SHARDS_MANIFEST).read_text()


def test_zip_images_reads_nested_members(tmp_path: Path) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images_001/images/b.png", png_bytes(2))
        archive.writestr("images_002/images/a.png", png_bytes(1))
        archive.writestr("Data_Entry_2017.csv", "x\n")
    images = preprocess.ZipImages(path)
    assert images.names == ["a.png", "b.png"]
    assert images.read("b.png") == png_bytes(2)
    images.close()


def test_zip_images_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images_001/images/a.png", png_bytes(1))
        archive.writestr("images_002/images/a.png", png_bytes(2))
    with pytest.raises(ValueError, match="appears twice"):
        preprocess.ZipImages(path)
