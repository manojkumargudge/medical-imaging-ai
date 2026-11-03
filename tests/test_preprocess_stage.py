import csv
import json
from pathlib import Path

import pytest
from conftest import PARAMS, Synthetic, build_synthetic, make_splits, write_dataset_zip

from src.data import preprocess, split

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_manifest_has_every_split_image_with_its_shard(synthetic: Synthetic) -> None:
    rows = read_rows(synthetic.manifest)
    expected = {entry.image: name for name in split.SPLITS for entry in synthetic.splits[name]}
    print(f"manifest rows {len(rows)}, split rows {len(expected)}")
    assert rows and expected
    assert list(rows[0]) == list(preprocess.MANIFEST_COLUMNS)
    assert {row["image"]: row["split"] for row in rows} == expected
    stored = {row["image"]: row for row in read_rows(synthetic.processed_dir / "images.csv")}
    assert all(stored[row["image"]]["shard"] == row["shard"] for row in rows)
    assert all(stored[row["image"]]["sha256"] == row["sha256"] for row in rows)


def test_manifest_is_patient_level(synthetic: Synthetic) -> None:
    rows = read_rows(synthetic.manifest)
    patients = {
        name: {row["patient"] for row in rows if row["split"] == name} for name in split.SPLITS
    }
    for name, members in patients.items():
        print(f"{name}: {len(members)} patients")
        assert members
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]


def test_preprocess_info_records_size_and_imagenet_stats(synthetic: Synthetic) -> None:
    info = json.loads((synthetic.processed_dir / preprocess.PREPROCESS_INFO).read_text())
    assert info["image_size"] == PARAMS["image_size"]
    assert info["mean"] == PARAMS["mean"] and info["std"] == PARAMS["std"]
    assert info["images"] == sum(len(rows) for rows in synthetic.splits.values())
    assert preprocess.verify_shards(synthetic.processed_dir).images == info["images"]


def test_resume_dir_gives_the_same_outputs(synthetic: Synthetic, tmp_path: Path) -> None:
    resume = tmp_path / "vm-resume"
    other = build_synthetic(tmp_path / "run", resume_dir=resume)
    assert preprocess.verify_shards(resume / "32px-5").images > 0
    assert other.manifest.read_text() == synthetic.manifest.read_text()
    for name in ("shards.csv", "images.csv", preprocess.PREPROCESS_INFO):
        assert (other.processed_dir / name).read_text() == (
            synthetic.processed_dir / name
        ).read_text()


def test_resume_dir_skips_finished_shards(tmp_path: Path) -> None:
    resume = tmp_path / "vm-resume"
    build_synthetic(tmp_path / "first", resume_dir=resume)
    before = {path.name: path.stat().st_mtime_ns for path in resume.glob("*/*.zip")}
    build_synthetic(tmp_path / "second", resume_dir=resume)
    after = {path.name: path.stat().st_mtime_ns for path in resume.glob("*/*.zip")}
    assert before and before == after


def test_split_image_missing_from_zip_fails(tmp_path: Path) -> None:
    splits = make_splits()
    split.write_splits(splits, tmp_path / "splits")
    images = [entry.image for name in split.SPLITS for entry in splits[name]]
    write_dataset_zip(tmp_path / "data.zip", images[1:])
    with pytest.raises(ValueError, match="not in the zip"):
        preprocess.run_stage(
            tmp_path / "data.zip",
            tmp_path / "splits",
            tmp_path / "out",
            tmp_path / "manifest.csv",
            tmp_path / "work",
            PARAMS,
        )


def test_zip_image_in_no_split_fails(tmp_path: Path) -> None:
    splits = make_splits()
    split.write_splits(splits, tmp_path / "splits")
    images = [entry.image for name in split.SPLITS for entry in splits[name]]
    write_dataset_zip(tmp_path / "data.zip", [*images, "99999999_000.png"])
    with pytest.raises(ValueError, match="in no split"):
        preprocess.run_stage(
            tmp_path / "data.zip",
            tmp_path / "splits",
            tmp_path / "out",
            tmp_path / "manifest.csv",
            tmp_path / "work",
            PARAMS,
        )


def test_different_existing_manifest_is_not_overwritten(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "nih.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("old\n")
    with pytest.raises(ValueError, match="not overwriting"):
        build_synthetic(tmp_path)
    assert manifest.read_text() == "old\n"


def test_copy_into_non_empty_folder_fails(synthetic: Synthetic, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        preprocess.copy_shards(synthetic.processed_dir, target)
    assert (target / "keep.txt").read_text() == "keep"


def test_repo_params_are_224_and_imagenet() -> None:
    params = preprocess.read_params(REPO_ROOT / "params.yaml")
    assert params["image_size"] == 224
    assert params["mean"] == [0.485, 0.456, 0.406]
    assert params["std"] == [0.229, 0.224, 0.225]


def test_params_reject_wrong_stats(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        "preprocess:\n  image_size: 224\n  shard_size: 2\n  mean: [0.5]\n  std: [1.0]\n"
    )
    with pytest.raises(ValueError, match="three floats"):
        preprocess.read_params(path)


@pytest.mark.parametrize(
    "path", ["/content/drive", "/content/drive/MyDrive/x", "/content/drive/../drive/MyDrive"]
)
def test_paths_on_drive_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="Google Drive"):
        preprocess.check_off_drive(Path(path), "out_dir")


@pytest.mark.parametrize("path", ["/content/work", "/content/drive2", "/content/medical/data"])
def test_vm_paths_are_accepted(path: str) -> None:
    preprocess.check_off_drive(Path(path), "out_dir")


@pytest.mark.parametrize("argument", ["out_dir", "manifest_path", "work_dir", "resume_dir"])
def test_stage_refuses_to_write_to_drive(tmp_path: Path, argument: str) -> None:
    paths = {
        "out_dir": tmp_path / "out",
        "manifest_path": tmp_path / "manifest.csv",
        "work_dir": tmp_path / "work",
        "resume_dir": tmp_path / "resume",
    }
    paths[argument] = Path("/content/drive/MyDrive/aiml02") / argument
    with pytest.raises(ValueError, match="Google Drive"):
        preprocess.run_stage(
            tmp_path / "data.zip",
            tmp_path / "splits",
            paths["out_dir"],
            paths["manifest_path"],
            paths["work_dir"],
            PARAMS,
            paths["resume_dir"],
        )
    assert not (tmp_path / "out").exists()


def test_resume_folder_is_keyed_by_params(tmp_path: Path) -> None:
    resume = tmp_path / "vm-resume"
    build_synthetic(tmp_path / "run", resume_dir=resume)
    assert [path.name for path in resume.iterdir()] == ["32px-5"]
