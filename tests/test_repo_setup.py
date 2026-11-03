import subprocess
from pathlib import Path

import pytest
from dvc.repo import Repo

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_git_ignored(path: str) -> bool:
    result = subprocess.run(["git", "check-ignore", "--quiet", path], cwd=REPO_ROOT, check=False)
    return result.returncode == 0


def test_readme_starts_with_education_only_notice() -> None:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    assert "For education only, not for clinical use" in lines[0]


def test_dvc_sees_the_split_and_preprocess_stages() -> None:
    # Catches .gitignore patterns that hide dvc.yaml or data pointers from DVC.
    with Repo(str(REPO_ROOT)) as repo:
        stages = {stage.addressing: stage for stage in repo.index.stages}
    assert {"split", "preprocess"} <= set(stages)
    preprocess = stages["preprocess"]
    deps = {Path(str(dep.def_path)).as_posix() for dep in preprocess.deps}
    outs = {Path(str(out.def_path)).as_posix() for out in preprocess.outs}
    assert {"data/splits", "src/data/preprocess.py"} <= deps
    assert outs == {"data/processed/nih_224", "data/manifests/nih_224.csv"}


def test_default_dvc_remote_is_the_drive_folder() -> None:
    with Repo(str(REPO_ROOT)) as repo:
        config = repo.config.read("repo")
    remote = config["core"]["remote"]
    assert config["remote"][remote]["url"] == "/content/drive/MyDrive/aiml02-dvc"


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/Data_Entry_2017.csv",
        "data/raw/Data_Entry_2017.csv.zip",
        "data/splits/train.csv",
        "data/cache/shard.npz",
        "data/loose.csv",
        "data/other/image.png",
        "kaggle.json",
        "checkpoints/model.bin",
        "data/processed/nih_224/shard-00000.zip",
        "data/manifests/nih_224.csv",
        ".dvc/config.local",
    ],
)
def test_data_and_credentials_are_git_ignored(path: str) -> None:
    assert is_git_ignored(path)


@pytest.mark.parametrize(
    "path",
    [
        "data/README.md",
        "dvc.yaml",
        "dvc.lock",
        "params.yaml",
        "configs/classifier.yaml",
        "reports/classifier_metrics.json",
    ],
)
def test_pipeline_files_stay_tracked(path: str) -> None:
    assert not is_git_ignored(path)
