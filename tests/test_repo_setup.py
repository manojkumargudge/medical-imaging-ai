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


def test_dvc_sees_the_manifests_pointer() -> None:
    with Repo(str(REPO_ROOT)) as repo:
        stages = [Path(stage.addressing).as_posix() for stage in repo.index.stages]
    assert "data/manifests.dvc" in stages


@pytest.mark.parametrize(
    "path",
    [
        "data/manifests/nih_manifest.csv",
        "data/raw/image.png",
        "data/cache/shard.npz",
        "data/loose.csv",
        "data/other/image.png",
        "kaggle.json",
        "shard.npz",
        "checkpoints/model.bin",
    ],
)
def test_data_and_credentials_are_git_ignored(path: str) -> None:
    assert is_git_ignored(path)


@pytest.mark.parametrize("path", ["data/README.md", "data/manifests.dvc"])
def test_data_readme_and_pointer_stay_tracked(path: str) -> None:
    assert not is_git_ignored(path)
