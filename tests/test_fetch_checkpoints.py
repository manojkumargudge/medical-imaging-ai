"""scripts/fetch_checkpoints.py and the worker's behaviour without checkpoints. No network:
downloads are faked. For education only, not for clinical use."""

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_checkpoints as fc
from src.service import worker as worker_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def item(data: bytes, name: str = "model.pt") -> fc.Checkpoint:
    return fc.Checkpoint("X_CHECKPOINT", name, hashlib.sha256(data).hexdigest(), len(data))


def serving(data: bytes) -> tuple[list[str], fc.Download]:
    urls: list[str] = []

    def get(url: str, path: Path) -> None:
        urls.append(url)
        path.write_bytes(data)

    return urls, get


def test_download_is_checked_then_renamed(tmp_path: Path) -> None:
    urls, get = serving(b"weights")
    target = tmp_path / "ckpt" / "model.pt"
    assert fc.fetch(item(b"weights"), target, get) == "downloaded"
    assert target.read_bytes() == b"weights" and not (tmp_path / "ckpt" / "model.pt.part").exists()
    assert urls == [fc.RELEASE + "model.pt"]
    assert fc.fetch(item(b"weights"), target, get) == "kept" and len(urls) == 1  # no re-download


def test_a_bad_download_is_never_used(tmp_path: Path) -> None:
    _, get = serving(b"tampered")
    target = tmp_path / "model.pt"
    with pytest.raises(SystemExit, match="expected"):
        fc.fetch(item(b"weights"), target, get)
    assert not target.exists()  # only the .part file is left, under another name
    with pytest.raises(SystemExit, match="earlier run"):  # and it is not silently reused
        fc.fetch(item(b"weights"), target, get)


def test_an_existing_different_file_or_folder_is_not_touched(tmp_path: Path) -> None:
    _, get = serving(b"weights")
    target = tmp_path / "model.pt"
    target.write_bytes(b"mine")
    with pytest.raises(SystemExit, match="not overwritten"):
        fc.fetch(item(b"weights"), target, get)
    assert target.read_bytes() == b"mine"
    folder = tmp_path / "docker-made.pt"
    folder.mkdir()
    with pytest.raises(SystemExit, match="is a folder"):
        fc.fetch(item(b"weights"), folder, get)
    assert folder.is_dir()


def test_the_release_hashes_match_the_readme_and_env_example() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for checkpoint in fc.CHECKPOINTS:
        assert checkpoint.sha256 in readme, checkpoint.name
        assert re.search(rf"^{checkpoint.setting}=.*{checkpoint.name}$", example, re.M)


def test_the_worker_waits_with_a_clear_message_when_a_checkpoint_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "classifier.pt"
    folder.mkdir()  # what Docker mounts for a missing host file
    heartbeat = tmp_path / "heartbeat"
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("CLASSIFIER_CHECKPOINT", str(folder))
    monkeypatch.setenv("SEGMENTATION_CHECKPOINT", str(tmp_path / "absent.pt"))
    monkeypatch.setenv("WORKER_HEARTBEAT", str(heartbeat))
    waited: list[bool] = []

    def load() -> list[Any]:
        raise AssertionError("models must not be loaded")

    monkeypatch.setattr(worker_module, "analyzers_from_env", load)
    monkeypatch.setattr(worker_module, "idle", lambda: waited.append(True))
    worker_module.main([])
    out = capsys.readouterr().out
    print(out)
    assert waited == [True] and not heartbeat.exists()  # unhealthy, no restart
    assert "scripts.fetch_checkpoints" in out and "absent.pt" in out and str(folder) in out
