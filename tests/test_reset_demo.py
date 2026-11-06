"""scripts/reset_demo.py with a fake Orthanc and store: nothing real is deleted."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from scripts import reset_demo
from src.service.orthanc import Orthanc
from src.service.stack import DEMO, TEST


class FakeOrthanc(Orthanc):
    def __init__(self, name: str, studies: list[str]) -> None:
        super().__init__("http://fake", "u", "p")
        object.__setattr__(self, "state", {"name": name, "studies": list(studies), "deleted": []})

    def get_json(self, path: str) -> Any:
        state = self.state  # type: ignore[attr-defined]
        return {"Name": state["name"]} if path == "/system" else list(state["studies"])

    def delete(self, path: str) -> None:
        state = self.state  # type: ignore[attr-defined]
        study = path.removeprefix("/studies/")
        state["studies"].remove(study)
        state["deleted"].append(study)


@dataclass
class FakeStore:
    name: str = DEMO["db_name"]
    rows: int = 5
    deleted: list[int] = field(default_factory=list)
    kept: dict[str, int] = field(default_factory=lambda: {"reviews": 0, "audit_log": 7})
    reviewed: int = 0

    def database(self) -> str:
        return self.name

    def count_results(self) -> int:
        return self.rows

    def delete_results(self) -> int:
        self.deleted.append(self.rows)
        self.rows = 0
        return self.deleted[-1]

    def count_kept(self) -> dict[str, int]:
        return dict(self.kept)

    def reviewed_results(self) -> int:
        return self.reviewed


def uploader(orthanc: Orthanc) -> list[str]:
    orthanc.state["studies"] += ["demo-dx-1", "demo-dx-2", "demo-ct"]  # type: ignore[attr-defined]
    return ["1.2.1", "1.2.2", "1.2.3"]


def results(tmp_path: Path) -> Path:
    folder = tmp_path / "results" / "worker"
    (folder / "study" / "model").mkdir(parents=True)
    (folder / "study" / "model" / "overlay.png").write_bytes(b"x")
    return folder


def test_refusing_deletes_nothing(tmp_path: Path) -> None:
    orthanc, store, folder = (
        FakeOrthanc(DEMO["orthanc_name"], ["a", "b"]),
        FakeStore(),
        results(tmp_path),
    )
    done = reset_demo.reset(orthanc, store, folder, uploader, lambda _: False, tmp_path / "results")
    assert done is False
    assert orthanc.state["studies"] == ["a", "b"] and store.rows == 5  # type: ignore[attr-defined]
    assert len(reset_demo.output_files(folder)) == 1


def test_confirmed_reset_leaves_only_the_demo_studies(tmp_path: Path) -> None:
    orthanc = FakeOrthanc(DEMO["orthanc_name"], ["old-1", "old-2", "synthetic"])
    store, folder = FakeStore(), results(tmp_path)
    questions: list[str] = []

    def confirm(question: str) -> bool:
        questions.append(question)
        return True

    done = reset_demo.reset(
        orthanc,
        store,
        folder,
        uploader,
        confirm,
        tmp_path / "results",
    )
    assert done is True
    assert "3 studies" in questions[0] and "5 rows" in questions[0] and "1 files" in questions[0]
    state = orthanc.state  # type: ignore[attr-defined]
    assert state["deleted"] == ["old-1", "old-2", "synthetic"]
    assert state["studies"] == ["demo-dx-1", "demo-dx-2", "demo-ct"]
    assert store.deleted == [5] and store.rows == 0
    assert folder.exists() and reset_demo.output_files(folder) == []


def test_only_the_demo_stack_is_reset(tmp_path: Path) -> None:
    folder = results(tmp_path)
    with pytest.raises(RuntimeError, match="expected 'aiml02'"):
        reset_demo.reset(
            FakeOrthanc(TEST["orthanc_name"], ["a"]),
            FakeStore(),
            folder,
            uploader,
            lambda _: True,
            tmp_path / "results",
        )
    with pytest.raises(RuntimeError, match="is not 'aiml02_results'"):
        reset_demo.reset(
            FakeOrthanc(DEMO["orthanc_name"], ["a"]),
            FakeStore(TEST["db_name"]),
            folder,
            uploader,
            lambda _: True,
            tmp_path / "results",
        )


@pytest.mark.parametrize("where", ["outside", "root"])
def test_files_are_deleted_only_inside_results(tmp_path: Path, where: str) -> None:
    folder = tmp_path / "elsewhere" if where == "outside" else tmp_path / "results"
    folder.mkdir()
    with pytest.raises(ValueError, match="not deleting"):
        reset_demo.reset(
            FakeOrthanc(DEMO["orthanc_name"], ["a"]),
            FakeStore(),
            folder,
            uploader,
            lambda _: True,
            tmp_path / "results",
        )


def test_a_reviewed_result_stops_the_reset_before_anything_is_deleted(tmp_path: Path) -> None:
    orthanc = FakeOrthanc(DEMO["orthanc_name"], ["a", "b"])
    store, folder = FakeStore(reviewed=1), results(tmp_path)
    with pytest.raises(RuntimeError, match="have reviews"):
        reset_demo.reset(orthanc, store, folder, uploader, lambda _: True, tmp_path / "results")
    assert orthanc.state["studies"] == ["a", "b"] and store.rows == 5  # type: ignore[attr-defined]
    assert len(reset_demo.output_files(folder)) == 1


def test_the_kept_tables_must_not_change(tmp_path: Path) -> None:
    class Shrinking(FakeStore):
        def delete_results(self) -> int:
            self.kept["audit_log"] -= 1
            return super().delete_results()

    with pytest.raises(RuntimeError, match="kept tables changed"):
        reset_demo.reset(
            FakeOrthanc(DEMO["orthanc_name"], ["a"]),
            Shrinking(),
            results(tmp_path),
            uploader,
            lambda _: True,
            tmp_path / "results",
        )
