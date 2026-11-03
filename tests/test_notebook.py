import json
from pathlib import Path
from typing import Any

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
PREPROCESS = NOTEBOOKS / "colab_preprocess.ipynb"
TRAIN = NOTEBOOKS / "colab_train.ipynb"


def code_cells(path: Path) -> list[dict[str, Any]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def sources(path: Path) -> list[str]:
    return ["".join(cell["source"]) for cell in code_cells(path)]


def position(cells: list[str], text: str) -> int:
    """Index of the first cell containing text."""
    return next(index for index, source in enumerate(cells) if text in source)


@pytest.mark.parametrize("path", [PREPROCESS, TRAIN])
def test_code_cells_compile(path: Path) -> None:
    cells = sources(path)
    assert cells
    for number, source in enumerate(cells):
        compile(source, f"{path.name} cell {number}", "exec")


@pytest.mark.parametrize("path", [PREPROCESS, TRAIN])
def test_notebook_has_not_been_run(path: Path) -> None:
    cells = code_cells(path)
    assert cells
    assert all(not cell["outputs"] and cell["execution_count"] is None for cell in cells)


@pytest.mark.parametrize("path", [PREPROCESS, TRAIN])
def test_disk_is_checked_in_the_first_code_cell(path: Path) -> None:
    first = sources(path)[0]
    assert 'run("df", "-h"' in first
    assert "raise RuntimeError" in first


def test_disk_check_comes_before_the_download() -> None:
    cells = sources(PREPROCESS)
    assert position(cells, '"df", "-h"') < position(cells, '"kaggle", "datasets", "download"')


def test_raw_zip_stays_on_the_vm_and_is_read_without_extracting() -> None:
    source = "\n".join(sources(PREPROCESS))
    assert 'RAW = Path("/content/raw")' in source
    assert '"-p", RAW' in source
    assert "--unzip" not in source and "extractall" not in source
    assert '"NIH_ZIP": str(ZIP)' in source


def test_stage_resumes_from_the_drive_manifest() -> None:
    source = "\n".join(sources(PREPROCESS))
    assert 'RESUME = Path("/content/drive/MyDrive/aiml02-work/nih_224")' in source
    assert '"AIML02_RESUME_DIR": str(RESUME)' in source
    assert 'run("dvc", "repro", "preprocess"' in source


def test_zip_is_deleted_only_after_verification_and_confirmation() -> None:
    cells = sources(PREPROCESS)
    deleting = [index for index, source in enumerate(cells) if "ZIP.unlink()" in source]
    assert len(deleting) == 1
    cell = cells[deleting[0]]
    assert deleting[0] > position(cells, "MANIFEST_VERIFIED = True")
    assert cell.index('globals().get("MANIFEST_VERIFIED")') < cell.index("ZIP.unlink()")
    assert cell.index("input(") < cell.index("ZIP.unlink()")
    assert 'if answer.strip() == "DELETE":' in cell


def test_preprocess_pushes_to_the_drive_remote() -> None:
    source = "\n".join(sources(PREPROCESS))
    assert 'run("dvc", "push"' in source
    assert "/content/drive/MyDrive/aiml02-dvc" in source


def test_train_notebook_uses_the_fixed_config_and_resumes() -> None:
    source = "\n".join(sources(TRAIN))
    assert 'load_config(REPO / "configs" / "classifier.yaml")' in source
    assert "train.train(config)" in source
    assert "evaluate.evaluate(config" in source
    assert "torch.cuda.is_available()" in source
