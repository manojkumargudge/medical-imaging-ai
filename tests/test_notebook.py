import json
from pathlib import Path
from typing import Any

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_preprocess.ipynb"


def code_cells() -> list[dict[str, Any]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_code_cells_compile() -> None:
    cells = code_cells()
    assert cells
    for number, cell in enumerate(cells):
        compile("".join(cell["source"]), f"cell {number}", "exec")


def test_notebook_has_not_been_run() -> None:
    cells = code_cells()
    assert cells
    assert all(not cell["outputs"] and cell["execution_count"] is None for cell in cells)


def test_raw_data_is_kept_off_drive() -> None:
    source = "\n".join("".join(cell["source"]) for cell in code_cells())
    assert 'RAW = Path("/content/raw")' in source
    assert '"-p", str(RAW)' in source
    assert "preprocess.write_shards(names, images.read, OUT, WORK)" in source
