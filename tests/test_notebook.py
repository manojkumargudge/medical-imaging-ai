import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_colab.ipynb"


def code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_code_cells_compile() -> None:
    cells = code_cells()
    assert cells
    for number, source in enumerate(cells):
        compile(source, f"cell {number}", "exec")


def test_notebook_has_no_stored_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    assert all(not cell["outputs"] for cell in code)


def test_raw_data_is_kept_off_drive() -> None:
    source = "\n".join(code_cells())
    assert 'RAW = Path("/content/raw")' in source
    assert '"-p", str(RAW)' in source
