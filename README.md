**For education only, not for clinical use.**

# Medical Imaging AI (AIML-02)

Status: setup only. No data has been downloaded and no model has been trained.

## Layout

- `src/`: code
- `tests/`: pytest tests
- `configs/`: configuration
- `data/`: local data, gitignored (versioned with DVC)
- `docs/`: `brief-evidence.md` (requirement tracker), `dicom-notes.md`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Checks

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```
