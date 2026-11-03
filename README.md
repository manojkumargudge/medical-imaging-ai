**For education only, not for clinical use.**

# Medical Imaging AI (AIML-02)

Status: Step 1 data layer. NIH ChestX-ray14 metadata is split by patient, and the EDA is in
[docs/eda.md](docs/eda.md). No images have been downloaded or preprocessed, and no model has been
trained.

## Layout

- `src/data/split.py`: patient-level train/val/test split (standard library only)
- `src/data/eda.py`: EDA report, written to `docs/eda.md`
- `src/data/preprocess.py`: 224px grayscale shards with a checksum manifest (used on Colab)
- `notebooks/colab_preprocess.ipynb`: download and resize on Colab (not run yet)
- `dvc.yaml`, `params.yaml`: DVC pipeline (stage `split`)
- `tests/`: pytest tests
- `configs/`: configuration
- `data/`: local data, gitignored
- `docs/`: `eda.md`, `brief-evidence.md` (requirement tracker), `dicom-notes.md`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Data

Dataset: NIH ChestX-ray14 on Kaggle, `nih-chest-xrays/data` (CC0). The three metadata files
come from the Kaggle API, without credentials:

```powershell
foreach ($f in "Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt") {
    .venv\Scripts\kaggle datasets download nih-chest-xrays/data -f $f -p data/raw
}
```

`Data_Entry_2017.csv` and `train_val_list.txt` arrive as `.zip`; extract them in `data/raw/`.

Then build the splits and the report:

```powershell
.venv\Scripts\dvc repro split
.venv\Scripts\python -m src.data.eda --splits-dir data/splits --out docs/eda.md
```

- The test split is the official `test_list.txt`. Val is 10% of the `train_val_list.txt` patients,
  chosen with seed 42 (`params.yaml`). No patient is in two splits; the stage fails if one is.
- The stage runs `python` from the path, so any Python 3.11 works; `split.py` needs no packages.
- Labels are NLP-mined from radiology reports, not radiologist-verified.
- No DVC remote is set up, so `data/splits/` exists only where `dvc repro` has run.

## Checks

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```

CI (`.github/workflows/ci.yml`) runs the same checks. Tests marked `real_data` skip there, because
`data/` is not in git.
