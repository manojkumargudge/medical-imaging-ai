**For education only, not for clinical use.**

# Medical Imaging AI (AIML-02)

Status: data code is ready, but the notebook has not been run. No data has been downloaded and no
model has been trained.

## Layout

- `src/data/`: `nih.py` (labels, patient-level splits, manifest, EDA tables), `shards.py` (resized
  image shards)
- `notebooks/01_data_colab.ipynb`: download, manifest, EDA and shards on Colab
- `tests/`: pytest tests
- `configs/`: configuration
- `data/`: local data, gitignored (versioned with DVC)
- `docs/`: `brief-evidence.md` (requirement tracker), `dicom-notes.md`

## Data

NIH ChestX-ray14 from Kaggle (`nih-chest-xrays/data`). Raw images stay on the Colab VM; only the
manifest, reports and 224x224 shards go to Drive. The notebook's last cell lists the local steps.

The test split is the official `test_list.txt`. Validation is carved from `train_val_list.txt` by
patient (10% of patients, seed 42).

DVC tracks `data/manifests/` in a local remote, `C:\dvc-remote-medical`, so `dvc pull` works only on
this machine. It is not a backup. The tracked folder is empty until the notebook's manifest is added.

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
