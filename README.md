**For education only, not for clinical use.**

# Medical Imaging AI (AIML-02)

Status: the NIH ChestX-ray14 metadata is split by patient (EDA in [docs/eda.md](docs/eda.md)).
The preprocessing stage and the DenseNet121 classifier are written and tested on synthetic data,
but **neither has been run on the real images yet**: no model has been trained and there are no
results ([docs/results.md](docs/results.md)). Progress against the brief is tracked in
[docs/brief-evidence.md](docs/brief-evidence.md).

## Layout

- `src/data/split.py`: patient-level train/val/test split (standard library only)
- `src/data/eda.py`: EDA report, written to `docs/eda.md`
- `src/data/spleen.py`: MSD Task09_Spleen download, case-level split (DVC stage `seg_split`), EDA
- `src/data/preprocess.py`: DVC stage `preprocess`: 224px grayscale shards with a checksum
  manifest, plus the ImageNet mean and std for the loader
- `src/models/`: DenseNet121 classifier: config, data and augmentation, training, metrics,
  evaluation
- `configs/classifier.yaml`: the fixed training config
- `notebooks/colab_preprocess.ipynb`: download, `dvc repro preprocess`, `dvc push` on Colab (not
  run yet)
- `notebooks/colab_train.ipynb`: train and evaluate on a Colab GPU (not run yet)
- `dvc.yaml`, `params.yaml`: DVC pipeline (stages `split`, `preprocess`, `seg_split`)
- `tests/`: pytest tests
- `configs/`: configuration
- `data/`: local data, gitignored
- `reports/`: classifier metrics JSON (after the Colab run)
- `docs/`: `brief.md` (the brief), `brief-evidence.md` (requirement tracker), `eda.md`,
  `results.md`, `dicom-notes.md`

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
- The `split` stage runs `python` from the path, so any Python 3.11 works; `split.py` needs no
  packages.
- Labels are NLP-mined from radiology reports, not radiologist-verified.

## Segmentation data (MSD Task09_Spleen)

Medical Segmentation Decathlon Task09: 41 labelled portal-venous CT volumes (NIfTI), label 1 =
spleen. Source: the copy MONAI downloads (`msd-for-monai` on S3, CC-BY-SA 4.0), checked against its
md5. The 20 unlabelled challenge test volumes are not used.

```powershell
.venv\Scripts\python -m src.data.spleen download --root data/raw   # about 1.5 GB, md5 checked
.venv\Scripts\dvc repro seg_split
.venv\Scripts\python -m src.data.spleen eda --task-dir data/raw/Task09_Spleen --split data/seg_splits/spleen.csv --out docs/eda.md
```

- DVC stage `seg_split`: the labelled cases are split by case into 29 train, 6 val and 6 test (seed
  42; `params.yaml`). The stage prints the counts and fails if a case, image or label file is in two
  splits. MSD gives no patient IDs, so one case is treated as one patient.
- Only `dataset.json` is a dependency: the split depends on the case list, not on the volumes. The
  volumes are not tracked by DVC (public, md5-checked download), so they take no Drive space.
- `docs/eda.md` has a spleen section (volumes, spacing, sizes, spleen volume), generated from the
  NIfTI headers and labels. Each EDA generator replaces only its own section of that file.

## Data versioning (DVC remote on Google Drive)

The default DVC remote is a folder on Google Drive, `MyDrive/aiml02-dvc`, configured in
`.dvc/config` as the Colab mount path `/content/drive/MyDrive/aiml02-dvc`.

**Storage rule:** data is built and read on the Colab VM disk. Drive holds only the DVC remote
(written by `dvc push`), the training checkpoints and the reports. The notebooks delete nothing on
Drive. The code enforces this: the preprocess stage and the training loader refuse any data path
under `/content/drive`.

**On Colab**, `notebooks/colab_preprocess.ipynb` does the following:

1. It checks free disk space on the VM and on Drive (`df -h`) and stops if either is short.
2. It downloads the Kaggle zip (about 45 GB) to the VM and reads the images straight from it.
   Only the three metadata files are copied out, and they are checked against `dvc.lock`.
3. It runs `dvc repro preprocess` with `NIH_ZIP` (the zip) and `AIML02_RESUME_DIR`
   (`/content/work/nih_224`, on the VM). Shards are built there and recorded in its manifest only
   when complete. A rerun in the same session skips finished shards; a disconnect wipes the VM, so
   the run starts over. The DVC cache is set to hardlink, so it adds no second copy.
4. It verifies the shards and the manifest, with counts on both sides and a check that no patient
   is in two splits, then runs `dvc push` to the Drive remote.
5. It copies `dvc.lock` to `MyDrive/aiml02-runs/preprocess/dvc.lock.<commit>`.
6. It deletes the zip from the VM only if you type `DELETE`, and only after the checks pass.

Stage `preprocess`: deps `data/splits` and the code; params `preprocess.*` in `params.yaml`; outs
`data/processed/nih_224/` (shards, `shards.csv`, `images.csv`, `preprocess.json`) and
`data/manifests/nih_224.csv` (one row per image: split columns plus shard and sha256).

- Images are stored as 8-bit grayscale PNG, not as normalized floats. Float32 RGB would need about
  67 GB. The ImageNet mean and std are stage params, recorded in `preprocess.json` and applied by
  the loader (`src/models/data.py`), after the channel is copied to RGB.
- The zip is not a DVC dependency. Hashing 45 GB is slow, and the zip is deleted afterwards. The
  stage instead checks that the zip has exactly the split images and records its name and size.
- The stage command uses `$NIH_ZIP`, so it runs on Linux (Colab), not in Windows `cmd`.

Then commit the `dvc.lock` from Drive (step 5), so that `dvc pull` knows the outputs:

```powershell
Copy-Item "G:\My Drive\aiml02-runs\preprocess\dvc.lock.<commit>" dvc.lock
git diff dvc.lock   # only the preprocess entry should be new
```

**On the laptop**, through [Google Drive for desktop](https://www.google.com/drive/download/):

1. Install it and sign in with the same Google account. `My Drive` appears as a drive letter,
   for example `G:\My Drive`.
2. In its settings, either choose **Mirror files**, or right-click `My Drive\aiml02-dvc` and choose
   **Make available offline**. Streamed files are fetched one by one on first read, which makes
   `dvc pull` slow.
3. Point DVC at the local path. This goes to `.dvc/config.local`, which is not committed, so the
   Colab path in `.dvc/config` stays as it is:

   ```powershell
   .venv\Scripts\dvc remote modify --local drive url "G:\My Drive\aiml02-dvc"
   .venv\Scripts\dvc pull preprocess
   .venv\Scripts\dvc status --cloud
   ```

`dvc push` from the laptop goes to the same folder and syncs to Drive in the background. Wait for
Drive for desktop to finish syncing before you run the Colab notebooks.

`notebooks/colab_train.ipynb` runs `dvc pull` to the VM disk (about 4 to 6 GB) and trains from
there. It asserts that the data path is not under `/content/drive`.

### Drive usage

| What | Where on Drive | Size |
|---|---|---|
| DVC remote: NIH 224px shards and manifests | `aiml02-dvc/` | about 4 to 6 GB (estimate: 112,120 PNGs at 35 to 50 KB; not measured yet) |
| DVC remote: split CSVs | `aiml02-dvc/` | 5 MB (measured locally) |
| Classifier checkpoints, 5 epochs | `aiml02-runs/classifier/` | 5 × 84.5 MB = 0.42 GB (measured: DenseNet121 weights + AdamW state) |
| Reports and `dvc.lock` copies | `aiml02-runs/` | under 1 MB |
| **Total** | | **about 4.5 to 6.5 GB** |

The previous layout also kept a resume copy of the shards on Drive, about 12 GB in total. The
preprocess notebook stops if Drive has less than about 7 GB free (6 GB of shards plus 1 GB margin,
minus what is already in the remote).

## Classifier

DenseNet121 with ImageNet weights (torchvision `IMAGENET1K_V1`) and a 14-output head, one per NIH
finding, trained with multi-label BCE. "No Finding" means all 14 targets are 0. The fixed config
is in `configs/classifier.yaml`:

- batch 64, AdamW with lr 1e-4 (weight decay 0.01), at most 5 epochs, AMP on GPU
- one checkpoint per epoch to `MyDrive/aiml02-runs/classifier/`, never overwritten; a rerun
  resumes after the latest one, and refuses if the config changed
- best epoch = highest mean AUROC over the 14 labels on val
- augmentation on the train split only: rotation up to 5 degrees, shift up to 5%, brightness and
  contrast up to 10%. There are no flips, because left and right matter in a chest X-ray (heart
  position). Val and test are only scaled and normalized, and a test checks this.

Evaluation (`src/models/evaluate.py`) scores the best epoch on val and test and writes
`reports/classifier_metrics.json`, `reports/classifier_thresholds.json` and `docs/results.md`:

- AUROC per label, with positive and negative counts
- one threshold per label, chosen on val only (highest F1) and applied unchanged to test
- AUROC by sex and by age band. Ages above 100 are recording errors, so those images are counted
  but not scored.

Run `notebooks/colab_train.ipynb` on a GPU runtime. On Colab, the notebooks use Colab's own torch
build. The versions it uses are recorded in each checkpoint and in the report, and may differ from
the local pins.

## Checks

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```

CI (`.github/workflows/ci.yml`) runs the same checks on CPU, with torch from the PyTorch CPU
index. Training, augmentation, evaluation and the preprocess stage are tested on tiny synthetic
images (`tests/conftest.py`). Tests marked `real_data` skip in CI, because `data/` is not in git.
