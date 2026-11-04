**For education only, not for clinical use.**

# Medical Imaging AI (AIML-02)

Status: the NIH ChestX-ray14 metadata is split by patient, and the 41 labelled MSD Task09_Spleen
CT volumes are split by case (EDA of both in [docs/eda.md](docs/eda.md)). The NIH preprocessing
stage, the DenseNet121 classifier and the 3D U-Net are written and tested on synthetic data, but
**none has been run on real images yet**: no model has been trained and there are no results
([docs/results.md](docs/results.md)). Progress against the brief is tracked in
[docs/brief-evidence.md](docs/brief-evidence.md).

## Layout

- `src/data/split.py`: patient-level train/val/test split (standard library only)
- `src/data/eda.py`: EDA report, written to `docs/eda.md`
- `src/data/spleen.py`: MSD Task09_Spleen download, case-level split (DVC stage `seg_split`), EDA
- `src/data/preprocess.py`: DVC stage `preprocess`: 224px grayscale shards with a checksum
  manifest, plus the ImageNet mean and std for the loader
- `src/dicom/convert.py`: NIH PNG -> DICOM DX and spleen NIfTI -> DICOM CT series, read back with pydicom
- `src/models/`: DenseNet121 classifier (config, data and augmentation, training, metrics,
  evaluation), 3D U-Net segmentation (`segmentation.py`), shared checkpoint helpers
- `configs/classifier.yaml`, `configs/segmentation.yaml`: the fixed training configs
- `src/explain/gradcam.py`: Grad-CAM overlays for the classifier
- `notebooks/colab_preprocess.ipynb`: download, `dvc repro preprocess`, `dvc push` on Colab (not
  run yet)
- `notebooks/colab_train.ipynb`: train and evaluate on a Colab GPU (not run yet)
- `notebooks/colab_seg.ipynb`: 3D U-Net training and evaluation on a Colab GPU (not run yet)
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

## DICOM conversion

`src/dicom/convert.py` turns the public data into DICOM for the Orthanc and viewer steps. Both
paths read back with pydicom (`read_dx`, `read_ct_series`), and the tests compare pixels and
geometry. What this proved is in [docs/dicom-notes.md](docs/dicom-notes.md).

```powershell
.venv\Scripts\python -m src.dicom.convert nih --png <image.png> --data-entry data/raw/Data_Entry_2017.csv --out data/dicom/nih/<image>.dcm
.venv\Scripts\python -m src.dicom.convert spleen --nifti data/raw/Task09_Spleen/imagesTr/spleen_2.nii.gz --out data/dicom/spleen_2
```

- NIH PNG -> DX (For Presentation), MONOCHROME2, 8 bits. PatientID is the NIH patient ID;
  PixelSpacing is computed from the original spacing and the resize to 1024. No names or dates.
- MSD NIfTI -> CT series, one file per slice, HU with rescale tags. ImagePositionPatient,
  ImageOrientationPatient, PixelSpacing and SliceThickness are computed from the affine (RAS -> LPS).
- UIDs are new, under pydicom's root, and derived from the source name. Reconverting the same file
  gives the same UIDs.
- Existing files are never overwritten. `data/dicom/` is gitignored, like all data (`*.dcm`).

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

`notebooks/colab_train.ipynb` runs `dvc pull` to the VM disk (about 2.4 GB) and trains from
there. It asserts that the data path is not under `/content/drive`.

### Drive usage

| What | Where on Drive | Size |
|---|---|---|
| DVC remote: NIH 224px shards (`data/processed/nih_224`) | `aiml02-dvc/` | 2.40 GB (measured, `dvc.lock`: 57 shards plus manifests, 60 files; about 21 KB per image) |
| DVC remote: NIH manifest (`data/manifests/nih_224.csv`) | `aiml02-dvc/` | 14.7 MB (measured, `dvc.lock`) |
| DVC remote: split CSVs | `aiml02-dvc/` | 5 MB (measured locally) |
| Classifier checkpoints, 5 epochs | `aiml02-runs/classifier/` | 5 × 84.5 MB = 0.42 GB (measured: DenseNet121 weights + AdamW state) |
| Segmentation checkpoints, every 10 of 200 epochs | `aiml02-runs/segmentation/` | 20 × 57.9 MB = 1.16 GB (measured: 3D UNet weights + Adam state) |
| MSD Task09_Spleen volumes | not on Drive | 0 (downloaded to the VM disk, md5 checked) |
| Reports and `dvc.lock` copies | `aiml02-runs/` | under 1 MB |
| **Total** | | **about 4.0 GB** |

The NIH sizes are from the Colab preprocess run (`dvc.lock`). The checkpoint sizes were measured
locally by saving one checkpoint of each model; no training has run, so none are on Drive yet. The preprocess notebook still checks for about 7 GB
free on Drive, because it was written with the earlier 6 GB shard estimate; the real need is about
2.4 GB.

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

`python -m src.models.evaluate` now writes report version 2 by default:
`reports/classifier_metrics_v2.json`, `reports/thresholds_v2.json` and `docs/results_v2.md`.
Version 1 (`--version 1`) is unchanged, and an existing report is never replaced without
`--overwrite`. Version 2 adds:

- a second threshold set, also chosen on val only: per label, the highest threshold with val
  sensitivity of at least 0.80. Labels with fewer than 30 val positives get no threshold (score
  only). Test sensitivity and specificity are reported for both sets.
- 95% bootstrap intervals (1000 resamples, seed 42, percentile) for per-label test AUROC and the
  test mean AUROCs. Patients are resampled, each with all of its images.
- sex and age-band means over one label set per grouping: the labels with at least 10 positives in
  every group of that split, so that the groups are compared on the same labels.

On Colab, after training (`--checkpoint-dir` points at the Drive checkpoints; the reports go to
the same Drive folder):

```
!python -m src.models.evaluate --checkpoint-dir /content/drive/MyDrive/aiml02-runs/classifier \
    --metrics /content/drive/MyDrive/aiml02-runs/classifier/reports/classifier_metrics_v2.json \
    --thresholds /content/drive/MyDrive/aiml02-runs/classifier/reports/thresholds_v2.json \
    --results /content/drive/MyDrive/aiml02-runs/classifier/reports/results_v2.md \
    --repo-commit "$(git rev-parse HEAD)"
```

Run `notebooks/colab_train.ipynb` on a GPU runtime. On Colab, the notebooks use Colab's own torch
build. The versions it uses are recorded in each checkpoint and in the report, and may differ from
the local pins.

## Grad-CAM

`src/explain/gradcam.py` shows which regions of a chest X-ray raised one label's score.

- **Target layer:** the classifier's last dense block (`features.denseblock4`).
- **Preprocessing:** the image goes through the evaluation path: `resize_png` from the
  preprocess stage, then the test-split transform from `src/models/data.py`.
- **Output:** a map the size of the model input with values in [0, 1].

```
python -m src.explain.gradcam --checkpoint epoch_003.pt --image 00000013_045.png     --label Cardiomegaly --out-dir reports/gradcam
```

This writes:

- an overlay PNG
- a JSON with the score, plus the threshold and reliability from `configs/viewer_thresholds.yaml`

It runs on CPU and never overwrites a file. Tests check four things:

- the map size
- the value range
- determinism
- the sanity check: the map changes when the top layers are randomized

The sanity check was also run on the trained checkpoint (`epoch_003.pt`). The classifier and
the last dense block were randomized, and the trained and randomized maps compared:

- correlation -0.19 for 00000013_045 (Cardiomegaly)
- correlation 0.23 for 00000013_021 (Effusion)
- correlation -0.18 for 00000013_020 (Pneumothorax)

So the maps depend on the trained weights. That does not show they mark the finding.

Twelve demo overlays from test-split images are in
[docs/figures/gradcam](docs/figures/gradcam/README.md): two sets chosen by fixed rules, with
scores and where each map sits. A map shows what raised the score, not where a finding is.
Caveats from the demo:

- Devices: 4 of the 6 first-set images show a pacemaker, including both Cardiomegaly images.
- Label specificity: maps for different labels can cover nearly the same region.

## Segmentation model (3D U-Net)

MONAI `UNet` with `spatial_dims=3`, channels 16-32-64-128-256, residual units, batch norm, 2
output channels (background, spleen). The fixed config is in `configs/segmentation.yaml`; the code is
`src/models/segmentation.py`.

- Preprocessing (every split): reorient to RAS, resample to 1.5 mm isotropic, CT window -57 to
  164 HU mapped to [0, 1] and clipped, crop to the body.
- Train split only: four random 96^3 crops per volume (half centred on spleen), then light
  augmentation: a small random affine (rotation up to 10 degrees, scaling up to 10%, probability
  0.2) and an intensity shift of up to 0.1. There are no flips, because the spleen is on the left.
  A test checks that val and test get no random transform.
- Dice + cross-entropy loss, Adam with lr 1e-4, batch 2 volumes (8 crops), at most 200 epochs, AMP
  on GPU.
- Every 10 epochs: sliding-window inference over each whole val volume (96^3 windows, overlap
  0.25), spleen Dice per case, and a checkpoint to `MyDrive/aiml02-runs/segmentation/`. Checkpoints
  are never overwritten; a rerun resumes after the latest one, and refuses if the config changed.
  The best checkpoint has the highest val mean Dice.
- `python -m src.models.segmentation evaluate` scores the best checkpoint on val and test (Dice
  per case) and writes `reports/segmentation_metrics.json`.

Run `notebooks/colab_seg.ipynb` on a GPU runtime. It downloads the data to the VM disk, rebuilds
`seg_split` and stops if `dvc.lock` changes, then trains and evaluates.

## Orthanc and the results database

`docker-compose.yml` runs two services, for synthetic or public de-identified data only:

- **Orthanc** (`orthancteam/orthanc:26.9.1`):
  - REST API and web UI on `127.0.0.1:8042`, with basic auth on.
  - DICOM on `127.0.0.1:4242`. The DICOM protocol itself has no password, so the port is bound
    to localhost only.
  - A study becomes `StableStudy` 5 s after its last instance arrives.
- **Results Postgres** (`postgres:16.14`) on `127.0.0.1:5434`, separate from the AIML-01 database
  on 5432. Port 5433 is taken by a local PostgreSQL 18 service on the development machine. The
  port is `RESULTS_DB_PORT` in `.env`.

```
cp .env.example .env              # then set both passwords; .env is never committed
docker compose up -d --wait
python -m scripts.upload_demo     # 2 NIH demo PNGs as DX, MSD spleen_14 as a CT series
```

`scripts/upload_demo.py` converts the files with `src/dicom/convert.py`: no names or dates, and
UIDs derived from the source. A rerun therefore reports `AlreadyStored` instead of duplicating.
It then checks that each study can be found in Orthanc.

`tests/test_orthanc_integration.py` uploads synthetic DX and CT studies, finds them, and checks
that a wrong password is refused.

**Integration tests never use the demo stack.** They run against a separate Compose project,
`aiml02-test`, configured by `.env.test`:

| setting | demo stack | test stack |
|---|---|---|
| ports | 8042, 4242, 5434 | 18042, 14242, 15434 |
| Docker volumes | `aiml02_*` | `aiml02-test_*` |
| Orthanc name | `aiml02` | `aiml02-test` |
| database | `aiml02_results` | `aiml02_results_test` |

The tests read only `.env.test`, and they check the Orthanc name and the database name before
doing anything.

```
python -m scripts.make_test_env                               # writes .env.test, never overwrites
docker compose -p aiml02-test --env-file .env.test up -d --wait
AIML02_INTEGRATION=1 python -m pytest -m integration
```

The CI job `integration` does the same.

**Resetting the demo.** `scripts/reset_demo.py` returns the demo stack to the three demo studies
only. It deletes:

- every study in the demo Orthanc
- every row of `inference_results`
- the worker's output files under `results/`

It then re-uploads with `upload_demo.py`. It shows the counts and asks you to type `wipe` before
deleting anything. It checks that it is on the demo stack by Orthanc name and database name, and
it keeps the worker's change-log position. Stop the worker before running it, and start it again
afterwards.

## Inference worker

`src/service/worker.py` polls Orthanc's `/changes` for `StableStudy` events and stores the last
change sequence in the results database, so a restart resumes where it stopped. For each study:

- **DX/CR series:** the DenseNet121 classifier with Grad-CAM (all 14 labels). It records:
  - scores, which are not calibrated probabilities
  - the threshold, its set and the reliability flag from `configs/viewer_thresholds.yaml`
  - one overlay PNG per label
  The 8-bit MONOCHROME2 images converted from NIH PNGs go through unchanged. Other DX images are
  windowed and scaled to 8 bits, which differs from the training data.
- **CT series:** the 3D U-Net with the evaluation transforms and sliding-window inference. It
  records the spleen mask as NIfTI in the model's space (RAS, 1.5 mm, body crop, with its
  affine) and the spleen volume.

Results go to the Postgres table `inference_results`:

- **Key:** one row per (StudyInstanceUID, checkpoint sha256), so reruns and repeated
  `StableStudy` events add nothing.
- **Columns:** model, checkpoint name and hash, status, results JSON, seconds, and created_at.
  Output paths in the results JSON are relative to `RESULTS_DIR`.
- **Errors:** a study that fails is stored with status `error`, and the worker moves on. Delete
  that row to retry it.

The worker runs CPU only. The image (`Dockerfile`) holds code and configs only. The checkpoints
are files on this machine, kept in `C:\aiml02-checkpoints\` (outside the repository):

- `epoch_003.pt`: the classifier, sha256 `1a35108e…501708`
- `epoch_180.pt`: the U-Net, sha256 `8839d537…26cc36`

They are set as `CLASSIFIER_CHECKPOINT` and `SEGMENTATION_CHECKPOINT` in `.env` and mounted
read-only. They are never in the image or in git, and a test checks both. The worker logs each
checkpoint's sha256 at start-up, and every result row records it.

Worker time per study on CPU, from the `seconds` column, one run each after
`scripts/reset_demo.py`:

| study | model | time |
|---|---|---|
| DX 00000116_000 (1 image) | classifier + Grad-CAM for 14 labels | 24.2 s |
| DX 00000013_021 (1 image) | classifier + Grad-CAM for 14 labels | 16.2 s |
| CT spleen_14 (54 slices) | U-Net, sliding window | 58.5 s |

- **Machine:** Intel Core i5-1235U laptop, Docker Desktop VM with 12 CPUs and 10 GB.
- **Threads:** the worker uses `WORKER_THREADS=4`.
- **What is timed:** download from Orthanc, inference and writing outputs.
- **First DX study:** it includes first-call overhead, so it is slower than the second.
- **Grad-CAM:** most of the DX time. It runs one forward and backward pass per label.

```
docker compose -f docker-compose.yml -f docker-compose.worker.yml up -d --build
python -m scripts.upload_demo
python -m scripts.check_worker    # real checks against the published outputs
```

Tests:

- `tests/test_worker.py` runs the worker with fake models and an in-memory Orthanc.
- `tests/test_worker_integration.py` runs it against the real services, using temporary tables
  (CI job `integration`).
- `scripts/check_worker.py` compares the worker on the real demo studies with the published
  outputs:
  - DX: scores must equal the Grad-CAM JSON scores within 1e-6.
  - CT: Dice on `spleen_14` must match `segmentation_metrics.json` within 0.005.

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
