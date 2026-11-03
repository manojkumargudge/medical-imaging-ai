# Brief vs evidence

Status: **Yes** = done and evidenced; **Partial** = done with a material deviation or missing
piece; **No** = not done.

The Requirement column is copied word for word from [brief.md](brief.md).

## Steps

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Step 1 | Choose public data (CheXpert or NIH ChestX-ray14 for classification, Medical Segmentation Decathlon for segmentation). Learn the DICOM format. | Partial | [PR #1](https://github.com/manojkumargudge/medical-imaging-ai/pull/1): NIH ChestX-ray14 (Kaggle `nih-chest-xrays/data`) chosen; metadata checked against the published counts (112,120 images, 30,805 patients); patient-level split in `src/data/split.py` (DVC stage `split`, no patient in two splits); EDA in [eda.md](eda.md). DICOM notes in [dicom-notes.md](dicom-notes.md). Missing: no segmentation dataset (Medical Segmentation Decathlon) chosen yet; no DICOM file has been read in code (NIH images are PNG). |
| Step 2 | Preprocess: resize, normalize, augment. Version your data with DVC. | Partial | Branch `classifier`: DVC stage `preprocess` (`dvc.yaml`, `src/data/preprocess.py`) resizes to 224px and records the ImageNet mean/std that the loader applies (`src/models/data.py`); train-only augmentation (small rotation, shift, brightness/contrast, no flips), with a test that val/test are not augmented (`tests/test_augment.py`); DVC remote on Drive (`.dvc/config`); Colab notebook `notebooks/colab_preprocess.ipynb`. Missing: **not run on the real images**, so nothing is in the remote yet; tested on synthetic images only. Normalization is applied at load time, not stored (see README). |
| Step 3 | Train a classification model with transfer learning (DenseNet or EfficientNet) and a segmentation model (3D U-Net in MONAI). | Partial | Branch `classifier`: DenseNet121 with ImageNet weights, 14-label BCE (`src/models/`, `configs/classifier.yaml`), resumable per-epoch checkpoints, evaluation with per-label AUROC, val-chosen thresholds and sex/age subgroups; `notebooks/colab_train.ipynb`. Missing: **not trained**, no results ([results.md](results.md)); no segmentation model. |
| Step 4 | Add Grad-CAM so each prediction shows which region influenced it. | No | |
| Step 5 | Install an Orthanc DICOM server, then write a service that pulls new studies, runs inference, and stores results. | No | |
| Step 6 | Build a React viewer showing the scan, the AI overlay, a confidence score, and Approve/Reject buttons for the radiologist. Keep an audit log of every decision. | No | |
| Step 7 | Containerize with Docker and document the model's limits and bias. Include a clear 'for education only, not clinical use' notice. | No | |

## Deliverables

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Deliverable 1 | Working demo (deployed or screen-recorded) | No | |
| Deliverable 2 | GitHub repository with clean commits, README, and setup instructions | No | |
| Deliverable 3 | Architecture diagram and data or API documentation | No | |
| Deliverable 4 | Test results and, where relevant, performance or accuracy numbers | No | |
| Deliverable 5 | 10-minute final presentation covering problem, design, challenges, results, and next steps | No | |
| Deliverable 6 | Individual reflection from each intern on what they learned and what they owned | No | |
