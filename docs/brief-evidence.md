# Brief vs evidence

Status: **Yes** = done and evidenced; **Partial** = done with a material deviation or missing
piece; **No** = not done.

The Requirement column is copied word for word from [brief.md](brief.md).

## Steps

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Step 1 | Choose public data (CheXpert or NIH ChestX-ray14 for classification, Medical Segmentation Decathlon for segmentation). Learn the DICOM format. | Yes | [PR #1](https://github.com/manojkumargudge/medical-imaging-ai/pull/1): NIH ChestX-ray14 chosen; metadata matches the published counts (112,120 images, 30,805 patients); patient-level split (DVC stage `split`, no patient in two splits). [PR #4](https://github.com/manojkumargudge/medical-imaging-ai/pull/4): Medical Segmentation Decathlon Task09_Spleen chosen (md5-checked download); 41 labelled cases split by case 29/6/6 (DVC stage `seg_split`, no case in two splits). EDA of both in [eda.md](eda.md). DICOM: notes in [dicom-notes.md](dicom-notes.md); [PR #6](https://github.com/manojkumargudge/medical-imaging-ai/pull/6) converts NIH PNG to DX and NIfTI to a CT series and reads them back with pydicom; a real MSD volume (`spleen_2`, 90 slices) round-trips with 0 HU error and the affine within 1e-4 mm. Caveat: the DX path is tested on synthetic PNGs only (NIH images not downloaded). |
| Step 2 | Preprocess: resize, normalize, augment. Version your data with DVC. | Partial | [PR #2](https://github.com/manojkumargudge/medical-imaging-ai/pull/2): DVC stage `preprocess` (224px, ImageNet mean/std recorded and applied by the loader); train-only augmentation (small rotation, shift, brightness/contrast, no flips) with a test that val/test are not augmented; DVC remote on Drive. [PR #3](https://github.com/manojkumargudge/medical-imaging-ai/pull/3): shards are built on the Colab VM disk and only `dvc push` writes to Drive; training reads from the VM disk (enforced in code). [PR #5](https://github.com/manojkumargudge/medical-imaging-ai/pull/5): 3D preprocessing for CT (1.5 mm, HU window) and train-only light augmentation. Missing: **`preprocess` not run on the real NIH images**, so the remote holds no images yet. Normalization is applied at load time, not stored (see README). |
| Step 3 | Train a classification model with transfer learning (DenseNet or EfficientNet) and a segmentation model (3D U-Net in MONAI). | Partial | [PR #2](https://github.com/manojkumargudge/medical-imaging-ai/pull/2): DenseNet121 (ImageNet weights), 14-label BCE, resumable checkpoints, per-label AUROC, val-chosen thresholds, sex/age subgroups. [PR #5](https://github.com/manojkumargudge/medical-imaging-ai/pull/5): MONAI 3D UNet (spatial_dims=3), Dice+CE, sliding-window val Dice, resumable checkpoints; real-volume transform smoke check only. Missing: **neither model is trained**; no AUROC or Dice results ([results.md](results.md)). |
| Step 4 | Add Grad-CAM so each prediction shows which region influenced it. | No | |
| Step 5 | Install an Orthanc DICOM server, then write a service that pulls new studies, runs inference, and stores results. | No | |
| Step 6 | Build a React viewer showing the scan, the AI overlay, a confidence score, and Approve/Reject buttons for the radiologist. Keep an audit log of every decision. | No | |
| Step 7 | Containerize with Docker and document the model's limits and bias. Include a clear 'for education only, not clinical use' notice. | No | |

## Deliverables

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Deliverable 1 | Working demo (deployed or screen-recorded) | No | |
| Deliverable 2 | GitHub repository with clean commits, README, and setup instructions | Partial | Repository with README and setup instructions; one PR per step of work ([PR #1](https://github.com/manojkumargudge/medical-imaging-ai/pull/1) to [PR #6](https://github.com/manojkumargudge/medical-imaging-ai/pull/6)), CI on every PR. Missing: README covers only the data and model layers built so far. |
| Deliverable 3 | Architecture diagram and data or API documentation | Partial | Data documentation: README (Data, Segmentation data, Data versioning, DICOM conversion), [eda.md](eda.md), [dicom-notes.md](dicom-notes.md). Missing: architecture diagram; no API yet. |
| Deliverable 4 | Test results and, where relevant, performance or accuracy numbers | Partial | Test results: ruff, mypy (strict) and pytest pass in CI on every PR (178 passed, 3 real-data tests skipped in CI, as of PR #6). Missing: no accuracy numbers, because no model is trained. |
| Deliverable 5 | 10-minute final presentation covering problem, design, challenges, results, and next steps | No | |
| Deliverable 6 | Individual reflection from each intern on what they learned and what they owned | No | |
