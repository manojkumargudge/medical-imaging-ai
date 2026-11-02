# Brief vs evidence

Status: **Yes** = done and evidenced; **Partial** = done with a material deviation or missing
piece; **No** = not done.

## Steps

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| S1 | Public data (CheXpert or NIH ChestX-ray14 classification; Medical Segmentation Decathlon segmentation); learn DICOM. | No | |
| S2 | Preprocess: resize, normalize, augment; version data with DVC. | No | |
| S3 | Classification with transfer learning (DenseNet/EfficientNet) + segmentation 3D U-Net in MONAI. | No | |
| S4 | Grad-CAM showing region that influenced each prediction. | No | |
| S5 | Orthanc DICOM server + service that pulls new studies, runs inference, stores results. | No | |
| S6 | React viewer: scan, AI overlay, confidence, Approve/Reject; audit log of every decision. | No | |
| S7 | Docker; document model limits and bias; "for education only, not clinical use" notice. | No | |

## Deliverables

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| D1 | Working demo. | No | |
| D2 | Repo, clean commits, README, setup. | No | |
| D3 | Architecture diagram + data/API docs. | No | |
| D4 | Test results + accuracy numbers. | No | |
| D5 | 10-min presentation. | No | |
| D6 | Individual reflection. | No | |
