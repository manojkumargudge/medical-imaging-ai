# Brief vs evidence

Status: **Yes** = done and evidenced; **Partial** = done with a material deviation or missing
piece; **No** = not done.

The Requirement column is copied word for word from [brief.md](brief.md).

## Steps

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Step 1 | Choose public data (CheXpert or NIH ChestX-ray14 for classification, Medical Segmentation Decathlon for segmentation). Learn the DICOM format. | Yes | [PR #1](https://github.com/manojkumargudge/medical-imaging-ai/pull/1): NIH ChestX-ray14 chosen; metadata matches the published counts (112,120 images, 30,805 patients); patient-level split (DVC stage `split`, no patient in two splits). [PR #4](https://github.com/manojkumargudge/medical-imaging-ai/pull/4): Medical Segmentation Decathlon Task09_Spleen chosen (md5-checked download); 41 labelled cases split by case 29/6/6 (DVC stage `seg_split`, no case in two splits). EDA of both in [eda.md](eda.md). DICOM: notes in [dicom-notes.md](dicom-notes.md); [PR #6](https://github.com/manojkumargudge/medical-imaging-ai/pull/6) converts NIH PNG to DX and NIfTI to a CT series and reads them back with pydicom; a real MSD volume (`spleen_2`, 90 slices) round-trips with 0 HU error and the affine within 1e-4 mm. Caveat: the DX path is tested on synthetic PNGs only (NIH images not downloaded). |
| Step 2 | Preprocess: resize, normalize, augment. Version your data with DVC. | Yes | [PR #2](https://github.com/manojkumargudge/medical-imaging-ai/pull/2): DVC stage `preprocess` (224px, ImageNet mean/std recorded and applied by the loader); train-only augmentation (small rotation, shift, brightness/contrast, no flips) with a test that val/test are not augmented; DVC remote on Drive. [PR #3](https://github.com/manojkumargudge/medical-imaging-ai/pull/3): shards are built on the Colab VM disk and only `dvc push` writes to Drive; training reads from the VM disk (enforced in code). [PR #5](https://github.com/manojkumargudge/medical-imaging-ai/pull/5): 3D preprocessing for CT (1.5 mm, HU window) and train-only light augmentation. `preprocess` run on Colab on all 112,120 real NIH images. Counts verified: train 77,757, val 8,767, test 25,596, with 0 patients in two splits. Outputs recorded in `dvc.lock` (commit 27bf788) and pushed to the Drive remote. Normalization is applied at load time, not stored (see README). |
| Step 3 | Train a classification model with transfer learning (DenseNet or EfficientNet) and a segmentation model (3D U-Net in MONAI). | Yes | [PR #2](https://github.com/manojkumargudge/medical-imaging-ai/pull/2): DenseNet121 (ImageNet weights), 14-label BCE, resumable checkpoints, per-label AUROC, val-chosen thresholds, sex/age subgroups. [PR #5](https://github.com/manojkumargudge/medical-imaging-ai/pull/5): MONAI 3D UNet (spatial_dims=3), Dice+CE, sliding-window val Dice, resumable checkpoints; real-volume transform smoke check only. Both models trained on Colab. DenseNet121: best epoch 3 of 5 (by val mean AUROC); mean AUROC on the NIH ChestX-ray14 official test split is 0.804 (95% CI 0.797-0.810), measured against NLP-mined labels ([results_v2.md](results_v2.md)). 3D U-Net: best epoch 180 of 200 (by val mean Dice); mean Dice on MSD Task09_Spleen is 0.9195 on val and 0.9176 on test (6 test cases) ([results_seg.md](results_seg.md)). |
| Step 4 | Add Grad-CAM so each prediction shows which region influenced it. | Yes | [PR #13](https://github.com/manojkumargudge/medical-imaging-ai/pull/13): Grad-CAM for the DenseNet121 classifier (`src/explain/gradcam.py`, target = last dense block, evaluation preprocessing reused). The CLI writes an overlay plus a JSON with score, threshold and reliability per prediction. Tests cover map size, values in [0, 1], determinism and a weight-randomization sanity check. On the trained checkpoint, randomizing the top layers changes the map (correlation -0.19 to 0.23). Demo: 12 NIH test-split positives (Cardiomegaly, Effusion, Pneumothorax), chosen by fixed rules ([gradcam demo](figures/gradcam/README.md)). Caveats: (1) 4 of the 6 first-set images show a cardiac device, including both Cardiomegaly images. The heat sits on the heart, not on the device, but a device-Cardiomegaly confound is possible. (2) Maps may not be label-specific: patient 13's Effusion and Pneumothorax maps cover nearly the same region (correlation 0.79). (3) All 4 missed positives have maps outside the lungs, at the image edge or just below the heart. (4) 12 images show what maps look like, not how often they mark the finding. (5) Grad-CAM covers the classifier only, not the U-Net. |
| Step 5 | Install an Orthanc DICOM server, then write a service that pulls new studies, runs inference, and stores results. | No | |
| Step 6 | Build a React viewer showing the scan, the AI overlay, a confidence score, and Approve/Reject buttons for the radiologist. Keep an audit log of every decision. | No | |
| Step 7 | Containerize with Docker and document the model's limits and bias. Include a clear 'for education only, not clinical use' notice. | No | |

## Deliverables

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| Deliverable 1 | Working demo (deployed or screen-recorded) | No | |
| Deliverable 2 | GitHub repository with clean commits, README, and setup instructions | Partial | Repository with README and setup instructions; one PR per step of work ([PR #1](https://github.com/manojkumargudge/medical-imaging-ai/pull/1) to [PR #6](https://github.com/manojkumargudge/medical-imaging-ai/pull/6)), CI on every PR. Missing: README covers only the data and model layers built so far. |
| Deliverable 3 | Architecture diagram and data or API documentation | Partial | Data documentation: README (Data, Segmentation data, Data versioning, DICOM conversion), [eda.md](eda.md), [dicom-notes.md](dicom-notes.md). Missing: architecture diagram; no API yet. |
| Deliverable 4 | Test results and, where relevant, performance or accuracy numbers | Yes | Test results: ruff, mypy (strict) and pytest pass in CI on every PR (178 passed, 3 real-data tests skipped in CI, as of PR #6). Classifier metrics measured on NIH ChestX-ray14 val and official test splits in [results_v2.md](results_v2.md): per-label AUROC with 95% CIs, test sensitivity and specificity at val-chosen thresholds, and sex and age subgroups. Segmentation metrics measured on MSD Task09_Spleen val and test cases in [results_seg.md](results_seg.md): per-case Dice, mean, and min-max. |
| Deliverable 5 | 10-minute final presentation covering problem, design, challenges, results, and next steps | No | |
| Deliverable 6 | Individual reflection from each intern on what they learned and what they owned | No | |

## Limitations

### Classifier

**For education only, not for clinical use.** All numbers come from the DenseNet121 classifier
(checkpoint epoch 3) on NIH ChestX-ray14. The labels were NLP-mined from reports and were not
verified by radiologists. Numbers are for the official `test_list.txt` split unless the val
split is named. Intervals are 95% bootstrap intervals (1000 patient-level resamples). Source:
[results_v2.md](results_v2.md) and `reports/classifier_metrics_v2.json`.

- **Overall:** mean test AUROC is 0.804 (0.797-0.810). This measures agreement with NLP-mined
  labels, not clinical accuracy.
- **Hernia:** the test AUROC interval is wide (0.808-0.922), and val has only 10 positives. Too
  few for a sensitivity-based threshold, so the viewer shows the score only.
- **Sex and age:** no evidence of a sex or age gap on test, because the intervals overlap.
  This does not show that there is no gap.
  - Sex: F 0.802 (0.791-0.811), M 0.804 (0.796-0.812).
  - Age bands: compared over the 8 labels with at least 10 positives in every band.
  - On val, only 1 label (Effusion) has at least 10 positives in every age band, so the val
    age comparison is not reportable.
- **Thresholds that did not work on test:**
  - For Infiltration, Pneumonia and Nodule, the threshold reaching val sensitivity >= 0.80 gives
    test specificity below 0.40: 0.234, 0.271 and 0.363. No single threshold is usable for
    these labels. The viewer uses the highest-F1 threshold and flags them as low reliability,
    as it does for Atelectasis, Pleural_Thickening and Consolidation.
  - For Cardiomegaly, the val sensitivity-0.80 threshold gives test sensitivity 0.749: the
    threshold did not transfer from val to test.
- **Scores:** the scores are not calibrated probabilities. A threshold is a cut-off on the
  model's score; a score of 0.3 does not mean a 30% chance of the finding. Viewer thresholds:
  `configs/viewer_thresholds.yaml`.

### Segmentation

**For education only, not for clinical use.** All numbers come from the MONAI 3D U-Net
(checkpoint epoch 180) on the labelled cases of Medical Segmentation Decathlon Task09_Spleen. The
cases were split by case: 29 train, 6 val, 6 test. Source: [results_seg.md](results_seg.md) and
`reports/segmentation_metrics.json`.

- **Small test set:** test mean Dice is 0.9176 over only 6 cases, with a range of 0.826-0.958.
  With n = 6, no confidence interval is given, and one case changes the mean by a lot.
- **Model choice on val:** the reported checkpoint is the best of 20 (one every 10 epochs, out of
  200), picked by val mean Dice on 6 cases. The val mean of 0.9195 is therefore optimistic. The
  test cases were not used for the choice.
