**For education only. Not for clinical use.**

# Limits and bias

What this project's models and system cannot do, in plain terms. Numbers are measured on the
named dataset and split only; none is a clinical accuracy. Sources: [eda.md](eda.md),
[results_v2.md](results_v2.md) (classifier), [results_seg.md](results_seg.md) (segmentation),
[figures/gradcam/README.md](figures/gradcam/README.md) (Grad-CAM demo).

## Data

- **Noisy labels.** The NIH ChestX-ray14 labels were mined from report text by NLP, not read by
  radiologists. Every metric measures agreement with those mined labels.
- **One hospital.** All NIH images come from one hospital (the NIH Clinical Center). Other
  hospitals, scanners and populations were not seen.
- **Test differs from train.** The official test split has 38.5% No Finding images against
  58.5% in train, and 9.15 images per patient against 3.08. Test results are not directly
  comparable with val results.
- **Invalid ages.** 16 images have an age above 100, almost certainly a recording error. They
  are kept as recorded; age-band results leave them out.
- **Small segmentation data.** MSD Task09_Spleen has 41 labelled cases from one centre; the test
  split has 6 cases.

## Bias

- **Sex and age.** Mean test AUROC by sex: F 0.802 (0.791-0.811), M 0.804 (0.796-0.812). By age
  band (8 labels with enough positives): 0.785 to 0.795, with overlapping intervals. No gap was
  found, but overlapping intervals do not show that there is none.
- **Not checked:** race, scanner or hospital. The data has no such fields, so any gap there is
  unknown.

## Classifier

- **Scores are not calibrated probabilities.** A threshold is a cut-off on the model's score.
- **Low specificity.** For Infiltration, Pneumonia and Nodule, the threshold that reached val
  sensitivity >= 0.80 gave test specificity 0.234, 0.271 and 0.363. The viewer uses the
  highest-F1 threshold for them instead and marks them low reliability.
- **Cardiomegaly.** Its val sensitivity-0.80 threshold gave test sensitivity 0.749: the
  threshold did not transfer.
- **Hernia** has a score only, no threshold: val has 10 positives, too few to choose one.

## Grad-CAM

From 12 test images chosen by fixed rules: they show how maps look, not how often they are right.

- All 4 missed positives had maps outside the lungs (image edge, or below the heart).
- A pacemaker was visible in 4 of the first 6 images, including both Cardiomegaly images. The
  map sat on the heart, not the device, but a pacemaker and Cardiomegaly confound is possible.
- Maps may not be label-specific: one patient's Effusion and Pneumothorax maps covered nearly
  the same region (correlation 0.79).
- A map shows which regions raised the score, not where a finding is.

## Segmentation

- **Test Dice 0.9176** (range 0.826-0.958) on 6 test cases. With n = 6 there are no confidence
  intervals.
- The viewer's **DICOM-grid Dice of 0.9673** for spleen_14 is an orientation check of the
  viewer's resampling, not a result. It is one case, and that case is in the test split.

## CT orientation

The converted CT series keep ImageOrientationPatient `-1\0\0\0\-1\0` (valid DICOM). A viewer that
draws the stored pixels as they are, such as Orthanc's own, shows them rotated by 180 degrees.
This project's viewer reorients every axial series to anterior up and the patient's right on the
image's left, and labels the edges A, P, R, L.

## System

- **No login.** Reviewer and uploader names are typed in and not verified.
- **Timing is CPU-only** on one laptop (Intel i5-1235U) and varies run to run. Median and range
  of 5 warm runs per study: DX 00000116_000 6.9 s (5.5-21.8), DX 00000013_021 7.6 s (7.0-10.3),
  CT spleen_14 13.7 s (12.2-18.7). The first run in a new process is slower (CT 66.6 s). Source:
  `reports/worker_timing.json`.
- **CT upload is not supported:** a CT needs a full series, and the upload takes one file.
- **Large direct uploads.** A request sent straight to the API (port 8000) is received in full
  before the 20 MB check rejects it; through the frontend, nginx caps the body at 21 MB.
- **Orphan uploads.** If the database write fails after Orthanc stored an upload, the study stays
  in Orthanc without an uploads row and shows as source "unknown".
