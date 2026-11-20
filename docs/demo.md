**For education only. Not for clinical use.**

# Demo recording script (Deliverable 1)

A plan for an 8 to 10 minute screen recording of the demo stack: 8 segments of about a minute,
each with what to click and 2 to 3 lines to say. Every number is copied from a file in this
repository, named in brackets after it; check the file again if it has changed since. Numbers
are measured on the named dataset and split only. None is a clinical accuracy, so do not say
"accurate", "diagnoses" or "detects" in the narration: say "agrees with the labels" or "the
score is above the threshold".

Stack: the demo Compose project `aiml02` (`compose.yaml`), viewer at http://127.0.0.1:8080, API
docs at http://127.0.0.1:8000/docs ([api.md](api.md)).

Upload file for segment 4: `data/raw/real_dicom/covid19ar_dx.dcm` (TCIA COVID-19-AR,
COVID-19-AR-16424105, Philips DX, AP, MONOCHROME2, 2140 x 2051, 8.8 MB, PatientIdentityRemoved
YES, CC BY 4.0; README "Real scanner DICOMs"). It is gitignored; fetch it with
`python -m scripts.fetch_real_dicom`. Do not show the RG1 files: they have no formal license
and are for local use only ([reports/real_dicom_check.json](../reports/real_dicom_check.json)).
Put the TCIA citation from the README in the video description.

## Before recording

1. **Close heavy apps** (other Docker projects, IDE indexing, browser tabs you do not need):
   the worker runs on the CPU and its times vary run to run
   ([limitations.md](limitations.md), "Timing is CPU-only").
2. **Do not disturb:** Windows Focus / Do not disturb on, notifications off, phone silent.
3. **Nothing private on screen:** do not open `.env` (passwords) or Orthanc's own UI. The
   browser only needs the viewer (port 8080) and the API docs (port 8000).
4. **Start the demo stack only:** `docker compose up -d --wait` in the repository (project
   `aiml02`; the test stack `aiml02-test` stays stopped). All 5 services must be healthy.
5. **Check the counts** (they were these on 2026-11-17):
   ```powershell
   docker exec aiml02-results-db-1 psql -U aiml02 -d aiml02_results -tAc "select 'results', count(*) from inference_results union all select 'reviews', count(*) from reviews union all select 'uploads', count(*) from uploads union all select 'audit_log', count(*) from audit_log"
   ```
   Expected: 3 studies in the worklist (2 DX, 1 CT, all Done, source demo), results 3,
   reviews 0, uploads 0, audit_log 18.
6. **Only if the counts differ** (for example, a leftover study), run
   `python -m scripts.reset_demo`, with the worker stopped first, as its docstring says
   (README "Resetting the demo"). It asks you to type `wipe`. It refuses while any review exists.
   It adds 3 `inference_done` rows to audit_log every time, so after a reset expect
   audit_log = 21, not 18. Do not run it when the counts already match.
7. **Back up the results database** (writes a file to the gitignored `backups/`; deletes nothing):
   ```powershell
   docker exec aiml02-results-db-1 pg_dump -U aiml02 -d aiml02_results > backups/demo-before-recording.sql
   ```
   The recording adds rows that no script can remove (see "After recording").
8. **Rehearse the clicks once without saving anything:** no upload, no review. A rehearsal
   upload or review is permanent too.
9. Browser at 100% zoom, window 1920 x 1080, viewer worklist open; the upload file ready in
   File Explorer.

## Segments

### 1. Problem and notice (about 1 min)

**Click:** the worklist at http://127.0.0.1:8080; point at the banner.

**Say:**
- "The brief: a platform where X-ray or CT scans are uploaded, analysed by deep learning models,
  shown with explanation heatmaps, and reviewed by a doctor" ([brief.md](brief.md)).
- "This is for education only, not for clinical use. Every page says so, and so does the API"
  (banner "For education only. Not for clinical use.", README "Viewer"; `/api/health`
  notice, README "Viewer API").
- "It uses two public datasets, NIH ChestX-ray14 and the Medical Segmentation Decathlon spleen
  task. None of the numbers you will see is a clinical accuracy."

### 2. Data (about 1 min)

**Click:** [eda.md](eda.md) on GitHub (split sizes table), then `dvc.yaml`.

**Say:**
- "NIH ChestX-ray14 has 112,120 images of 30,805 patients. The split is by patient: train
  77,757 images, val 8,767, test 25,596, and no patient is in two splits" ([eda.md](eda.md),
  "Split sizes").
- "The official test split is different: 38.5% No Finding images against 58.5% in train, and
  9.15 images per patient against 3.08. The labels were mined from report text by NLP, not read
  by radiologists" ([eda.md](eda.md); [limitations.md](limitations.md), "Data").
- "For CT, the 41 labelled MSD spleen cases are split by case, 29 train, 6 val, 6 test. Both
  splits and the preprocessing are DVC stages, with the data on a DVC remote on Google Drive"
  ([eda.md](eda.md), spleen section; [brief-evidence.md](brief-evidence.md), Steps 1 and 2).

### 3. Models (about 1 min)

**Click:** [results_v2.md](results_v2.md) (AUROC table), [results_seg.md](results_seg.md), then
the GitHub release page
[v1.0-models](https://github.com/manojkumargudge/medical-imaging-ai/releases/tag/v1.0-models).

**Say:**
- "The classifier is DenseNet121 with ImageNet weights, 14 labels, checkpoint epoch 3 of 5.
  Mean AUROC is 0.827 on val and 0.804 on the official test split, with a 95% bootstrap
  interval of 0.797 to 0.810" ([results_v2.md](results_v2.md); the interval is on test only,
  1000 patient-level resamples).
- "The segmentation model is a MONAI 3D U-Net, epoch 180 of 200. Mean spleen Dice is 0.9195 on
  6 val cases and 0.9176 on 6 test cases" ([results_seg.md](results_seg.md)).
- "Both were trained on Colab GPUs (T4 runtime) and published as the release v1.0-models. The
  fetch script checks their sha256, 1a35108e... for the classifier and 8839d537... for the
  U-Net, and every result row records the hash of the checkpoint that made it" (notebooks
  `colab_train.ipynb` and `colab_seg.ipynb`, `gpuType: T4`; README "Inference worker", full
  hashes).

### 4. Upload to result (about 1 min)

**Click:** Upload in the header. Point at the warning "Trained on NIH ChestX-ray14 frontal chest
X-rays. Scores for other images have no meaning." Choose `covid19ar_dx.dcm` ("Looks like a
DICOM file."), type a name in "Your name", click Upload. The worklist opens and the new
study shows "Processing", then "Done".

**Expected time:** about 10 to 20 s from Upload to Done. Orthanc marks a study stable after
5 s (`ORTHANC__STABLE_AGE`, `docker-compose.base.yml`), the worker polls every 5 s
(`--interval`, `src/service/worker.py`), and this file took 6.0 s in the worker on the test
stack ([reports/real_dicom_check.json](../reports/real_dicom_check.json)). The worklist
refreshes every 3 s while a study is processing (README "Viewer").

**Say:**
- "The upload is checked by content, not by name: DICOM, modality DX or CR, de-identified, at
  most 20 MB. The API stores it in Orthanc with new UIDs" (README "Viewer API").
- "The worker polls Orthanc's change log, runs the classifier and Grad-CAM on DX, and writes
  the result to Postgres" (README "Inference worker").
- "This is a real scanner file from TCIA, not an NIH image, so its source is upload and the
  viewer warns that the scores have no meaning."

### 5. DX view, review and audit log (about 1.5 min)

**Click:** open DX 00000116_000 from the worklist. Grad-CAM overlay on; label Cardiomegaly;
move Opacity. Scroll the scores table to Cardiomegaly, then to a Low reliability label (for
example Infiltration) and to Hernia ("Score only (no threshold)"). In the review panel:
Approve, your name, Save review. Then open the uploaded study from the worklist (study name
"Upload <id>", source upload). **Before Reject, confirm the header:** it must read "DX study,
patient COVID-19-AR-16424105" (for the retake file `covid19ar_cr.dcm`: "CR study, patient
COVID-19-AR-16434356"), with the model-scope warning under it. A demo study shows an NIH
patient number and no warning; if you see that, go back to the worklist. On a take on
2026-11-20, a reject meant for the upload was saved on a demo DX study, and a saved review
cannot be undone. Then Reject, your name, and the comment
"Out of distribution: TCIA DX, model trained on NIH only" (for `covid19ar_cr.dcm`:
"Out of distribution: TCIA CR, model trained on NIH only"), Save review. Finally, Audit log
in the header.

**Say:**
- "Cardiomegaly scores 0.734 here against a threshold of 0.0248, so it is above. The map sits
  over the heart" ([figures/gradcam/README.md](figures/gradcam/README.md); the worker's scores
  equal the Grad-CAM JSON within 1.19e-07, [brief-evidence.md](brief-evidence.md), Step 5).
- "Scores are not calibrated probabilities. Six labels use a highest-F1 threshold and are
  flagged low reliability; Hernia has 10 val positives, too few for a threshold, so it is
  score only" (`configs/viewer_thresholds.yaml`; [limitations.md](limitations.md),
  "Classifier").
- "A reject needs a comment. Each review is saved with an audit row in one transaction, and
  the audit log is append-only: the database refuses updates and deletes" (README "Viewer
  API" and "Resetting the demo").

### 6. CT view (about 1 min)

**Click:** open the CT study spleen_14 from the worklist. Jump to spleen, move the slice
slider, Spleen mask off and on, move Opacity. Point at the A/P/R/L edge labels and the spleen
volume.

**Say:**
- "spleen_14 is a 54-slice CT series converted from MSD. The U-Net's mask is drawn on each
  slice; the stored spleen volume is 440.8 mL" ([brief-evidence.md](brief-evidence.md), Step 5,
  for the 54 slices; README "Viewer API", CT paragraph, for the volume).
- "spleen_14 is a test case. Its Dice is 0.9579 on the 1.5 mm model grid"
  ([results_seg.md](results_seg.md)).
- "On the DICOM grid the viewer shows, the Dice is 0.9673. That is a check that the mask lines
  up after resampling, not a result" ([limitations.md](limitations.md), "Segmentation").

### 7. Engineering (about 1 min)

**Click:** the GitHub Actions run of the latest PR (jobs test, integration, frontend), then
http://127.0.0.1:8000/docs.

**Say:**
- "CI runs lint, strict types and tests on every PR. As of PR #31: 346 pytest tests passed and
  39 skipped, 24 integration tests against a separate Compose test stack, and 36 Vitest tests
  for the viewer" ([brief-evidence.md](brief-evidence.md), Deliverable 4).
- "A fresh clone, following the README only, passed every step on 2026-11-16: all 5 services
  healthy, the 3 demo studies Done, and pytest 348 passed, 37 skipped, 0 failed"
  ([brief-evidence.md](brief-evidence.md), Deliverable 2).
- "One command starts the stack. Every service has a healthcheck, every port is bound to
  127.0.0.1, and the API documents itself here at /docs" ([brief-evidence.md](brief-evidence.md),
  Step 7; [api.md](api.md)).

### 8. Limits and bias (about 1 min)

**Click:** [limitations.md](limitations.md) on GitHub.

**Say:**
- "For Infiltration, Pneumonia and Nodule, the threshold that reached 0.80 sensitivity on val
  gave test specificity of only 0.234, 0.271 and 0.363. The Cardiomegaly threshold did not
  transfer either: test sensitivity 0.749. And the scores are not calibrated probabilities"
  ([limitations.md](limitations.md), "Classifier"; [results_v2.md](results_v2.md)).
- "Grad-CAM: a pacemaker was visible in 4 of the first 6 demo images, including both
  Cardiomegaly images. The map sat on the heart, but a device confound cannot be ruled out.
  The segmentation test set is only 6 cases, Dice 0.826 to 0.958, so there are no confidence
  intervals" ([limitations.md](limitations.md), "Grad-CAM" and "Segmentation").
- "Timing is CPU-only, on one laptop: median 6.9 and 7.6 s per DX study and 13.7 s for the CT,
  66.6 s for the first CT run in a new process. Again: education only, not for clinical use"
  ([reports/worker_timing.json](../reports/worker_timing.json);
  [limitations.md](limitations.md), "System").

## After recording

1. **Check the counts** with the command from "Before recording", step 5. Expected after one
   clean take: 4 studies in the worklist, results 4, reviews 2, uploads 1, audit_log 22 (18 +
   1 upload + 1 inference_done + 2 reviews).
2. **Stop the stack** (volumes kept): `docker compose -p aiml02 stop`.
3. **The demo cannot be scripted back to 3 / 3 / 0 / 0 / 18:**
   - `reset_demo.py` refuses while any review exists, and deletes nothing in that case.
   - audit_log is append-only, so its count never goes down.
   - Removing the 2 reviews would be a database delete. Decide that separately, with the
     backup from "Before recording", step 7, at hand. Nothing in this document does it.

   The simplest choice is to keep the recorded state as the new demo baseline and write its
   counts into the state file.
4. **Retakes:** a take that saved an upload or a review leaves it in place. Uploading
   `covid19ar_dx.dcm` again while its study is still in Orthanc adds nothing; the page says
   "This file was uploaded before" with a link to the earlier study, so "Processing" is not
   shown again (README "Viewer API"). For a retake of segment 4, use
   `covid19ar_cr.dcm` instead (CC BY 4.0, PatientIdentityRemoved YES). It is landscape and is
   squashed to 224 x 224 for the model (README "Known limitations").
