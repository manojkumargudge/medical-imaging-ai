# data/

Local data only. Everything here is gitignored except this file.

- `raw/`: Kaggle metadata files (see the README "Data" section). No images.
- `splits/`: `train.csv`, `val.csv`, `test.csv` from `dvc repro split`.
- `processed/nih_224/`: 224px grayscale PNG shards, `shards.csv`, `images.csv` and
  `preprocess.json` from `dvc repro preprocess` (on Colab), or from `dvc pull`.
- `manifests/nih_224.csv`: one row per image (split, labels, age, sex, shard, sha256) from the same
  stage.
- `cache/`: temporary files of the preprocess stage.
- `raw/real_dicom/`: four real chest X-ray DICOMs from `python -m scripts.fetch_real_dicom`
  (sha256-checked; see the README "Real scanner DICOMs"). Never committed.
  - `covid19ar_cr.dcm`, `covid19ar_dx.dcm`: TCIA COVID-19-AR, **CC BY 4.0**. Cite Desai S, et
    al. (2020), The Cancer Imaging Archive, https://doi.org/10.7937/tcia.2020.py71-5978.
  - `RG1_J2KR.dcm`, `RG1_J2KI.dcm`: NEMA DICOM WG04 compression test images, mirrored by
    pydicom-data. NEMA gives **no formal license** for them, so they are used locally only and
    not redistributed.
