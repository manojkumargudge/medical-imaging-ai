# data/

Local data only. Everything here is gitignored except this file.

- `raw/`: Kaggle metadata files (see the README "Data" section). No images.
- `splits/`: `train.csv`, `val.csv`, `test.csv` from `dvc repro split`.
- `processed/nih_224/`: 224px grayscale PNG shards, `shards.csv`, `images.csv` and
  `preprocess.json` from `dvc repro preprocess` (on Colab), or from `dvc pull`.
- `manifests/nih_224.csv`: one row per image (split, labels, age, sex, shard, sha256) from the same
  stage.
- `cache/`: temporary files of the preprocess stage.
