# Classifier results

**For education only, not for clinical use.**

**No results yet.** The classifier has not been trained: `notebooks/colab_train.ipynb` has not
been run. The code has been tested only on tiny synthetic images, and those tests give no
information about accuracy.

This file is replaced by `python -m src.models.evaluate` (run from the notebook) with:

- the AUROC per label on val and on test (official `test_list.txt`), with positive counts
- the val-selected thresholds applied to test
- the mean AUROC by sex and age band

All numbers will be measured on NIH ChestX-ray14, whose labels are NLP-mined from reports.
