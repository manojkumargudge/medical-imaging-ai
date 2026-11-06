"""The worker's real spleen_14 mask on the DICOM grid: volume and orientation checks.

Needs the MSD Task09_Spleen data (DVC, data/raw) and the worker's output for the demo CT
(results/worker); skipped otherwise (CI). Reads files only.
- Volume: the mask on the DICOM grid against the worker's stored spleen_ml (1.5 mm grid);
  nearest-neighbour resampling may differ by a few percent, the limit is 3%.
- Orientation: the resampled prediction against the ground-truth label of spleen_14 on the same
  DICOM grid. A flipped or shifted axis would drop Dice far below the 0.958 measured on the
  1.5 mm grid (reports/segmentation_metrics.json).
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom import dcmread
from test_ct import as_orthanc

from src.api import ct
from src.dicom import convert
from src.service.stack import DEMO_SPLEEN_CASE

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = REPO_ROOT / "data" / "raw" / "Task09_Spleen"
IMAGE = TASK / "imagesTr" / f"{DEMO_SPLEEN_CASE}.nii.gz"
LABEL = TASK / "labelsTr" / f"{DEMO_SPLEEN_CASE}.nii.gz"
STUDY_UID = convert.uid_for("ct", DEMO_SPLEEN_CASE, "study")
MASKS = sorted(
    (REPO_ROOT / "results" / "worker" / STUDY_UID).glob("monai-unet-spleen/*_mask.nii.gz")
)
SEG_METRICS = REPO_ROOT / "reports" / "segmentation_metrics.json"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not (IMAGE.exists() and LABEL.exists() and MASKS),
        reason="needs data/raw/Task09_Spleen and the worker's spleen_14 mask in results/worker",
    ),
]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def test_spleen_14_on_the_dicom_grid(tmp_path: Path) -> None:
    paths = convert.nifti_to_ct_series(IMAGE, tmp_path / "dcm", DEMO_SPLEEN_CASE)
    datasets = [dcmread(p, stop_before_pixels=True) for p in paths]
    grid = ct.geometry(as_orthanc(datasets), datasets[0])
    [mask_path] = MASKS
    mask_image = nib.load(mask_path)
    model_mask = np.asarray(mask_image.dataobj)
    stored_ml = float(model_mask.sum()) * float(np.prod(mask_image.header.get_zooms()[:3])) / 1000

    on_grid = ct.CtCache(None).mask_on_grid(STUDY_UID, mask_path, grid)
    grid_ml = float(on_grid.sum()) * grid.voxel_ml
    difference = (grid_ml - stored_ml) / stored_ml * 100

    label_image = nib.load(LABEL)
    label = ct.resample_nearest(
        np.asarray(label_image.dataobj), np.asarray(label_image.affine), grid.shape, grid.affine
    )
    reported = {
        row["case"]: row["dice"]
        for split in json.loads(SEG_METRICS.read_text(encoding="utf-8"))["splits"].values()
        for row in split["cases"]
    }[DEMO_SPLEEN_CASE]
    measured = dice(on_grid, label)
    slices = np.flatnonzero(on_grid.any(axis=(0, 1)))
    print(
        f"DICOM grid {grid.shape}, voxel {grid.voxel_ml * 1000:.3f} mm3; mask on slices "
        f"{slices[0]}-{slices[-1]}; spleen {grid_ml:.1f} mL on the DICOM grid vs stored "
        f"{stored_ml:.1f} mL ({difference:+.2f}%); Dice vs ground truth on the DICOM grid "
        f"{measured:.4f} (reported on the 1.5 mm grid {reported:.4f})"
    )
    assert abs(difference) <= 3
    # The label lies on the DICOM grid already (same NIfTI): resampling it changes nothing.
    assert np.array_equal(label, np.asarray(label_image.dataobj).astype(np.uint8))
    assert abs(measured - reported) <= 0.02
