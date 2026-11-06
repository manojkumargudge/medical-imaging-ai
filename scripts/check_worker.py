"""Real checks of the worker's stored results against the published evaluation outputs.

- DX: for each demo PNG uploaded by scripts/upload_demo.py, the worker's score for the label in
  docs/figures/gradcam/<image>_<label>.json must equal the Grad-CAM CLI score. Tolerance 1e-6:
  both runs use the same weights and preprocessing on CPU, so only float differences between
  torch builds (laptop wheel vs the worker image) are allowed. The overlay pixels are compared too.
  For every label, the raw Grad-CAM PNG must exist, and re-blending it over the 224px image must
  give the stored overlay within 1 grey level (8-bit rounding of the map).
- CT: Dice between the worker's mask and the reference label must match the Dice in
  reports/segmentation_metrics.json. The label goes through the evaluation transforms (RAS,
  1.5 mm, body crop), as in the evaluation, so the two grids must agree. Tolerance 0.005: the
  evaluation ran on a GPU with float16 autocast and read the NIfTI directly; the worker runs
  float32 on CPU on a volume that went through DICOM (int16 HU, affine within 1e-4 mm).

Reads the results database; writes nothing. For education only, not for clinical use.

Run: python -m scripts.check_worker
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from pydicom import dcmread

from scripts.upload_demo import DATA_ENTRY, DEFAULT_PNGS, DEFAULT_SPLEEN_CASE, dx_from_png
from src.data.preprocess import read_params
from src.dicom import convert
from src.explain import gradcam
from src.models.segmentation import build_transforms, load_seg_config
from src.service.analyzers import to_png
from src.service.orthanc import load_env
from src.service.store import PostgresStore

GRADCAM = Path("docs/figures/gradcam")
SEG_METRICS = Path("reports/segmentation_metrics.json")
TASK = Path("data/raw/Task09_Spleen")
DX_TOLERANCE = 1e-6
DICE_TOLERANCE = 0.005
CAM_TOLERANCE = 1


def dice(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 1.0


def check_raw_cams(labels: dict[str, Any], results_dir: Path, ds: Any, name: str) -> bool:
    seen, _ = gradcam.preprocess(
        to_png(ds), read_params(Path("params.yaml")), Path("configs/classifier.yaml")
    )
    worst, missing = 0, [label for label, entry in labels.items() if "cam" not in entry]
    for entry in labels.values():
        if "cam" not in entry:
            continue
        with (
            Image.open(results_dir / entry["cam"]) as cam,
            Image.open(results_dir / entry["overlay"]) as stored,
        ):
            reblended = gradcam.overlay(seen, gradcam.cam_from_image(cam))
            difference = np.abs(np.asarray(reblended, int) - np.asarray(stored, int)).max()
        worst = max(worst, int(difference))
    ok = bool(labels) and not missing and worst <= CAM_TOLERANCE
    print(
        f"DX {name} raw CAMs: {len(labels) - len(missing)} of {len(labels)} labels; re-blended "
        f"vs stored overlay max pixel diff {worst} (tolerance {CAM_TOLERANCE}) -> "
        f"{'OK' if ok else 'FAIL'}"
    )
    return ok


def check_dx(rows: dict[str, dict[str, Any]], results_dir: Path, png: Path) -> bool:
    ds = dx_from_png(png, DATA_ENTRY)
    study_uid = str(ds.StudyInstanceUID)
    row = rows[study_uid]
    [instance] = row["results"]["instances"]
    [cli_path] = sorted(GRADCAM.glob(f"{png.stem}_*.json"))
    cli = json.loads(cli_path.read_text(encoding="utf-8"))
    worker = instance["labels"][cli["label"]]
    difference = abs(worker["score"] - cli["score"])
    with (
        Image.open(results_dir / worker["overlay"]) as a,
        Image.open(GRADCAM / cli["overlay"]) as b,
    ):
        pixels = np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).max()
    ok = difference <= DX_TOLERANCE and worker["threshold"] == cli["threshold"]
    print(
        f"DX {png.stem} {cli['label']}: worker {worker['score']:.9f}, Grad-CAM CLI "
        f"{cli['score']:.9f}, |diff| {difference:.2e} (tolerance {DX_TOLERANCE:g}); threshold "
        f"{worker['threshold']} vs {cli['threshold']}; overlay max pixel diff {pixels} -> "
        f"{'OK' if ok else 'FAIL'}"
    )
    return bool(ok) and check_raw_cams(instance["labels"], results_dir, ds, png.stem)


def check_ct(rows: dict[str, dict[str, Any]], results_dir: Path, case: str) -> bool:
    with tempfile.TemporaryDirectory() as folder:
        paths = convert.nifti_to_ct_series(TASK / "imagesTr" / f"{case}.nii.gz", Path(folder), case)
        study_uid = str(dcmread(paths[0], stop_before_pixels=True).StudyInstanceUID)
    [series] = rows[study_uid]["results"]["series"]
    mask_image = nib.load(results_dir / series["mask"])
    mask = np.asarray(mask_image.dataobj)

    config = load_seg_config(Path("configs/segmentation.yaml"))
    reference = build_transforms(config, train=False)(
        {
            "image": str(TASK / "imagesTr" / f"{case}.nii.gz"),
            "label": str(TASK / "labelsTr" / f"{case}.nii.gz"),
        }
    )["label"]
    label = reference[0].numpy()
    reported = {
        row["case"]: row["dice"]
        for split in json.loads(SEG_METRICS.read_text(encoding="utf-8"))["splits"].values()
        for row in split["cases"]
    }[case]
    same_grid = mask.shape == label.shape and np.allclose(
        mask_image.affine, np.asarray(reference.affine), atol=1e-3
    )
    if not same_grid:
        print(f"CT {case}: grids differ, mask {mask.shape} vs label {label.shape} -> FAIL")
        return False
    measured = dice(mask, label)
    difference = abs(measured - reported)
    ok = difference <= DICE_TOLERANCE
    print(
        f"CT {case}: worker Dice {measured:.4f}, reported {reported:.4f}, |diff| "
        f"{difference:.4f} (tolerance {DICE_TOLERANCE}); grid {mask.shape}, mask voxels "
        f"{int(mask.sum())}, label voxels {int(label.sum())} -> {'OK' if ok else 'FAIL'}"
    )
    return bool(ok)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/worker"))
    parser.add_argument("--spleen-case", default=DEFAULT_SPLEEN_CASE)
    args = parser.parse_args(argv)
    load_env()
    port = os.environ.get("RESULTS_DB_PORT", "5434")
    url = f"postgresql://aiml02:{os.environ['RESULTS_DB_PASSWORD']}@127.0.0.1:{port}/aiml02_results"
    store = PostgresStore(url)
    rows = {row["study_uid"]: row for row in store.rows() if row["status"] == "ok"}
    store.close()
    print(f"{len(rows)} ok rows in the results database")
    checks = [check_dx(rows, args.results_dir, png) for png in DEFAULT_PNGS]
    checks.append(check_ct(rows, args.results_dir, args.spleen_case))
    if not all(checks):
        raise SystemExit("worker check failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
