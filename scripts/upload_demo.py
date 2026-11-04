"""Upload demo studies to Orthanc: NIH PNGs as DX and one MSD spleen case as a CT series.

The conversion is src/dicom/convert.py (no names or dates; new UIDs derived from the source, so a
rerun gives the same UIDs and Orthanc reports "AlreadyStored"). The DICOM files are written to a
temporary folder and removed after the upload. For education only, not for clinical use.

Run (Orthanc up, .env filled in):
    python -m scripts.upload_demo
    python -m scripts.upload_demo --png data/raw/demo/images_001/images/00000116_000.png \
        --spleen-case spleen_14
"""

from __future__ import annotations

import argparse
import io
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset

from src.dicom import convert
from src.service.orthanc import Orthanc, load_env

DEMO = Path("data/raw/demo/images_001/images")
DEFAULT_PNGS = (DEMO / "00000116_000.png", DEMO / "00000013_021.png")
DATA_ENTRY = Path("data/raw/Data_Entry_2017.csv")
SPLEEN_IMAGES = Path("data/raw/Task09_Spleen/imagesTr")
DEFAULT_SPLEEN_CASE = "spleen_14"  # a test case in docs/results_seg.md


def dx_from_png(png_path: Path, data_entry: Path) -> Dataset:
    entry = convert.read_nih_entry(data_entry, png_path.name)
    png = png_path.read_bytes()
    with Image.open(io.BytesIO(png)) as image:
        spacing = convert.nih_pixel_spacing(entry, image.size)
    return convert.png_to_dx(
        png, entry.patient, entry.sex, entry.age, entry.view, spacing, entry.image
    )


def dicom_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def upload_dx(orthanc: Orthanc, ds: Dataset) -> dict[str, Any]:
    result = orthanc.upload(dicom_bytes(ds))
    return {"study_uid": str(ds.StudyInstanceUID), "orthanc": result, "instances": 1}


def upload_ct(orthanc: Orthanc, nifti: Path, case: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as folder:
        paths = convert.nifti_to_ct_series(nifti, Path(folder) / "ct", case, f"MSD Task09 {case}")
        results = [orthanc.upload(path.read_bytes()) for path in paths]
        study_uid = str(dcmread(paths[0], stop_before_pixels=True).StudyInstanceUID)
    return {"study_uid": study_uid, "orthanc": results[0], "instances": len(results)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, nargs="*", default=list(DEFAULT_PNGS))
    parser.add_argument("--data-entry", type=Path, default=DATA_ENTRY)
    parser.add_argument("--spleen-case", default=DEFAULT_SPLEEN_CASE)
    parser.add_argument("--spleen-dir", type=Path, default=SPLEEN_IMAGES)
    args = parser.parse_args(argv)
    load_env()
    orthanc = Orthanc.from_env()

    uploads = [
        (path.name, upload_dx(orthanc, dx_from_png(path, args.data_entry))) for path in args.png
    ]
    nifti = args.spleen_dir / f"{args.spleen_case}.nii.gz"
    uploads.append((args.spleen_case, upload_ct(orthanc, nifti, args.spleen_case)))
    for source, upload in uploads:
        found = orthanc.find_study(upload["study_uid"])
        print(
            f"{source}: {upload['instances']} instance(s), {upload['orthanc']['Status']}, "
            f"study {upload['study_uid']} -> Orthanc {found}"
        )
        if not found:
            raise RuntimeError(f"{source}: study {upload['study_uid']} not found after upload")


if __name__ == "__main__":
    main()
