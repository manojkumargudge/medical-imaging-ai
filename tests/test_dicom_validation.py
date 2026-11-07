"""The DX and CT files src/dicom/convert.py writes carry what their IODs require.

dciodvfy (dicom3tools, David Clunie) reported missing Type 1/2 attributes in the generated DX
(For Presentation) and CT images; these tests keep the fixes. If dciodvfy is installed and
DCIODVFY points to it, the generated files must also pass it with no errors (warnings are
listed in README "Known limitations"). For education only, not for clinical use.
"""

import io
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset

from src.dicom import convert
from src.dicom.render import to_png

DCIODVFY = os.environ.get("DCIODVFY", "")


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    pixels = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def nih_dx() -> Dataset:
    return convert.png_to_dx(png_bytes(), "116", "M", 50, "PA", (0.143, 0.143), "a.png")


def ct_slices(tmp_path: Path) -> list[Path]:
    volume = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3) - 50
    nifti = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([0.8, 0.8, 2.5, 1.0])), nifti)
    return convert.nifti_to_ct_series(nifti, tmp_path / "ct", "spleen_x", "MSD Task09 spleen_x")


def test_nih_dx_has_the_dx_image_module() -> None:
    ds = nih_dx()
    region = ds.AnatomicRegionSequence[0]
    assert (region.CodeValue, region.CodingSchemeDesignator, region.CodeMeaning) == (
        "51185008",
        "SCT",
        "Chest",
    )
    assert ds.ImageLaterality == "U" and list(ds.PatientOrientation) == ["L", "F"]
    assert (ds.PixelIntensityRelationship, ds.PixelIntensityRelationshipSign) == ("LOG", -1)
    assert (ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType) == (0, 1, "US")
    assert ds.PresentationLUTShape == "IDENTITY" and ds.LossyImageCompression == "00"
    assert (ds.WindowCenter, ds.WindowWidth) == (128, 256)
    for keyword in ("DetectorType", "PositionerType", "AcquisitionContextSequence"):
        assert keyword in ds  # Type 2: present, may be empty


def test_an_upload_without_a_view_gets_no_made_up_orientation() -> None:
    ds = convert.png_to_dx(png_bytes(), "UPLOAD-1", source="seed")
    assert ds.PatientOrientation == "" and "ImagerPixelSpacing" not in ds


def test_the_new_tags_do_not_change_what_the_model_sees() -> None:
    """8-bit MONOCHROME2 pixels still pass through to_png unchanged (window not applied)."""
    ds = nih_dx()
    with Image.open(io.BytesIO(to_png(ds))) as image:
        assert np.array_equal(np.asarray(image), ds.pixel_array)


def test_ct_has_the_general_series_and_ct_image_attributes(tmp_path: Path) -> None:
    ds = dcmread(ct_slices(tmp_path)[0])
    assert ds.BodyPartExamined == "ABDOMEN"
    assert "PatientPosition" in ds and "AcquisitionNumber" in ds


@pytest.mark.skipif(not DCIODVFY, reason="set DCIODVFY to the dciodvfy executable")
def test_dciodvfy_finds_no_errors(tmp_path: Path) -> None:
    dx = tmp_path / "dx.dcm"
    convert.save_new(nih_dx(), dx)
    for path in [dx, *ct_slices(tmp_path)]:
        run = subprocess.run([DCIODVFY, str(path)], capture_output=True, text=True, check=False)
        lines = (run.stdout + run.stderr).splitlines()
        errors = [line for line in lines if line.startswith("Error")]
        warnings = [line for line in lines if line.startswith("Warning")]
        print(f"{path.name}: {len(errors)} errors, {len(warnings)} warnings")
        assert lines and not errors, errors
