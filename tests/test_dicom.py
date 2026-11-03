"""DICOM conversion round trips: written with pydicom, read back with pydicom, compared."""

import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from pydicom import dcmread
from pydicom.uid import CTImageStorage, DigitalXRayImageStorageForPresentation

from src.dicom import convert

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "nih"
REAL_SPLEEN = REPO_ROOT / "data" / "raw" / "Task09_Spleen" / "imagesTr" / "spleen_2.nii.gz"


def png_bytes(pixels: np.ndarray, mode: str = "L") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


def gradient(rows: int = 30, columns: int = 40) -> np.ndarray:
    # Different values along rows and columns, so a transpose would show.
    return (np.add.outer(np.arange(rows) * 5, np.arange(columns) * 2) % 256).astype(np.uint8)


def test_dx_round_trip_keeps_pixels_spacing_and_patient_id(tmp_path: Path) -> None:
    pixels = gradient()
    ds = convert.png_to_dx(png_bytes(pixels), "1335", "F", 58, "PA", pixel_spacing=(0.2, 0.25))
    convert.save_new(ds, tmp_path / "x.dcm")
    read, spacing, back = convert.read_dx(tmp_path / "x.dcm")
    assert read.shape == pixels.shape and np.array_equal(read, pixels)
    assert spacing == (0.2, 0.25)
    assert back.Modality == "DX" and back.SOPClassUID == DigitalXRayImageStorageForPresentation
    assert back.PhotometricInterpretation == "MONOCHROME2"
    assert back.PatientID == "1335" and back.PatientAge == "058Y" and back.PatientSex == "F"
    assert back.file_meta.MediaStorageSOPInstanceUID == back.SOPInstanceUID


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_dx_from_colour_png_is_grayscale(mode: str) -> None:
    ds = convert.png_to_dx(png_bytes(gradient(), mode), "1")
    assert ds.SamplesPerPixel == 1 and ds.BitsAllocated == 8


def test_dx_uids_are_new_and_derived_from_the_source() -> None:
    png = png_bytes(gradient())
    first = convert.png_to_dx(png, "1", source="00000001_000.png")
    again = convert.png_to_dx(png, "1", source="00000001_000.png")
    other = convert.png_to_dx(png, "1", source="00000001_001.png")
    names = ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID")
    # Converting the same image again gives the same UIDs, so Orthanc does not duplicate it.
    assert all(getattr(first, name) == getattr(again, name) for name in names)
    assert all(getattr(first, name) != getattr(other, name) for name in names)
    assert len({str(getattr(first, name)) for name in names}) == 3
    # New UIDs under pydicom's root, not taken from any source.
    assert all(str(getattr(first, name)).startswith("1.2.826.0.1.3680043.8.498.") for name in names)
    assert all(len(str(getattr(first, name))) <= 64 for name in names)


def test_dx_carries_no_names_or_dates() -> None:
    ds = convert.png_to_dx(png_bytes(gradient()), "1335", age=148)
    assert str(ds.PatientName) == "ANON^1335"
    assert ds.PatientBirthDate == "" and ds.StudyDate == ""
    assert ds.PatientAge == ""  # 148 is an NIH recording error, not copied
    assert ds.PatientIdentityRemoved == "YES"


def test_nih_spacing_accounts_for_the_resize() -> None:
    entry = convert.read_nih_entry(FIXTURES / "Data_Entry_2017.csv", "00000001_000.png")
    assert entry.patient == "1"
    row, column = convert.nih_pixel_spacing(entry, (1024, 1024))
    width, height = entry.original_size
    assert column == pytest.approx(entry.original_spacing[0] * width / 1024)
    assert row == pytest.approx(entry.original_spacing[1] * height / 1024)


def test_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "x.dcm"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        convert.save_new(convert.png_to_dx(png_bytes(gradient()), "1"), path)
    assert path.read_bytes() == b"keep"


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )


AFFINES = {
    "axis-aligned RAS": np.diag([0.7, 0.8, 2.5, 1.0]),
    "flipped x and y (LPS-like)": np.diag([-0.79, -0.79, 5.0, 1.0]),
    "oblique": np.vstack(
        [np.c_[rotation_z(17) @ np.diag([0.6, 0.9, 3.0]), [10, -20, 30]], [0, 0, 0, 1]]
    ),
}


@pytest.mark.parametrize("name", list(AFFINES))
def test_ct_round_trip_keeps_hu_spacing_and_orientation(tmp_path: Path, name: str) -> None:
    affine = AFFINES[name].copy()
    affine[:3, 3] += [-120.5, 80.25, -300.0]
    rng = np.random.default_rng(0)
    volume = rng.integers(-1024, 2000, size=(12, 9, 7)).astype(np.int16)
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "ct.nii.gz")

    paths = convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", tmp_path / "dcm", "spleen_x")
    print(f"slices written {len(paths)}, NIfTI slices {volume.shape[2]}")
    assert len(paths) == volume.shape[2] > 0
    # Read back in shuffled order: the geometry tags alone must give the slice order.
    back, back_affine = convert.read_ct_series(paths[::-1][1:] + paths[-1:])
    assert back.shape == volume.shape
    assert np.array_equal(back, volume.astype(np.float64))
    assert np.allclose(back_affine, affine, atol=1e-5)
    assert nib.aff2axcodes(back_affine) == nib.aff2axcodes(affine)

    first = dcmread(paths[0])
    assert first.Modality == "CT" and first.SOPClassUID == CTImageStorage
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    assert [float(v) for v in first.PixelSpacing] == pytest.approx([zooms[1], zooms[0]])
    assert float(first.SliceThickness) == pytest.approx(zooms[2])
    assert first.Rows == volume.shape[1] and first.Columns == volume.shape[0]
    uids = {(str(dcmread(p).SeriesInstanceUID), str(dcmread(p).FrameOfReferenceUID)) for p in paths}
    assert len(uids) == 1
    assert len({str(dcmread(p).SOPInstanceUID) for p in paths}) == len(paths)
    again = convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", tmp_path / "again", "spleen_x")
    assert [str(dcmread(p).SOPInstanceUID) for p in again] == [
        str(dcmread(p).SOPInstanceUID) for p in paths
    ]


def test_ct_with_fractional_values_uses_a_rescale_slope(tmp_path: Path) -> None:
    volume = np.linspace(-1000.25, 1500.75, 6 * 5 * 4).reshape(6, 5, 4)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "ct.nii.gz")
    paths = convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", tmp_path / "dcm", "x")
    back, _ = convert.read_ct_series(paths)
    slope = float(dcmread(paths[0]).RescaleSlope)
    assert slope != 1
    assert np.allclose(back, volume, atol=slope / 2 + 1e-9)


def test_ct_series_is_not_written_into_a_non_empty_folder(tmp_path: Path) -> None:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 2), np.int16), np.eye(4)), tmp_path / "ct.nii.gz")
    out = tmp_path / "dcm"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", out, "x")


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_SPLEEN.exists(), reason="MSD Task09_Spleen not downloaded")
def test_real_spleen_volume_round_trip(tmp_path: Path) -> None:
    image = nib.load(REAL_SPLEEN)
    volume = image.get_fdata()
    paths = convert.nifti_to_ct_series(REAL_SPLEEN, tmp_path / "dcm", "spleen_2")
    back, affine = convert.read_ct_series(paths)
    slope = float(dcmread(paths[0]).RescaleSlope)
    print(
        f"spleen_2: {volume.shape} voxels, zooms {image.header.get_zooms()}, "
        f"{len(paths)} slices, slope {slope}, max HU error {np.abs(back - volume).max()}"
    )
    assert back.shape == volume.shape
    assert np.allclose(back, volume, atol=slope / 2 + 1e-6)
    assert np.allclose(affine, image.affine, atol=1e-4)
