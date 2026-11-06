"""CT slice order, the DICOM grid, mask resampling, PNGs and the .npz cache (no network)."""

import io
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset

from src.api import ct, studies
from src.dicom import convert
from src.service.stack import DEMO_NIH_IMAGES, DEMO_SPLEEN_CASE, demo_study_uids


def series(
    tmp_path: Path, affine: np.ndarray, shape: tuple[int, int, int] = (20, 16, 7)
) -> tuple[list[Dataset], np.ndarray]:
    """A CT series from a NIfTI volume whose voxel values encode (i, j, k)."""
    i, j, k = np.indices(shape)
    volume = (i + 100 * j + 1000 * k).astype(np.int16)
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "ct.nii.gz")
    paths = convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", tmp_path / "dcm", "case")
    return [dcmread(p) for p in paths], volume


def as_orthanc(datasets: Sequence[Dataset], shuffle_numbers: bool = True) -> list[dict[str, Any]]:
    """Orthanc-like instance entries (positions as "a\\b\\c" strings), shuffled, with
    InstanceNumbers that do not follow the slice order."""
    numbers = list(range(1, len(datasets) + 1))
    if shuffle_numbers:
        random.Random(1).shuffle(numbers)
    entries = [
        {
            "ID": str(ds.SOPInstanceUID),
            "MainDicomTags": {
                "ImagePositionPatient": "\\".join(str(v) for v in ds.ImagePositionPatient),
                "ImageOrientationPatient": "\\".join(str(v) for v in ds.ImageOrientationPatient),
                "InstanceNumber": str(number),
            },
        }
        for ds, number in zip(datasets, numbers, strict=True)
    ]
    random.Random(2).shuffle(entries)
    return entries


@pytest.mark.parametrize(
    "affine",
    [
        np.diag([-0.8, -0.8, 2.5, 1.0]),  # RAS flipped in x and y, like many CT exports
        np.array(
            [[0.7, 0, 0, -50], [0, 0.7, 0, 20], [0, 0, -3.0, 90], [0, 0, 0, 1]]
        ),  # k goes down
    ],
)
def test_slices_follow_position_not_instance_number(tmp_path: Path, affine: np.ndarray) -> None:
    datasets, volume = series(tmp_path, affine)
    entries = as_orthanc(datasets)
    ordered = ct.slice_order(entries)
    by_uid = {str(ds.SOPInstanceUID): ds for ds in datasets}
    numbers = [int(e["MainDicomTags"]["InstanceNumber"]) for e in ordered]
    print(f"InstanceNumbers in slice order: {numbers}")
    assert numbers != sorted(numbers)  # the order did not come from InstanceNumber
    # In patient order, slice k holds the NIfTI's k (as read_ct_series reads it).
    _, read_affine = convert.read_ct_series(sorted(tmp_path.glob("dcm/*.dcm")))
    grid = ct.geometry(entries, datasets[0])
    assert np.allclose(grid.affine, read_affine, atol=1e-4)
    assert grid.shape == volume.shape
    expected_k = np.linalg.solve(affine, grid.affine)[:3, 2].round(6)
    for k, entry in enumerate(ordered):
        pixels = by_uid[entry["ID"]].pixel_array.T  # (i, j)
        source_k = k if expected_k[2] > 0 else volume.shape[2] - 1 - k
        assert np.array_equal(pixels, volume[:, :, source_k])


def test_resampling_a_flipped_and_permuted_mask_gives_it_back(tmp_path: Path) -> None:
    affine = np.diag([-0.8, -0.8, 2.5, 1.0])
    datasets, _ = series(tmp_path, affine)
    grid = ct.geometry(as_orthanc(datasets), datasets[0])
    rng = np.random.default_rng(0)
    label = (rng.random(grid.shape) > 0.7).astype(np.uint8)
    # The same mask stored with i flipped and i, j swapped: another grid, same positions.
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, grid.shape[0] - 1
    swap = np.eye(4)[[1, 0, 2, 3]]
    stored = np.transpose(label[::-1, :, :], (1, 0, 2))
    stored_affine = grid.affine @ flip @ swap
    back = ct.resample_nearest(stored, stored_affine, grid.shape, grid.affine)
    assert np.array_equal(back, label)


def test_a_ball_lands_where_it_is_in_mm(tmp_path: Path) -> None:
    datasets, _ = series(tmp_path, np.diag([-0.8, -0.8, 2.5, 1.0]), shape=(40, 36, 12))
    grid = ct.geometry(as_orthanc(datasets), datasets[0])
    centre_mm = grid.affine @ np.array([20, 15, 6, 1.0])
    # The mask in a 1.5 mm RAS grid with another origin, like the U-Net's space.
    mask_affine = np.diag([1.5, 1.5, 1.5, 1.0])
    mask_affine[:3, 3] = centre_mm[:3] - 15
    index = np.indices((21, 21, 21))
    ball = (((index - 10) * 1.5) ** 2).sum(axis=0) <= 6**2
    out = ct.resample_nearest(ball.astype(np.uint8), mask_affine, grid.shape, grid.affine)
    found = np.argwhere(out).mean(axis=0)
    found_mm = grid.affine @ np.array([*found, 1.0])
    print(f"ball centre {centre_mm[:3]}, found {found_mm[:3]}")
    assert np.allclose(found_mm[:3], centre_mm[:3], atol=1.5)
    volume_ml = out.sum() * grid.voxel_ml
    assert volume_ml == pytest.approx(4 / 3 * np.pi * 6**3 / 1000, rel=0.2)


def test_window_and_mask_pngs() -> None:
    ds = Dataset()
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    pixels = np.array([[1024 - 160, 1024 + 40], [1024 + 240, 0]], dtype=np.int16)
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = pixels.tobytes()
    ds.file_meta = convert._file_meta("1.2.840.10008.5.1.4.1.1.2", "1.2.3")
    with Image.open(io.BytesIO(ct.windowed_png(ds))) as image:
        assert np.asarray(image).tolist() == [[0, 128], [255, 0]]
    mask = np.zeros((3, 2), dtype=np.uint8)  # (i, j)
    mask[2, 0] = 1
    with Image.open(io.BytesIO(ct.mask_png(mask))) as image:
        rgba = np.asarray(image)
    assert rgba.shape == (2, 3, 4)  # rows (j) down, columns (i) across
    assert rgba[0, 2].tolist() == [*ct.MASK_RGB, 255] and rgba[..., 3].sum() == 255


def test_the_mask_is_resampled_once_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets, _ = series(tmp_path, np.diag([-0.8, -0.8, 2.5, 1.0]))
    grid = ct.geometry(as_orthanc(datasets), datasets[0])
    label = np.zeros(grid.shape, dtype=np.uint8)
    label[5:9, 4:7, 2:5] = 1
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(label, grid.affine), path)
    first = ct.CtCache(tmp_path / "cache").mask_on_grid("1.2", path, grid)
    files = list((tmp_path / "cache").glob("*.npz"))
    assert np.array_equal(first, label) and len(files) == 1

    def fail(*_: Any) -> None:
        raise AssertionError("resampled again")

    monkeypatch.setattr(ct, "resample_nearest", fail)
    again = ct.CtCache(tmp_path / "cache").mask_on_grid("1.2", path, grid)  # a new process
    assert np.array_equal(again, label)
    # A missing or read-only cache folder still works, in memory.
    monkeypatch.undo()
    blocked = tmp_path / "file-not-folder"
    blocked.write_text("x")
    assert np.array_equal(ct.CtCache(blocked).mask_on_grid("1.2", path, grid), label)


def test_source_is_demo_only_for_the_demo_studies() -> None:
    demo = demo_study_uids()
    dx = convert.png_to_dx(_png(), "116", source=DEMO_NIH_IMAGES[0]).StudyInstanceUID
    ct_uid = convert.uid_for("ct", DEMO_SPLEEN_CASE, "study")
    assert {str(dx), ct_uid} <= demo and len(demo) == 3
    assert studies.source("o1", str(dx), set(), demo) == "demo"
    assert studies.source("o2", "1.2.3", set(), demo) == "unknown"
    assert studies.source("o3", "1.2.4", {"o3"}, demo) == "upload"
    assert studies.source("o4", str(dx), {str(dx)}, demo) == "upload"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()
