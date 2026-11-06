"""CT slices and the spleen mask on the DICOM grid, for the viewer. For education only.

- Slices are ordered by ImagePositionPatient along the slice normal (the cross product of the
  ImageOrientationPatient row and column directions), never by InstanceNumber.
- The DICOM grid is (column i, row j, slice k), as src/dicom/convert.py `read_ct_series` builds
  it, with its RAS affine from the geometry tags alone.
- The worker stores the mask in the model's space (RAS, 1.5 mm, cropped to the body) with its own
  affine. It is put on the DICOM grid by nearest neighbour: each DICOM voxel takes the mask
  voxel at the same patient position (inverse mask affine x DICOM affine). Outside the mask's
  box is 0. The result is computed once per (mask file, geometry) and cached as .npz.
- Slice PNGs use the CT window 40/400 (HU -160 to 240).
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydicom.dataset import Dataset

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
WINDOW_CENTER, WINDOW_WIDTH = 40.0, 400.0
MASK_RGB = (0, 200, 83)


def _floats(value: Any) -> NDArray[np.float64]:
    """A DICOM multi-value as floats: Orthanc's "a\\b\\c" string, or a list."""
    if isinstance(value, str):
        value = value.split("\\")
    return np.array([float(v) for v in value], dtype=np.float64)


def slice_order(instances: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Orthanc instances (with MainDicomTags) in patient order along the slice normal."""
    if not instances:
        raise ValueError("no instances")
    orientation = _floats(instances[0]["MainDicomTags"]["ImageOrientationPatient"])
    normal = np.cross(orientation[:3], orientation[3:])
    keyed = [
        (float(_floats(i["MainDicomTags"]["ImagePositionPatient"]) @ normal), i) for i in instances
    ]
    return [instance for _, instance in sorted(keyed, key=lambda pair: pair[0])]


@dataclass(frozen=True)
class Geometry:
    instance_ids: tuple[str, ...]  # in slice order (k)
    shape: tuple[int, int, int]  # columns, rows, slices
    affine: NDArray[np.float64]  # RAS, voxel (i, j, k) -> mm
    voxel_ml: float

    def key(self) -> str:
        data = np.round(self.affine, 4).tobytes() + repr(self.shape).encode()
        return hashlib.sha256(data).hexdigest()[:16]


def geometry(instances: Sequence[dict[str, Any]], first: Dataset) -> Geometry:
    """The grid read_ct_series builds, from the tags: positions of the ordered instances, the
    orientation and pixel spacing (and rows, columns) of one slice header."""
    ordered = slice_order(instances)
    orientation = _floats(first.ImageOrientationPatient)
    spacing = _floats(first.PixelSpacing)  # row spacing, column spacing
    row_cosine, column_cosine = orientation[:3], orientation[3:]
    positions = np.array(
        [_floats(i["MainDicomTags"]["ImagePositionPatient"]) for i in ordered], dtype=np.float64
    )
    if len(positions) > 1:
        steps = np.diff(positions, axis=0)
        if not np.allclose(steps, steps[0], atol=1e-3):
            raise ValueError("slices are not evenly spaced")
        slice_step = steps[0]
    else:
        slice_step = np.cross(row_cosine, column_cosine) * float(first.SliceThickness)
    lps = np.eye(4)
    lps[:3, 0] = row_cosine * spacing[1]
    lps[:3, 1] = column_cosine * spacing[0]
    lps[:3, 2] = slice_step
    lps[:3, 3] = positions[0]
    voxel_ml = float(spacing[0] * spacing[1] * np.linalg.norm(slice_step)) / 1000
    shape = (int(first.Columns), int(first.Rows), len(ordered))
    return Geometry(tuple(str(i["ID"]) for i in ordered), shape, RAS_TO_LPS @ lps, voxel_ml)


def resample_nearest(
    mask: NDArray[Any],
    mask_affine: NDArray[np.float64],
    shape: tuple[int, int, int],
    target_affine: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """The mask on the target grid: each target voxel takes the nearest mask voxel at the same
    position in mm (0 outside the mask). Done one slice at a time to keep memory small."""
    to_mask = np.linalg.inv(mask_affine) @ target_affine
    out = np.zeros(shape, dtype=np.uint8)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    ones = np.ones_like(i)
    bounds = np.array(mask.shape)[:, None]
    for k in range(shape[2]):
        points = np.stack([i.ravel(), j.ravel(), np.full(i.size, k), ones.ravel()])
        index = np.rint(to_mask @ points)[:3].astype(np.int64)
        inside = np.all((index >= 0) & (index < bounds), axis=0)
        values = np.zeros(i.size, dtype=np.uint8)
        values[inside] = mask[index[0, inside], index[1, inside], index[2, inside]] > 0
        out[:, :, k] = values.reshape(shape[0], shape[1])
    return out


def windowed_png(ds: Dataset) -> bytes:
    """One CT slice as an 8-bit PNG with window 40/400 (rows down, columns across)."""
    hu = ds.pixel_array.astype(np.float64) * float(ds.get("RescaleSlope", 1)) + float(
        ds.get("RescaleIntercept", 0)
    )
    low = WINDOW_CENTER - WINDOW_WIDTH / 2
    pixels = np.clip((hu - low) / WINDOW_WIDTH, 0, 1) * 255
    buffer = io.BytesIO()
    Image.fromarray(np.round(pixels).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def mask_png(mask_slice: NDArray[np.uint8]) -> bytes:
    """One mask slice (i, j) as an RGBA PNG (rows down): colour where 1, transparent where 0."""
    rows = mask_slice.T > 0
    rgba = np.zeros((*rows.shape, 4), dtype=np.uint8)
    rgba[rows] = (*MASK_RGB, 255)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class CtCache:
    """In-memory caches (bounded) and the .npz folder for masks on the DICOM grid."""

    folder: Path | None = None
    max_slices: int = 400
    geometries: dict[str, Geometry] = field(default_factory=dict)
    masks: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    slices: OrderedDict[tuple[str, int], bytes] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def slice_png(self, key: tuple[str, int]) -> bytes | None:
        with self.lock:
            png = self.slices.get(key)
            if png is not None:
                self.slices.move_to_end(key)
            return png

    def keep_slice(self, key: tuple[str, int], png: bytes) -> None:
        with self.lock:
            self.slices[key] = png
            while len(self.slices) > self.max_slices:
                self.slices.popitem(last=False)

    def mask_on_grid(self, name: str, mask_path: Path, grid: Geometry) -> NDArray[np.uint8]:
        """The mask on the DICOM grid: from memory, the .npz cache, or resampled once."""
        key = f"{name}_{file_sha256(mask_path)[:16]}_{grid.key()}"
        if key in self.masks:
            return self.masks[key]
        cached = self.folder / f"{key}.npz" if self.folder else None
        if cached is not None and cached.is_file():
            with np.load(cached) as data:
                mask: NDArray[np.uint8] = data["mask"].astype(np.uint8)
        else:
            import nibabel as nib  # only the CT endpoints need it

            image = nib.load(mask_path)
            mask = resample_nearest(
                np.asarray(image.dataobj), np.asarray(image.affine), grid.shape, grid.affine
            )
            if cached is not None:
                try:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cached, mask=mask)
                except OSError:
                    pass  # a read-only or missing cache folder: keep it in memory only
        self.masks[key] = mask
        return mask
