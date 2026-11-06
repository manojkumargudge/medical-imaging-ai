"""DICOM pixels as 8-bit PNG, without the model libraries (used by the worker and the API).

For education only, not for clinical use.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.pixels.processing import apply_voi_lut


def to_png(ds: Dataset) -> bytes:
    """A grayscale DX/CR image as an 8-bit PNG, the format the classifier was trained on.

    8-bit MONOCHROME2 pixels (NIH images converted by src/dicom) are passed through unchanged.
    Other images get their VOI LUT or window, are scaled to 0-255 and inverted if MONOCHROME1;
    such images differ from the NIH training data.
    """
    pixels = ds.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"expected one grayscale frame, got shape {pixels.shape}")
    photometric = str(ds.PhotometricInterpretation)
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"expected a grayscale image, got {photometric}")
    if pixels.dtype != np.uint8 or photometric == "MONOCHROME1":
        values = np.asarray(apply_voi_lut(pixels, ds), dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
        if photometric == "MONOCHROME1":
            scaled = 1 - scaled
        pixels = np.round(scaled * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
