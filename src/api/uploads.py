"""Checks and conversion for uploaded files, without I/O. For education only, not for clinical use.

A file is accepted by its content, never by its name:
- PNG: the PNG signature, a mode the classifier reads (L, RGB, RGBA), and a whole image that
  decodes. It becomes a DX image (src/dicom/convert.py: png_to_dx).
- DICOM: the "DICM" preamble, modality DX or CR (a CT needs a full series, which is not
  supported), PatientIdentityRemoved = YES, and pixels the worker can read.
- Each side 256 to 4096 pixels. The size cap (20 MB) is applied by the API while reading.

The study gets new UIDs seeded by the upload ID and the file's sha256, so every upload is a new
study and never collides with a demo study. StudyDescription is "Upload <id>"; a PNG's PatientID
is "UPLOAD-<id>". The file name is never written into the DICOM.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset

from src.dicom.convert import png_to_dx, uid_for
from src.dicom.render import to_png

MAX_BYTES = 20 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 256, 4096
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_MODES = ("L", "RGB", "RGBA")
SCOPE_WARNING = (
    "Trained on NIH ChestX-ray14 frontal chest X-rays. Scores for other images have no meaning."
)
CT_NOT_SUPPORTED = "CT needs a full series; not supported for upload"

Kind = Literal["png", "dicom"]


class UploadRejectedError(ValueError):
    """422: the file is not one the viewer accepts; the message says why."""


@dataclass(frozen=True)
class Checked:
    kind: Kind
    width: int
    height: int


def _check_size(width: int, height: int) -> None:
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        raise UploadRejectedError(
            f"image is {width}x{height} pixels; each side must be {MIN_SIDE} to {MAX_SIDE}"
        )


def is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def check(data: bytes) -> Checked:
    if data.startswith(PNG_SIGNATURE):
        return _check_png(data)
    if is_dicom(data):
        return _check_dicom(data)
    raise UploadRejectedError("not a PNG or DICOM file (checked by its content, not its name)")


def _check_png(data: bytes) -> Checked:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise UploadRejectedError("not a valid PNG file")
            mode, (width, height) = image.mode, image.size
            if mode not in PNG_MODES:
                raise UploadRejectedError(
                    f"PNG mode {mode} is not supported; use 8-bit grey, RGB or RGBA"
                )
            _check_size(width, height)
            image.load()  # the whole image must decode, not only the header
    except UploadRejectedError:
        raise
    except Exception as error:  # truncated, corrupt, decompression bomb
        raise UploadRejectedError(f"not a valid PNG file ({type(error).__name__})") from None
    return Checked("png", width, height)


def _check_dicom(data: bytes) -> Checked:
    try:
        ds = dcmread(io.BytesIO(data))
    except Exception as error:
        raise UploadRejectedError(f"not a valid DICOM file ({type(error).__name__})") from None
    if "PixelData" not in ds:
        raise UploadRejectedError("not a valid DICOM image (no pixel data)")
    modality = str(ds.get("Modality", ""))
    if modality == "CT":
        raise UploadRejectedError(CT_NOT_SUPPORTED)
    if modality not in ("DX", "CR"):
        raise UploadRejectedError(f"modality {modality or 'missing'}: only DX or CR is accepted")
    if str(ds.get("PatientIdentityRemoved", "")).upper() != "YES":
        raise UploadRejectedError(
            "the file is not marked de-identified: PatientIdentityRemoved must be YES"
        )
    width, height = int(ds.get("Columns", 0)), int(ds.get("Rows", 0))
    _check_size(width, height)
    try:
        to_png(ds)  # the worker must be able to read the pixels
    except Exception as error:
        raise UploadRejectedError(f"cannot read the image pixels ({error})") from None
    return Checked("dicom", width, height)


def to_dicom(data: bytes, checked: Checked, upload_id: int, sha256: str) -> Dataset:
    """The DICOM to store: new UIDs seeded by the upload, StudyDescription "Upload <id>"."""
    seed = f"upload-{upload_id}-{sha256}"
    if checked.kind == "png":
        ds = png_to_dx(data, f"UPLOAD-{upload_id}", source=seed)
        ds.DerivationDescription = "Converted from an uploaded PNG"
    else:
        ds = dcmread(io.BytesIO(data))
        ds.StudyInstanceUID = uid_for("upload", seed, "study")
        ds.SeriesInstanceUID = uid_for("upload", seed, "series")
        ds.SOPInstanceUID = uid_for("upload", seed, "instance")
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.StudyDescription = f"Upload {upload_id}"
    return ds


def dicom_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()
