"""Convert the public datasets to DICOM, and read the DICOM back.

- NIH ChestX-ray14 PNG -> one DX image (Digital X-Ray, For Presentation), MONOCHROME2. The NIH
  patient ID is kept as PatientID; there is no other identifying data (the dataset is public and
  de-identified, and PatientName is a placeholder built from the ID).
- MSD Task09_Spleen NIfTI -> a CT series, one file per slice, with ImagePositionPatient,
  ImageOrientationPatient, PixelSpacing and SliceThickness derived from the NIfTI affine.

All UIDs are new (under pydicom's root; nothing is copied from a source), and they are derived
from the source name (NIH image name, MSD case and slice). Converting the same file again gives the
same UIDs, so a re-upload to Orthanc updates instead of duplicating. The output is meant for testing
a DICOM pipeline (Orthanc, viewer), not for clinical use.

NIfTI uses RAS+ world coordinates and DICOM uses LPS+, so x and y change sign. A DICOM row runs
along increasing column index: the NIfTI i axis maps to DICOM columns, j to rows, k to slices.

Run:
    python -m src.dicom.convert nih --png data/raw/00000001_000.png
        --data-entry data/raw/Data_Entry_2017.csv --out data/dicom/nih/00000001_000.dcm
    python -m src.dicom.convert spleen --nifti data/raw/Task09_Spleen/imagesTr/spleen_2.nii.gz
        --out data/dicom/spleen_2
"""

from __future__ import annotations

import argparse
import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    PYDICOM_IMPLEMENTATION_UID,
    UID,
    CTImageStorage,
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    generate_uid,
)

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
NOTICE = "For education only, not for clinical use"
_INT16 = np.iinfo(np.int16)


@dataclass(frozen=True)
class NihEntry:
    """One row of Data_Entry_2017.csv, with what the DX header needs."""

    image: str
    patient: str
    age: int
    sex: str
    view: str
    original_size: tuple[int, int]  # width, height
    original_spacing: tuple[float, float]  # x, y in mm


def read_nih_entry(data_entry: Path, image: str) -> NihEntry:
    with data_entry.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["Image Index"] == image:
                return NihEntry(
                    image=image,
                    patient=row["Patient ID"],
                    age=int(row["Patient Age"]),
                    sex=row["Patient Gender"],
                    view=row["View Position"],
                    original_size=(int(row["OriginalImage[Width"]), int(row["Height]"])),
                    original_spacing=(
                        float(row["OriginalImagePixelSpacing[x"]),
                        float(row["y]"]),
                    ),
                )
    raise KeyError(f"{image} is not in {data_entry}")


def nih_pixel_spacing(entry: NihEntry, png_size: tuple[int, int]) -> tuple[float, float]:
    """(row, column) spacing in mm of the released PNG, which was resized from the original."""
    width, height = png_size
    original_width, original_height = entry.original_size
    spacing_x, spacing_y = entry.original_spacing
    return spacing_y * original_height / height, spacing_x * original_width / width


def _file_meta(sop_class: str, sop_instance: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(sop_class)
    meta.MediaStorageSOPInstanceUID = UID(sop_instance)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    return meta


def uid_for(*parts: str) -> str:
    """A new UID under pydicom's root, derived from the parts (same parts, same UID)."""
    return str(generate_uid(entropy_srcs=["aiml02", *parts]))


def _base(
    sop_class: str, patient_id: str, study_uid: str, series_uid: str, sop_instance: str
) -> Dataset:
    ds = Dataset()
    ds.file_meta = _file_meta(sop_class, sop_instance)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_instance
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = patient_id
    ds.PatientName = f"ANON^{patient_id}"
    ds.PatientBirthDate = ""
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Public research dataset; new UIDs; no names or dates"
    ds.StudyID = ""
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.Manufacturer = ""
    ds.ImageComments = NOTICE
    return ds


def png_to_dx(
    png: bytes,
    patient_id: str,
    sex: str = "",
    age: int | None = None,
    view: str = "",
    pixel_spacing: tuple[float, float] | None = None,
    source: str = "",
) -> Dataset:
    """An 8-bit grayscale PNG as a DX For Presentation image, MONOCHROME2.

    source (the NIH image name) seeds the UIDs; one image is one study with one series.
    """
    with Image.open(io.BytesIO(png)) as image:
        if image.mode not in {"L", "RGB", "RGBA"}:
            raise ValueError(f"unexpected PNG mode {image.mode}")
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    ds = _base(
        DigitalXRayImageStorageForPresentation,
        patient_id,
        uid_for("nih", source, "study"),
        uid_for("nih", source, "series"),
        uid_for("nih", source, "instance"),
    )
    ds.Modality = "DX"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.PresentationIntentType = "FOR PRESENTATION"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = view
    ds.PatientOrientation = ""
    ds.PatientSex = sex if sex in {"M", "F"} else ""
    # Ages above 100 in NIH are recording errors; they are left out rather than copied.
    ds.PatientAge = f"{age:03d}Y" if age is not None and 0 <= age <= 100 else ""
    ds.DerivationDescription = "Converted from NIH ChestX-ray14 PNG (8-bit, resized by NIH)"
    # The source image name, for the viewer's worklist (it holds only the NIH patient ID).
    ds.StudyDescription = f"NIH ChestX-ray14 {source}" if source else ""
    ds.BurnedInAnnotation = "NO"
    if pixel_spacing is not None:
        ds.ImagerPixelSpacing = [round(value, 6) for value in pixel_spacing]
        ds.PixelSpacing = [round(value, 6) for value in pixel_spacing]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    return ds


def save_new(ds: Dataset, path: Path) -> None:
    """Write a DICOM file (with preamble and meta); never replaces an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def read_dx(path: Path) -> tuple[NDArray[Any], tuple[float, float] | None, Dataset]:
    ds = dcmread(path)
    spacing = tuple(float(value) for value in ds.PixelSpacing) if "PixelSpacing" in ds else None
    return ds.pixel_array, spacing, ds  # type: ignore[return-value]


def _to_int16(volume: NDArray[Any]) -> tuple[NDArray[np.int16], float]:
    """Stored values and RescaleSlope; slope 1 when the data are whole numbers in int16 range."""
    if (
        np.allclose(volume, np.round(volume))
        and volume.min() >= _INT16.min
        and (volume.max() <= _INT16.max)
    ):
        return np.round(volume).astype(np.int16), 1.0
    slope = float(np.abs(volume).max()) / _INT16.max or 1.0
    return np.round(volume / slope).astype(np.int16), slope


def nifti_to_ct_series(
    nifti: Path,
    out_dir: Path,
    patient_id: str,
    series_description: str = "",
) -> list[Path]:
    """One CT file per slice (k axis) into an empty folder; returns the paths in slice order."""
    import nibabel as nib  # here, not at the top: the API image (PNG -> DX only) has no nibabel

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty; not overwriting")
    image = nib.load(nifti)
    volume = np.asarray(image.get_fdata(), dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"{nifti}: expected a 3D volume, got shape {volume.shape}")
    stored, slope = _to_int16(volume)
    lps = RAS_TO_LPS @ image.affine
    column_step, row_step, slice_step = (lps[:3, axis] for axis in range(3))
    column_spacing, row_spacing = np.linalg.norm(column_step), np.linalg.norm(row_step)
    slice_spacing = float(np.linalg.norm(slice_step))
    orientation = [*(column_step / column_spacing), *(row_step / row_spacing)]

    # Seeded by the patient (case) ID: one study, one series and one frame of reference per case.
    study_uid = uid_for("ct", patient_id, "study")
    series_uid = uid_for("ct", patient_id, "series")
    frame_uid = uid_for("ct", patient_id, "frame")
    paths = []
    for k in range(volume.shape[2]):
        ds = _base(
            CTImageStorage,
            patient_id,
            study_uid,
            series_uid,
            uid_for("ct", patient_id, "instance", str(k)),
        )
        ds.Modality = "CT"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = series_description
        ds.StudyDescription = series_description
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = ""
        ds.KVP = ""
        ds.PatientSex = ""
        ds.PatientAge = ""
        position = lps @ np.array([0.0, 0.0, float(k), 1.0])
        ds.ImagePositionPatient = [round(float(value), 6) for value in position[:3]]
        ds.ImageOrientationPatient = [round(float(value), 8) for value in orientation]
        ds.PixelSpacing = [round(float(row_spacing), 6), round(float(column_spacing), 6)]
        ds.SliceThickness = round(slice_spacing, 6)
        ds.SpacingBetweenSlices = round(slice_spacing, 6)
        ds.Rows, ds.Columns = volume.shape[1], volume.shape[0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = slope
        ds.RescaleType = "HU"
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.PixelData = np.ascontiguousarray(stored[:, :, k].T).tobytes()
        path = out_dir / f"IM_{k + 1:04d}.dcm"
        save_new(ds, path)
        paths.append(path)
    return paths


def read_ct_series(paths: Sequence[Path]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Volume (i, j, k) in HU and its RAS affine, rebuilt from the DICOM geometry tags alone."""
    slices = [dcmread(path) for path in paths]
    if not slices:
        raise ValueError("no files")
    for name in ("SeriesInstanceUID", "FrameOfReferenceUID"):
        if len({str(getattr(ds, name)) for ds in slices}) != 1:
            raise ValueError(f"files differ in {name}")
    orientation = np.array(slices[0].ImageOrientationPatient, dtype=np.float64)
    spacing = np.array(slices[0].PixelSpacing, dtype=np.float64)  # row, column
    for ds in slices:
        if not np.allclose(ds.ImageOrientationPatient, orientation) or not np.allclose(
            ds.PixelSpacing, spacing
        ):
            raise ValueError("slices differ in orientation or pixel spacing")
    row_cosine, column_cosine = orientation[:3], orientation[3:]
    normal = np.cross(row_cosine, column_cosine)
    positions = np.array([ds.ImagePositionPatient for ds in slices], dtype=np.float64)
    order = np.argsort(positions @ normal)
    slices = [slices[index] for index in order]
    positions = positions[order]

    if len(slices) > 1:
        steps = np.diff(positions, axis=0)
        if not np.allclose(steps, steps[0], atol=1e-3):
            raise ValueError("slices are not evenly spaced")
        slice_step = steps[0]
    else:
        slice_step = normal * float(slices[0].SliceThickness)

    lps = np.eye(4)
    lps[:3, 0] = row_cosine * spacing[1]
    lps[:3, 1] = column_cosine * spacing[0]
    lps[:3, 2] = slice_step
    lps[:3, 3] = positions[0]
    volume = np.stack(
        [
            ds.pixel_array.T.astype(np.float64) * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept)
            for ds in slices
        ],
        axis=2,
    )
    return volume, RAS_TO_LPS @ lps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    nih = commands.add_parser("nih")
    nih.add_argument("--png", type=Path, required=True)
    nih.add_argument("--data-entry", type=Path, required=True)
    nih.add_argument("--out", type=Path, required=True)
    ct = commands.add_parser("spleen")
    ct.add_argument("--nifti", type=Path, required=True)
    ct.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "nih":
        entry = read_nih_entry(args.data_entry, args.png.name)
        png = args.png.read_bytes()
        with Image.open(io.BytesIO(png)) as image:
            spacing = nih_pixel_spacing(entry, image.size)
        ds = png_to_dx(png, entry.patient, entry.sex, entry.age, entry.view, spacing, entry.image)
        save_new(ds, args.out)
        print(f"wrote {args.out}")
    else:
        case = args.nifti.name.removesuffix(".nii.gz")
        paths = nifti_to_ct_series(args.nifti, args.out, case, f"MSD Task09 {case}")
        print(f"wrote {len(paths)} slices to {args.out}")


if __name__ == "__main__":
    main()
