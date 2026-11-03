# DICOM notes

## Structure

DICOM organises imaging data in four levels:

| Level | Meaning | Identifier |
|---|---|---|
| Patient | One person | PatientID (0010,0020) |
| Study | One imaging exam or visit | StudyInstanceUID (0020,000D) |
| Series | One acquisition or reconstruction within a study; one modality | SeriesInstanceUID (0020,000E) |
| Instance | One object, usually one image or slice; one `.dcm` file | SOPInstanceUID (0008,0018) |

A file holds a 128-byte preamble, the letters `DICM`, file meta information (group 0002, including
TransferSyntaxUID), then a list of tagged data elements. Each tag is a (group,element) pair. There
is no separate header and image: pixels are one more element.

## Key tags

| Group | Tags |
|---|---|
| Patient | PatientName (0010,0010), PatientID (0010,0020), PatientBirthDate (0010,0030), PatientSex (0010,0040) |
| Study | StudyDate (0008,0020), StudyDescription (0008,1030), AccessionNumber (0008,0050) |
| Series | Modality (0008,0060), SeriesNumber (0020,0011), SeriesDescription (0008,103E), BodyPartExamined (0018,0015) |
| Instance | SOPClassUID (0008,0016), InstanceNumber (0020,0013) |
| Geometry | PixelSpacing (0028,0030), SliceThickness (0018,0050), ImagePositionPatient (0020,0032), ImageOrientationPatient (0020,0037) |

## Pixel data

- Pixels are in PixelData (7FE0,0010), and these tags say how to read them: Rows (0028,0010),
  Columns (0028,0011), SamplesPerPixel (0028,0002), PhotometricInterpretation (0028,0004),
  BitsAllocated (0028,0100), BitsStored (0028,0101), HighBit (0028,0102),
  PixelRepresentation (0028,0103).
- MONOCHROME1 means low values display white; MONOCHROME2 means low values display black.
- Stored values are often not physical values. RescaleSlope (0028,1053) and RescaleIntercept
  (0028,1052) convert them, for example to Hounsfield units in CT.
- WindowCenter (0028,1050) and WindowWidth (0028,1051) are display settings, not data changes.
- TransferSyntaxUID (0002,0010) gives the encoding, such as Explicit VR Little Endian
  (1.2.840.10008.1.2.1) or a compressed syntax like JPEG 2000.

## UIDs

- A UID is digits and dots, at most 64 characters, with no leading zeros in a component.
- UIDs must be globally unique. Without a registered root, use the `2.25.` prefix followed by a
  UUID written as a decimal integer.
- SOPClassUID says what kind of object it is and is a fixed standard value, for example CT Image
  Storage (1.2.840.10008.5.1.4.1.1.2), MR Image Storage (1.2.840.10008.5.1.4.1.1.4), Secondary
  Capture (1.2.840.10008.5.1.4.1.1.7). SOPInstanceUID identifies the object itself.

## Why our PNG/NIfTI data needs conversion for Orthanc

Orthanc is a DICOM server. It indexes objects by the patient/study/series/instance tags and
receives DICOM instances, so it has nothing to index in a PNG or NIfTI file.

| Format | What it lacks | What conversion must do |
|---|---|---|
| PNG | Patient, study and series tags; UIDs; pixel spacing; usually already windowed to 8 bits, so original values are gone | Wrap each image as a Secondary Capture instance; add IDs from the dataset's metadata |
| NIfTI | Patient, study and series tags; UIDs; one file holds a whole volume | Split the volume into one instance per slice (or one multi-frame object); compute ImagePositionPatient and ImageOrientationPatient for each slice; store values with valid rescale tags |

Points to get right:

- **Coordinates:** NIfTI world space is RAS+ and DICOM patient space is LPS+, so x and y change
  sign when the affine is converted. A mistake here can flip an image left to right with no error.
- **Stable UIDs:** derive UIDs from source identifiers, not at random, so a repeated conversion
  gives the same IDs. Orthanc builds its own resource IDs from a hash of PatientID and the three
  instance-level UIDs, so changed UIDs would create duplicate studies.
- **Patient IDs:** keep the dataset's real patient grouping in PatientID. Patient-level splits and
  the Orthanc hierarchy both depend on it. If one patient's images get different IDs, the
  patient-level split check can pass while patients still leak across splits.
- **Honest metadata:** missing values must not be invented to look like a real acquisition. Mark
  converted images as derived (ConversionType (0008,0064) = `WSD` for Secondary Capture) and keep a
  source-file mapping.
- **Identifiers:** public datasets are de-identified; conversion must not add personal data.
- **Verification:** after upload, compare patient, study, series and instance counts from Orthanc
  with the source counts, assert both sides are non-empty and print them.

Tools to evaluate: pydicom (writing instances), highdicom (derived objects such as segmentations),
SimpleITK (NIfTI reading and geometry), and Orthanc's REST API (`POST /instances` to upload).

## What the code proved (src/dicom/convert.py, tests/test_dicom.py)

Every point below is checked by a test that writes DICOM with pydicom 3.0.2 and reads it back with
`pydicom.dcmread`. Nothing has been sent to Orthanc yet, and the files have not been checked with a
DICOM validator such as `dciodvfy`.

**NIH PNG -> DX**

- An 8-bit PNG becomes a Digital X-Ray Image Storage - For Presentation instance
  (1.2.840.10008.5.1.4.1.1.1.1), Modality `DX`, `MONOCHROME2`, 8 bits stored. The pixels read back
  identical, and a test image with different row and column gradients shows they are not
  transposed. RGB and RGBA PNGs are converted to one grayscale channel.
- PixelSpacing and ImagerPixelSpacing read back exactly. NIH resized the originals to 1024 x 1024
  without keeping the aspect ratio. The spacing of the released PNG is therefore
  `OriginalImagePixelSpacing x original size / 1024`, per axis, and the code computes it that way.
- PatientID is the NIH patient ID, so DICOM groups images by the same patient as the
  patient-level split. PatientName is `ANON^<id>`. Birth date and study date are empty, and
  PatientIdentityRemoved is `YES`. NIH ages above 100 (recording errors) are left out of
  PatientAge instead of being copied.

**MSD NIfTI -> CT series**

- One CT Image Storage instance (1.2.840.10008.5.1.4.1.1.2) per k slice. The int16 values are
  stored with RescaleSlope 1 and intercept 0 (HU); non-integer volumes get a slope instead.
- The geometry comes from the NIfTI affine after converting RAS to LPS (negate x and y):
  - ImageOrientationPatient = the unit i and j axes
  - PixelSpacing = [j spacing, i spacing], so rows first
  - ImagePositionPatient = the affine applied to (0, 0, k)
  - SliceThickness = the k spacing
- `read_ct_series` rebuilds the volume and the affine from these tags alone. It sorts the slices
  by position along the slice normal, not by file name or InstanceNumber. Round trips reproduce
  the voxels exactly and the affine within 1e-5 mm for three geometries:
  - axis-aligned RAS
  - x and y flipped (as in many CT exports)
  - an oblique, 17-degree rotated affine with unequal spacing
- All slices of a series share StudyInstanceUID, SeriesInstanceUID and FrameOfReferenceUID. Each
  slice has its own SOPInstanceUID.
- On real data, the MSD volume `spleen_2` (512 x 512 x 90, 0.795 x 0.795 x 5 mm, RAS) became 90 CT
  files. Reading them back gave a maximum HU error of 0 and the affine within 1e-4 mm (test
  `test_real_spleen_volume_round_trip`, skipped in CI because it needs the downloaded data). No NIH
  PNG has been converted yet: the images are not downloaded locally, so the DX path is tested on
  synthetic PNGs only.

**UIDs**

- UIDs are new: they use pydicom's root `1.2.826.0.1.3680043.8.498.` and nothing comes from a
  source. They are derived from the source name (NIH image name; MSD case and slice). Converting
  the same file twice gives the same UIDs, and different files give different UIDs. As noted
  above, this matters for Orthanc, which would otherwise store a re-upload as a new study.
- DX was used instead of Secondary Capture because the brief's viewer shows X-rays. The NIH PNGs
  are, however, derived 8-bit images, so the files are marked `DERIVED\SECONDARY`.
