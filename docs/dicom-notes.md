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
