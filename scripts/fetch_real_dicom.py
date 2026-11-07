"""Download the real scanner chest X-ray DICOMs used by tests/test_real_dicom.py.

    python -m scripts.fetch_real_dicom

Files go to data/raw/real_dicom/ (gitignored; none of them is committed). Each file is checked
against the sha256 below. An existing file is never overwritten: it is kept if its hash matches,
and the script stops with an error if it does not.

Sources and licenses:
- RG1_J2KR / RG1_J2KI: NEMA DICOM WG04 compression test set (Philips CR, thorax), as mirrored
  by pydicom-data (MIT repository). NEMA gives no formal license for the WG04 images, so they
  are used locally only and never redistributed here.
- covid19ar_cr / covid19ar_dx: TCIA COVID-19-AR, CC BY 4.0. Cite: Desai S, et al. (2020). Data
  from Chest Imaging with Clinical and Genomic Correlates Representing a Rural COVID-19
  Positive Population. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2020.py71-5978

For education only, not for clinical use.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

OUT = Path("data/raw/real_dicom")
PYDICOM_DATA = (
    "https://github.com/pydicom/pydicom-data/raw/39a2eb31815eec435dc26c322c27aec5cfcbddb6/data/"
)
TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="


@dataclass(frozen=True)
class RealDicom:
    name: str  # file name under data/raw/real_dicom/
    url: str
    sha256: str  # of the .dcm file itself
    zipped: bool  # TCIA sends a zip holding the one image of the series
    source: str
    license: str


FILES = [
    RealDicom(
        "RG1_J2KR.dcm",
        PYDICOM_DATA + "RG1_J2KR.dcm",
        "7fbfd29360af806770102fd7c4ffcb2a133075bd00920a4bb63460d516f67ac4",
        False,
        "NEMA WG04 RG1 (Philips CR thorax) via pydicom-data",
        "no formal license (NEMA WG04 test data); local use only",
    ),
    RealDicom(
        "RG1_J2KI.dcm",
        PYDICOM_DATA + "RG1_J2KI.dcm",
        "744d01372fdda4e21b507bb7f97329065de961f4f263342079b369b430064d65",
        False,
        "NEMA WG04 RG1 (Philips CR thorax) via pydicom-data",
        "no formal license (NEMA WG04 test data); local use only",
    ),
    RealDicom(
        "covid19ar_cr.dcm",
        TCIA + "1.3.6.1.4.1.14519.5.2.1.9999.103.3144050301958868320974875419705",
        "dc0f21f5dbb910145e0901fde6e9b4861d50742104ffe46ee938bfff54e2cf86",
        True,
        "TCIA COVID-19-AR, patient COVID-19-AR-16434356 (Fujifilm CR, chest)",
        "CC BY 4.0",
    ),
    RealDicom(
        "covid19ar_dx.dcm",
        TCIA + "1.3.6.1.4.1.14519.5.2.1.9999.103.2982116618087775257016552924598",
        "aeca28499565b98c2f3eb6a664e48ae94e528adb77f488ba79c233f6d05015ff",
        True,
        "TCIA COVID-19-AR, patient COVID-19-AR-16424105 (Philips MobileDiagnost wDR, chest AP)",
        "CC BY 4.0",
    ),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def single_dicom(archive: bytes) -> bytes:
    """The one .dcm inside a TCIA series zip (members are read, never extracted to disk)."""
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
        if len(names) != 1:
            raise ValueError(f"expected one .dcm in the zip, found {names}")
        return z.read(names[0])


def fetch(item: RealDicom, out: Path = OUT) -> Path:
    path = out / item.name
    if path.exists():
        found = sha256(path.read_bytes())
        if found != item.sha256:
            raise SystemExit(
                f"{path} exists with sha256 {found}, expected {item.sha256}; not overwritten"
            )
        print(f"kept {path} (sha256 ok)")
        return path
    request = urllib.request.Request(item.url, headers={"User-Agent": "aiml02-fetch"})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (fixed https URLs)
        body = response.read()
    data = single_dicom(body) if item.zipped else body
    found = sha256(data)
    if found != item.sha256:
        raise SystemExit(f"{item.name}: sha256 {found}, expected {item.sha256}; not saved")
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"saved {path} ({len(data)} bytes, sha256 ok)")
    return path


def main() -> None:
    for item in FILES:
        fetch(item)


if __name__ == "__main__":
    main()
