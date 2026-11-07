"""Real scanner chest X-rays (scripts/fetch_real_dicom.py): decoding, rendering, model input.

The files are not in git (data/raw/ is ignored); the tests that need them skip when they are
absent, as in CI. Run `python -m scripts.fetch_real_dicom` first to include them. The fetch
script's own checks need no files and no network. For education only, not for clinical use.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset

from scripts import fetch_real_dicom
from scripts.check_real_dicom import as_bytes, with_new_uids
from src.dicom.render import to_png

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / fetch_real_dicom.OUT
REPORT = REPO_ROOT / "reports" / "real_dicom_check.json"
EXPECTED = {  # Modality, PhotometricInterpretation, TransferSyntaxUID, BitsStored
    "RG1_J2KR.dcm": ("CR", "MONOCHROME1", "1.2.840.10008.1.2.4.90", 15),
    "RG1_J2KI.dcm": ("CR", "MONOCHROME1", "1.2.840.10008.1.2.4.91", 15),
    "covid19ar_cr.dcm": ("CR", "MONOCHROME1", "1.2.840.10008.1.2", 10),
    "covid19ar_dx.dcm": ("DX", "MONOCHROME2", "1.2.840.10008.1.2", 12),
}


def real(name: str) -> Dataset:
    path = FOLDER / name
    if not path.exists():
        pytest.skip(f"{path} absent: run python -m scripts.fetch_real_dicom")
    return dcmread(path)


def test_the_fetch_list_is_pinned_and_licensed() -> None:
    names = [f.name for f in fetch_real_dicom.FILES]
    assert sorted(names) == sorted(EXPECTED)
    for item in fetch_real_dicom.FILES:
        assert len(item.sha256) == 64 and int(item.sha256, 16) >= 0, item.name
        assert item.url.startswith("https://") and item.license
    assert {f.license for f in fetch_real_dicom.FILES if f.name.startswith("covid")} == {
        "CC BY 4.0"
    }


def test_fetch_never_overwrites_a_different_file(tmp_path: Path) -> None:
    item = fetch_real_dicom.FILES[0]
    (tmp_path / item.name).write_bytes(b"something else")
    with pytest.raises(SystemExit, match="not overwritten"):
        fetch_real_dicom.fetch(item, tmp_path)
    assert (tmp_path / item.name).read_bytes() == b"something else"


def test_a_tcia_zip_must_hold_exactly_one_dicom() -> None:
    def archive(*names: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("LICENSE", "CC BY 4.0")
            for name in names:
                z.writestr(name, b"DICM")
        return buffer.getvalue()

    assert fetch_real_dicom.single_dicom(archive("1-1.dcm")) == b"DICM"
    with pytest.raises(ValueError, match="expected one"):
        fetch_real_dicom.single_dicom(archive("1-1.dcm", "1-2.dcm"))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_file_tags_and_hash(name: str) -> None:
    ds = real(name)
    item = next(f for f in fetch_real_dicom.FILES if f.name == name)
    assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == item.sha256
    found = (
        ds.Modality,
        ds.PhotometricInterpretation,
        str(ds.file_meta.TransferSyntaxUID),
        int(ds.BitsStored),
    )
    print(name, found)
    assert found == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_file_renders_with_bone_bright(name: str) -> None:
    """to_png decodes the file (JPEG 2000 too), keeps its size, uses the full grey range, and
    shows MONOCHROME1 inverted: the render runs opposite to the stored values."""
    ds = real(name)
    stored = ds.pixel_array.astype(np.float64)
    with Image.open(io.BytesIO(to_png(ds))) as image:
        shown = np.asarray(image, dtype=np.float64)
    assert shown.shape == stored.shape == (int(ds.Rows), int(ds.Columns))
    assert shown.min() == 0 and shown.max() == 255
    step = 8  # every 8th pixel is plenty for a correlation
    r = np.corrcoef(stored[::step, ::step].ravel(), shown[::step, ::step].ravel())[0, 1]
    print(name, ds.PhotometricInterpretation, f"r={r:.3f}")
    assert r < -0.9 if ds.PhotometricInterpretation == "MONOCHROME1" else r > 0.9


def test_lossy_and_lossless_rg1_render_alike() -> None:
    lossless = np.asarray(Image.open(io.BytesIO(to_png(real("RG1_J2KR.dcm")))), dtype=float)
    lossy = np.asarray(Image.open(io.BytesIO(to_png(real("RG1_J2KI.dcm")))), dtype=float)
    mean_diff = float(np.abs(lossless - lossy).mean())
    print(f"mean |J2KR - J2KI| {mean_diff:.2f} grey levels")
    assert mean_diff < 2


def test_new_uids_leave_the_compressed_pixels_alone() -> None:
    ds = real("RG1_J2KI.dcm")
    pixels = bytes(ds.PixelData)
    copy = dcmread(io.BytesIO(as_bytes(with_new_uids(ds, "seed"))))
    assert bytes(copy.PixelData) == pixels
    original = real("RG1_J2KI.dcm")
    for keyword in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        assert copy[keyword].value != original[keyword].value
    assert copy.file_meta.TransferSyntaxUID == original.file_meta.TransferSyntaxUID
    again = dcmread(io.BytesIO(as_bytes(with_new_uids(real("RG1_J2KI.dcm"), "seed"))))
    assert again.StudyInstanceUID == copy.StudyInstanceUID  # derived, so a rerun is the same


def test_the_stored_test_stack_report_says_every_file_passed() -> None:
    """reports/real_dicom_check.json is written by scripts/check_real_dicom.py on the test stack."""
    report: list[dict[str, Any]] = json.loads(REPORT.read_text(encoding="utf-8"))
    print([(e["file"], e["worker_status"], e["max_score_diff_vs_local"]) for e in report])
    assert sorted(e["file"] for e in report) == sorted(EXPECTED)
    for entry in report:
        assert entry["worker_status"] == "ok"
        assert entry["scores_match_local"] and entry["max_score_diff_vs_local"] <= 1e-6
        assert entry["cams_ok"] and entry["api"]["ok"] and entry["api"]["source"] == "unknown"
        assert len(entry["scores"]) == 14
