"""POST /api/uploads with the fakes from test_api.py (no network). The same on the real test
stack, through to a worker result: test_api_integration.py."""

import io
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom import dcmread
from test_api import THRESHOLDS, FakeDb, FakeOrthanc

from src.api import uploads
from src.api.app import Services, create_app, load_thresholds
from src.dicom import convert

Setup = tuple[FakeOrthanc, FakeDb, TestClient]
SECRET_NAME = "jane-doe-chest-2026.png"


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    orthanc, db = FakeOrthanc(), FakeDb()
    client = TestClient(create_app(Services(orthanc, db, tmp_path, load_thresholds(THRESHOLDS))))
    return orthanc, db, client


def png(width: int = 300, height: int = 280, mode: str = "L", seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    if mode == "I;16":
        image = Image.fromarray(rng.integers(0, 65535, (height, width), dtype=np.uint16))
    else:
        channels = {"L": (), "RGB": (3,), "RGBA": (4,)}[mode]
        pixels = rng.integers(0, 256, (height, width, *channels), dtype=np.uint8)
        image = Image.fromarray(pixels, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def dicom(modality: str = "DX", deidentified: bool = True, side: int = 300) -> bytes:
    ds = convert.png_to_dx(png(side, side), "SRC-1", source="source.png")
    ds.Modality = modality
    if not deidentified:
        del ds.PatientIdentityRemoved
    return uploads.dicom_bytes(ds)


def send(
    client: TestClient, data: bytes, name: str = SECRET_NAME, uploader: str = "A. Uploader"
) -> Any:
    return client.post(
        "/api/uploads",
        files={"file": (name, data, "application/octet-stream")},
        data={"uploader": uploader},
    )


def test_a_png_becomes_a_dx_study_with_one_upload_and_one_audit_row(setup: Setup) -> None:
    orthanc, db, client = setup
    response = send(client, png(mode="RGB"))
    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["duplicate"], body["kind"], body["upload_id"]) == (False, "png", 1)
    assert "Scores for other images have no meaning" in body["scope_warning"]
    stored: bytes = orthanc.state["uploaded"]  # type: ignore[attr-defined]
    ds = dcmread(io.BytesIO(stored))
    print(ds.PatientID, ds.StudyDescription, ds.Modality)
    assert (ds.PatientID, ds.StudyDescription, ds.Modality) == ("UPLOAD-1", "Upload 1", "DX")
    assert ds.PatientIdentityRemoved == "YES" and "NIH" not in ds.StudyDescription
    assert b"jane-doe" not in stored  # the file name is never in the DICOM
    assert [r["kind"] for r in db.upload_rows] == ["png"]
    assert [(a["event"], a["actor"]) for a in db.audit_rows] == [("upload", "A. Uploader")]
    [study] = client.get("/api/studies").json()["studies"]
    assert (study["id"], study["source"], study["description"]) == (
        body["study_id"],
        "upload",
        "Upload 1",
    )
    # Nor in any response the viewer reads.
    for path in ("/api/studies", f"/api/studies/{body['study_id']}", "/api/audit"):
        assert "jane-doe" not in client.get(path).text


def test_a_de_identified_dx_dicom_gets_new_uids(setup: Setup) -> None:
    orthanc, db, client = setup
    original = dcmread(io.BytesIO(dicom()))
    response = send(client, dicom(), name="scan.dcm")
    assert response.status_code == 201, response.text
    ds = dcmread(io.BytesIO(orthanc.state["uploaded"]))  # type: ignore[attr-defined]
    assert ds.StudyInstanceUID != original.StudyInstanceUID
    assert ds.SOPInstanceUID == ds.file_meta.MediaStorageSOPInstanceUID != original.SOPInstanceUID
    assert ds.StudyDescription == "Upload 1" and db.upload_rows[0]["kind"] == "dicom"


@pytest.mark.parametrize(
    ("data", "status", "message"),
    [
        (dicom("CT"), 422, "CT needs a full series; not supported for upload"),
        (dicom("MR"), 422, "only DX or CR"),
        (dicom(deidentified=False), 422, "PatientIdentityRemoved must be YES"),
        (png(100, 100), 422, "each side must be 256 to 4096"),
        (png(300, 4100), 422, "each side must be 256 to 4096"),
        (dicom(side=200), 422, "each side must be 256 to 4096"),
        (png(mode="I;16"), 422, "PNG mode I;16 is not supported"),
        (uploads.PNG_SIGNATURE + b"not really a png" * 10, 422, "not a valid PNG"),
        (png()[:-100], 422, "not a valid PNG"),
        (b"just some text, named .png", 422, "not a PNG or DICOM file"),
        (b"\0" * 128 + b"DICM" + b"garbage" * 10, 422, "not a valid DICOM"),
        (b"", 422, "the file is empty"),
        (uploads.PNG_SIGNATURE + b"\0" * (20 * 1024 * 1024), 413, "larger than 20 MB"),
    ],
)
def test_rejected_files_add_nothing(setup: Setup, data: bytes, status: int, message: str) -> None:
    orthanc, db, client = setup
    response = send(client, data)
    print(status, response.json()["detail"])
    assert response.status_code == status and message in response.json()["detail"]
    assert db.upload_rows == [] and db.audit_rows == []
    assert "uploaded" not in orthanc.state  # type: ignore[attr-defined]


def test_exactly_20_mb_is_accepted_by_the_size_check(setup: Setup) -> None:
    _, _, client = setup
    response = send(client, b"x" * uploads.MAX_BYTES)
    assert response.status_code == 422 and "not a PNG or DICOM" in response.json()["detail"]


@pytest.mark.parametrize("uploader", ["", "   ", "x" * 101])
def test_an_uploader_name_is_required(setup: Setup, uploader: str) -> None:
    _, db, client = setup
    response = send(client, png(), uploader=uploader)
    assert response.status_code == 422 and "uploader" in response.json()["detail"]
    assert db.upload_rows == []


def test_the_same_file_twice_links_to_the_first_study(setup: Setup) -> None:
    _, db, client = setup
    data = png(seed=7)
    first, second = send(client, data), send(client, data, name="renamed.png")
    print(first.status_code, second.status_code, second.json())
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["duplicate"] is True
    assert second.json()["study_id"] == first.json()["study_id"]
    assert len(db.upload_rows) == 1 and len(db.audit_rows) == 1


def test_orthanc_down_is_503_and_adds_no_rows(setup: Setup) -> None:
    orthanc, db, client = setup
    orthanc.state["down"] = urllib.error.URLError(ConnectionRefusedError(10061, "refused"))  # type: ignore[attr-defined]
    response = send(client, png())
    assert response.status_code == 503 and "Orthanc is not reachable" in response.json()["detail"]
    assert db.upload_rows == [] and db.audit_rows == []


def test_a_bad_png_mode_or_size_is_found_by_check_directly() -> None:
    assert uploads.check(png(256, 4096)).kind == "png"
    assert uploads.check(png(mode="RGBA")) == uploads.Checked("png", 300, 280)
    assert uploads.check(dicom()) == uploads.Checked("dicom", 300, 300)
    with pytest.raises(uploads.UploadRejectedError):
        uploads.check(png(255, 300))
