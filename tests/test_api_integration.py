"""The viewer API against the real Orthanc and Postgres of the test stack, and its container.

Runs only with AIML02_INTEGRATION=1 against the test stack (.env.test, compose project
aiml02-test), never the demo: `integration_orthanc()` refuses unless Orthanc's name is
aiml02-test, and `integration_db_url()` checks the database name. The test database is migrated
first (the migrations only add). Rows written here stay in the test database: reviews and the
audit log are append-only by design.
"""

import hashlib
import io
import json
import os
import random
import time
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import nibabel as nib
import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom import dcmread
from stack_helpers import REPO_ROOT, integration_db_url, integration_orthanc, integration_values
from test_worker import classifier, ct_series, segmenter
from test_worker_integration import tagged_dx

from scripts import upload_demo
from src.api import ct as ct_view
from src.api import studies
from src.api.app import Services, create_app, load_thresholds
from src.api.db import PostgresDatabase
from src.dicom import convert
from src.dicom.render import to_png
from src.service import migrate
from src.service.orthanc import Orthanc
from src.service.store import PostgresStore
from src.service.worker import Worker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIML02_INTEGRATION") != "1", reason="set AIML02_INTEGRATION=1"
    ),
]

THRESHOLDS = REPO_ROOT / "configs" / "viewer_thresholds.yaml"


@pytest.fixture(scope="module")
def stack() -> tuple[Orthanc, str]:
    orthanc = integration_orthanc()
    url = integration_db_url()
    with psycopg.connect(url) as connection:
        applied = migrate.migrate(connection, migrate.load())
        connection.commit()
    print(f"test database migrations applied now: {[m.name for m in applied] or 'none'}")
    return orthanc, url


def client(orthanc: Orthanc, url: str, results_dir: Path) -> TestClient:
    services = Services(
        replace(orthanc, timeout=10),
        PostgresDatabase(url),
        results_dir,
        load_thresholds(THRESHOLDS),
    )
    return TestClient(create_app(services))


def entry(api: TestClient, study_id: str) -> dict[str, Any]:
    response = api.get("/api/studies")
    assert response.status_code == 200, response.text
    matches: list[dict[str, Any]] = [s for s in response.json()["studies"] if s["id"] == study_id]
    [found] = matches
    return found


def execute(url: str, query: str, params: tuple[Any, ...]) -> Any:
    with psycopg.connect(url) as connection:
        row = connection.execute(query.encode(), params).fetchone()
        connection.commit()
    return row[0] if row else None


def test_a_study_from_upload_to_review(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    api = client(orthanc, url, tmp_path)
    ds = tagged_dx(uuid.uuid4().hex[:8])
    uid = upload_demo.upload_dx(orthanc, ds)["study_uid"]
    [study_id] = orthanc.find_study(uid)
    seen = [entry(api, study_id)["status"]]  # StableStudy comes 5 s after the last instance

    deadline = time.monotonic() + 60
    while not orthanc.get_json(f"/studies/{study_id}")["IsStable"]:
        assert time.monotonic() < deadline, "the study did not become stable"
        time.sleep(1)
    found = entry(api, study_id)
    seen.append(found["status"])
    assert (found["modality"], found["review_status"], found["source"]) == (
        "DX",
        "none",
        "unknown",
    )  # not a demo study, not an upload
    assert found["patient_id"] == str(ds.PatientID) and found["study_uid"] == uid

    cam = Path(uid) / studies.CLASSIFIER / "x_Cardiomegaly_cam.png"
    (tmp_path / cam).parent.mkdir(parents=True)
    (tmp_path / cam).write_bytes(b"\x89PNG raw map")
    results = {"instances": [{"labels": {"Cardiomegaly": {"score": 0.4, "cam": cam.as_posix()}}}]}
    result_id = execute(
        url,
        "INSERT INTO inference_results (study_uid, orthanc_study_id, modality, model, "
        "checkpoint_name, checkpoint_sha256, status, results, seconds) "
        "VALUES (%s, %s, 'DX', %s, 'fake.pt', 'sha-api-test', 'ok', %s, 1.0) RETURNING id",
        (uid, study_id, studies.CLASSIFIER, json.dumps(results)),
    )
    seen.append(entry(api, study_id)["status"])
    print(f"status: {seen}")
    assert seen == ["receiving", "processing", "done"]

    assert api.get(f"/api/studies/{study_id}/image").content == to_png(ds)
    assert api.get(f"/api/studies/{study_id}/cam/Cardiomegaly").content == b"\x89PNG raw map"
    for path in (
        f"/api/studies/{study_id}/cam/Bogus",
        f"/api/studies/{study_id}/cam/%2E%2E",
        f"/api/studies/{study_id}/cam/..%2F..%2F..%2Fetc%2Fpasswd",
        "/api/studies/..%2F..%2Fetc/cam/Cardiomegaly",
        "/api/studies/00000000-00000000-00000000-00000000-00000000/image",
    ):
        assert api.get(path).status_code == 404, path

    review = (
        "INSERT INTO reviews (study_uid, inference_result_id, decision, reviewer, comment, "
        "submit_token) VALUES (%s, %s, %s, 'Integration Test', %s, %s) RETURNING id"
    )
    execute(url, review, (uid, result_id, "approve", None, str(uuid.uuid4())))
    assert entry(api, study_id)["review_status"] == "approved"
    execute(url, review, (uid, result_id, "reject", "test comment", str(uuid.uuid4())))
    execute(
        url,
        "INSERT INTO uploads (uploader, kind, filename, size_bytes, sha256, study_uid, "
        "orthanc_study_id) VALUES ('Integration Test', 'dicom', 'x.dcm', 1, 'x', %s, %s) "
        "RETURNING id",
        (uid, study_id),
    )
    found = entry(api, study_id)
    detail = api.get(f"/api/studies/{study_id}").json()
    print(f"worklist entry: {found}")
    assert (found["review_status"], found["source"]) == ("rejected", "upload")
    assert detail["review"]["decision"] == "reject" and detail["results"][0]["id"] == result_id


def test_health_names_the_test_stack(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    body = client(*stack, tmp_path).get("/api/health").json()
    print(body)
    assert body["status"] == "ok"
    assert (body["orthanc"]["name"], body["orthanc"]["stack"]) == ("aiml02-test", "test")
    assert body["database"]["name"] == "aiml02_results_test"


def test_orthanc_or_database_down_is_503(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    closed = replace(orthanc, url="http://127.0.0.1:1", timeout=3)
    response = client(closed, url, tmp_path).get("/api/studies")
    print(response.status_code, response.json())
    assert response.status_code == 503
    assert "Orthanc is not reachable at http://127.0.0.1:1" in response.json()["detail"]

    services = Services(
        orthanc,
        PostgresDatabase("postgresql://aiml02:x@127.0.0.1:1/none", connect_timeout=2),
        tmp_path,
        load_thresholds(THRESHOLDS),
    )
    response = TestClient(create_app(services)).get("/api/studies")
    print(response.status_code, response.json())
    assert response.status_code == 503
    assert "results database is not reachable" in response.json()["detail"]


def test_the_api_container_serves_the_test_stack(stack: tuple[Orthanc, str]) -> None:
    """The container from docker-compose.api.yml (started with the test stack), on its port."""
    port = integration_values()["API_PORT"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=10) as response:
        body = json.loads(response.read())
    print(f"container on 127.0.0.1:{port}: {body}")
    assert body["status"] == "ok" and body["orthanc"]["name"] == "aiml02-test"
    assert body["database"]["name"] == "aiml02_results_test"
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/studies", timeout=30) as response:
        assert response.status == 200 and "studies" in json.loads(response.read())


def test_the_frontend_container_serves_the_viewer_and_proxies_the_api(
    stack: tuple[Orthanc, str],
) -> None:
    """The nginx container from docker-compose.frontend.yml, on the test stack's port."""
    port = integration_values()["FRONTEND_PORT"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
        page = response.read().decode()
        headers = response.headers
    assert '<div id="root">' in page and "For education only" in page
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=10) as response:
        body = json.loads(response.read())
    print(f"frontend on 127.0.0.1:{port}, /api/health through nginx: {body}")
    assert body["status"] == "ok" and body["orthanc"]["name"] == "aiml02-test"


def refused(url: str, query: str, params: tuple[Any, ...] = ()) -> str:
    with pytest.raises(psycopg.Error) as error, psycopg.connect(url) as connection:
        connection.execute(query.encode(), params)
    return str(error.value)


def count(url: str, query: str, params: tuple[Any, ...]) -> int:
    return int(execute(url, query, params))


def test_reviews_on_the_test_database(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    api = client(orthanc, url, tmp_path)
    uid = upload_demo.upload_dx(orthanc, tagged_dx(uuid.uuid4().hex[:8]))["study_uid"]
    [study_id] = orthanc.find_study(uid)
    insert = (
        "INSERT INTO inference_results (study_uid, orthanc_study_id, modality, model, "
        "checkpoint_name, checkpoint_sha256, status, results, seconds) "
        "VALUES (%s, %s, 'DX', %s, 'fake.pt', %s, 'ok', '{\"instances\": []}', 1.0) RETURNING id"
    )
    old = execute(url, insert, (uid, study_id, studies.CLASSIFIER, "sha-old"))
    new = execute(url, insert, (uid, study_id, studies.CLASSIFIER, "sha-new"))
    path = f"/api/studies/{study_id}/reviews"

    def review(**fields: Any) -> Any:
        body = {
            "decision": "approve",
            "reviewer": "Integration Test",
            "submit_token": str(uuid.uuid4()),
            "inference_result_id": new,
            **fields,
        }
        return api.post(path, json=body)

    stale = review(inference_result_id=old)
    assert stale.status_code == 409 and "newer result exists" in stale.json()["detail"]

    token = str(uuid.uuid4())
    first, again = review(submit_token=token), review(submit_token=token)
    rows = count(url, "SELECT count(*) FROM reviews WHERE submit_token = %s", (token,))
    print(f"same token twice: {first.status_code}, {again.status_code}; rows with it: {rows}")
    assert (first.status_code, again.status_code, rows) == (201, 409, 1)
    assert "already saved" in again.json()["detail"]

    no_comment = review(decision="reject")
    assert no_comment.status_code == 422
    assert "a comment is required to reject" in no_comment.json()["detail"]
    # The database refuses it too, if the API check were bypassed.
    message = refused(
        url,
        "INSERT INTO reviews (study_uid, inference_result_id, decision, reviewer, submit_token) "
        "VALUES (%s, %s, 'reject', 'Integration Test', %s)",
        (uid, new, str(uuid.uuid4())),
    )
    assert "reviews_reject_needs_comment" in message

    rejected = review(decision="reject", comment="map outside the lungs")
    assert rejected.status_code == 201
    saved = [first.json()["review"]["id"], rejected.json()["review"]["id"]]
    audited = [
        count(
            url,
            "SELECT count(*) FROM audit_log WHERE event = 'review' AND study_uid = %s "
            "AND (details ->> 'review_id')::bigint = %s",
            (uid, review_id),
        )
        for review_id in saved
    ]
    total = count(url, "SELECT count(*) FROM reviews WHERE study_uid = %s", (uid,))
    print(f"reviews saved {saved}, audit rows per review {audited}, reviews of the study {total}")
    assert audited == [1, 1] and total == 2

    history = api.get(path).json()["reviews"]
    assert [r["id"] for r in history] == saved[::-1]
    assert entry(api, study_id)["review_status"] == "rejected"
    page = api.get("/api/audit?page=1&page_size=5").json()
    assert page["total"] >= 2 and page["entries"][0]["details"]["review_id"] == saved[1]

    for table in ("reviews", "audit_log"):
        before = count(url, f"SELECT count(*) FROM {table} WHERE study_uid = %s", (uid,))
        for query in (
            f"UPDATE {table} SET study_uid = 'changed' WHERE study_uid = %s",
            f"DELETE FROM {table} WHERE study_uid = %s",
        ):
            assert "refused: the table is append-only" in refused(url, query, (uid,))
        after = count(url, f"SELECT count(*) FROM {table} WHERE study_uid = %s", (uid,))
        print(f"{table}: rows before {before}, after UPDATE/DELETE attempts {after}")
        assert before > 0 and before == after


# ---- uploads (POST /api/uploads) on the test stack ----


def upload_png(width: int = 320, height: int = 300, seed: int | None = None) -> bytes:
    rng = np.random.default_rng(seed if seed is not None else uuid.uuid4().int % 2**32)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (height, width), dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def rows_for(url: str, sha256: str) -> tuple[int, int]:
    uploads_rows = count(url, "SELECT count(*) FROM uploads WHERE sha256 = %s", (sha256,))
    audit_rows = count(
        url,
        "SELECT count(*) FROM audit_log WHERE event = 'upload' AND details ->> 'sha256' = %s",
        (sha256,),
    )
    return uploads_rows, audit_rows


def post_file(api: Any, data: bytes, name: str = "jane-doe-chest.png") -> Any:
    return api.post(
        "/api/uploads",
        files={"file": (name, data, "application/octet-stream")},
        data={"uploader": "Integration Test"},
    )


def test_an_uploaded_png_is_processed_to_done(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    api = client(orthanc, url, tmp_path)
    # The worker with fake models (no checkpoints on the test stack), on the test database.
    store = PostgresStore(url)
    store.set_last_seq(int(orthanc.get_json("/changes?last")["Last"]))
    worker = Worker(orthanc, store, [classifier(), segmenter()], tmp_path, log=lambda _: None)

    data = upload_png()
    sha256 = hashlib.sha256(data).hexdigest()
    response = post_file(api, data)
    assert response.status_code == 201, response.text
    body = response.json()
    study_id, upload_id = body["study_id"], body["upload_id"]
    seen = [entry(api, study_id)["status"]]
    deadline = time.monotonic() + 90
    while entry(api, study_id)["status"] != "done":
        assert time.monotonic() < deadline, f"not done: {entry(api, study_id)}"
        worker.poll_once()
        time.sleep(1)
    found = entry(api, study_id)
    seen.append(found["status"])
    store.close()
    print(f"upload {upload_id}: {seen}; worklist entry {found}")
    assert seen[0] in ("receiving", "processing") and seen[-1] == "done"
    assert (found["source"], found["description"], found["modality"]) == (
        "upload",
        f"Upload {upload_id}",
        "DX",
    )
    assert (
        found["patient_id"] == f"UPLOAD-{upload_id}" and "of 13 above threshold" in found["summary"]
    )

    [instance] = orthanc.get_json(f"/studies/{study_id}/instances")
    stored = orthanc.get_bytes(f"/instances/{instance['ID']}/file")
    assert b"jane-doe" not in stored  # the file name is not in the DICOM
    for path in ("/api/studies", f"/api/studies/{study_id}", "/api/audit?page_size=200"):
        assert "jane-doe" not in api.get(path).text  # nor anywhere the viewer reads
    assert rows_for(url, sha256) == (1, 1)


def test_a_de_identified_dx_dicom_is_accepted(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    ds = convert.png_to_dx(upload_png(), "SRC-IT", source=uuid.uuid4().hex)  # 320x300, valid
    original_uid = str(ds.StudyInstanceUID)
    data = upload_demo.dicom_bytes(ds)
    response = post_file(client(orthanc, url, tmp_path), data, name="scan.dcm")
    assert response.status_code == 201, response.text
    study = orthanc.get_json(f"/studies/{response.json()['study_id']}")
    assert study["MainDicomTags"]["StudyInstanceUID"] != original_uid
    assert rows_for(url, hashlib.sha256(data).hexdigest()) == (1, 1)


def test_rejected_uploads_add_no_rows(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    api = client(orthanc, url, tmp_path)
    ct = upload_demo.dicom_bytes(ct_series(tmp_path, f"up{uuid.uuid4().hex[:6]}")[0])
    no_phi_flag = tagged_dx(uuid.uuid4().hex[:8])
    del no_phi_flag.PatientIdentityRemoved
    cases = [
        ("CT", ct, 422, "CT needs a full series; not supported for upload"),
        ("PHI flag", upload_demo.dicom_bytes(no_phi_flag), 422, "PatientIdentityRemoved"),
        ("too small", upload_png(100, 100), 422, "each side must be 256 to 4096"),
        ("fake PNG", b"\x89PNG\r\n\x1a\n" + uuid.uuid4().bytes * 50, 422, "not a valid PNG"),
        ("over 20 MB", b"\x89PNG\r\n\x1a\n" + bytes(20 * 1024 * 1024), 413, "larger than 20 MB"),
    ]
    for name, data, status, message in cases:
        response = post_file(api, data)
        print(f"{name}: {response.status_code} {response.json()['detail']}")
        assert response.status_code == status and message in response.json()["detail"]
        assert rows_for(url, hashlib.sha256(data).hexdigest()) == (0, 0)


def test_a_duplicate_links_to_the_first_study(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    api = client(orthanc, url, tmp_path)
    data = upload_png()
    first, second = post_file(api, data), post_file(api, data, name="other-name.png")
    print(f"duplicate: {first.status_code} then {second.status_code} {second.json()}")
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["study_id"] == first.json()["study_id"] and second.json()["duplicate"]
    assert rows_for(url, hashlib.sha256(data).hexdigest()) == (1, 1)


def test_orthanc_down_is_503_and_adds_no_rows(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    closed = replace(orthanc, url="http://127.0.0.1:1", timeout=3)
    data = upload_png()
    response = post_file(client(closed, url, tmp_path), data)
    print(response.status_code, response.json())
    assert response.status_code == 503 and "Orthanc is not reachable" in response.json()["detail"]
    assert rows_for(url, hashlib.sha256(data).hexdigest()) == (0, 0)


def test_the_frontend_proxy_takes_uploads_over_1_mb(stack: tuple[Orthanc, str]) -> None:
    """nginx allows 21 MB on /api/uploads only (1 MB elsewhere)."""
    port = integration_values()["FRONTEND_PORT"]
    data = upload_png(1400, 1400)
    assert len(data) > 1024 * 1024
    response = httpx.post(
        f"http://127.0.0.1:{port}/api/uploads",
        files={"file": ("big.png", data, "image/png")},
        data={"uploader": "Integration Test"},
        timeout=60,
    )
    print(f"{len(data)} bytes through nginx: {response.status_code}")
    assert response.status_code == 201, response.text


# ---- CT slices and the mask on the DICOM grid ----


def test_ct_slices_and_mask_on_the_test_stack(stack: tuple[Orthanc, str], tmp_path: Path) -> None:
    orthanc, url = stack
    tag = uuid.uuid4().hex[:8]
    shape = (20, 16, 7)
    affine = np.diag([-0.8, -0.8, 2.5, 1.0])
    # Slice k holds HU -160 + 50 k: its windowed grey level shows which slice came back.
    volume = np.broadcast_to(-160 + 50 * np.arange(shape[2]), shape).astype(np.int16)
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "ct.nii.gz")
    paths = convert.nifti_to_ct_series(tmp_path / "ct.nii.gz", tmp_path / "dcm", f"ctit{tag}")
    datasets = [dcmread(p) for p in paths]
    numbers = list(range(1, len(datasets) + 1))
    random.Random(3).shuffle(numbers)  # InstanceNumbers that do not follow the slices
    for ds, number in zip(datasets, numbers, strict=True):
        ds.InstanceNumber = number
        orthanc.upload(upload_demo.dicom_bytes(ds))
    uid = str(datasets[0].StudyInstanceUID)
    [study_id] = orthanc.find_study(uid)

    # The worker's mask, stored on another grid (i flipped, i and j swapped), as the U-Net's is.
    label = np.zeros(shape, dtype=np.uint8)
    label[6:13, 4:10, 2:5] = 1
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, shape[0] - 1
    results_dir = tmp_path / "results"
    relative = f"{uid}/monai-unet-spleen/series_spleen_mask.nii.gz"
    (results_dir / relative).parent.mkdir(parents=True)
    stored = np.transpose(label[::-1, :, :], (1, 0, 2))
    _, ras = convert.read_ct_series(paths)
    nib.save(nib.Nifti1Image(stored, ras @ flip @ np.eye(4)[[1, 0, 2, 3]]), results_dir / relative)
    voxel_ml = 0.8 * 0.8 * 2.5 / 1000
    results = {"series": [{"mask": relative, "spleen_ml": float(label.sum()) * voxel_ml}]}
    execute(
        url,
        "INSERT INTO inference_results (study_uid, orthanc_study_id, modality, model, "
        "checkpoint_name, checkpoint_sha256, status, results, seconds) "
        "VALUES (%s, %s, 'CT', %s, 'fake.pt', 'sha-ct-test', 'ok', %s, 1.0) RETURNING id",
        (uid, study_id, studies.SEGMENTATION, json.dumps(results)),
    )

    cache = tmp_path / "cache"
    api = TestClient(
        create_app(
            Services(
                replace(orthanc, timeout=10),
                PostgresDatabase(url),
                results_dir,
                load_thresholds(THRESHOLDS),
                ct_view.CtCache(cache),
            )
        )
    )
    meta = api.get(f"/api/studies/{study_id}/ct/meta").json()
    print(f"CT meta: {meta}")
    assert (meta["slices"], meta["mask"], meta["first_mask_slice"], meta["last_mask_slice"]) == (
        7,
        True,
        2,
        4,
    )
    assert meta["window"] == {"center": 40.0, "width": 400.0}
    assert meta["spleen_ml_dicom_grid"] == pytest.approx(meta["spleen_ml"], rel=0.03)

    for k in range(shape[2]):
        with Image.open(io.BytesIO(api.get(f"/api/studies/{study_id}/ct/slice/{k}").content)) as im:
            grey = np.asarray(im)
        assert grey.shape == (shape[1], shape[0])  # rows, columns
        assert np.all(grey == round(50 * k / 400 * 255)), f"slice {k} is out of order"
        with Image.open(io.BytesIO(api.get(f"/api/studies/{study_id}/ct/mask/{k}").content)) as im:
            alpha = np.asarray(im)[..., 3] > 0
        assert np.array_equal(alpha, label[:, :, k].T > 0), f"mask slice {k} misplaced"
    for k in (7, -1, 1000):
        assert api.get(f"/api/studies/{study_id}/ct/slice/{k}").status_code == 404
        assert api.get(f"/api/studies/{study_id}/ct/mask/{k}").status_code == 404
    assert len(list(cache.glob("*.npz"))) == 1  # resampled once, then from the cache

    dx_uid = upload_demo.upload_dx(orthanc, tagged_dx(tag))["study_uid"]
    [dx_id] = orthanc.find_study(dx_uid)
    response = api.get(f"/api/studies/{dx_id}/ct/meta")
    assert response.status_code == 404 and "no CT series" in response.json()["detail"]


def test_a_missing_mask_file_still_serves_the_slices(
    stack: tuple[Orthanc, str], tmp_path: Path
) -> None:
    """The API container (port API_PORT) does not see the test's mask files: slices only."""
    port = integration_values()["API_PORT"]
    body = json.loads(
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/studies", timeout=30).read()
    )
    cts = [s for s in body["studies"] if s["modality"] == "CT" and s["status"] == "done"]
    assert cts, "run test_ct_slices_and_mask_on_the_test_stack first (same session)"
    base = f"http://127.0.0.1:{port}/api/studies/{cts[0]['id']}/ct"
    meta = json.loads(urllib.request.urlopen(f"{base}/meta", timeout=30).read())
    print(f"container CT meta without the mask file: {meta}")
    assert meta["slices"] > 0 and meta["mask"] is False and meta["first_mask_slice"] is None
    with urllib.request.urlopen(f"{base}/slice/0", timeout=30) as response:
        assert response.headers["Content-Type"] == "image/png"
