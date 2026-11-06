"""The viewer API against the real Orthanc and Postgres of the test stack, and its container.

Runs only with AIML02_INTEGRATION=1 against the test stack (.env.test, compose project
aiml02-test), never the demo: `integration_orthanc()` refuses unless Orthanc's name is
aiml02-test, and `integration_db_url()` checks the database name. The test database is migrated
first (the migrations only add). Rows written here stay in the test database: reviews and the
audit log are append-only by design.
"""

import json
import os
import time
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from stack_helpers import REPO_ROOT, integration_db_url, integration_orthanc, integration_values
from test_worker_integration import tagged_dx

from scripts import upload_demo
from src.api import studies
from src.api.app import Services, create_app, load_thresholds
from src.api.db import PostgresDatabase
from src.dicom.render import to_png
from src.service import migrate
from src.service.orthanc import Orthanc

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
    assert (found["modality"], found["review_status"], found["source"]) == ("DX", "none", "demo")
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
