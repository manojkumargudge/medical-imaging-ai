"""Reviews, the audit log and the worklist's study and summary columns, with the fakes from
test_api.py (no network). The same rules on the real Postgres: test_api_integration.py."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from test_api import LABELS, THRESHOLDS, FakeDb, FakeOrthanc
from test_worker import dx

from src.api import studies
from src.api.app import Services, create_app, load_thresholds

Setup = tuple[FakeOrthanc, FakeDb, TestClient]


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    orthanc, db = FakeOrthanc(), FakeDb()
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    db.add_result("1.1", studies.CLASSIFIER, instances=[{"labels": {}}])
    client = TestClient(create_app(Services(orthanc, db, tmp_path, load_thresholds(THRESHOLDS))))
    return orthanc, db, client


def body(**overrides: Any) -> dict[str, Any]:
    review = {
        "decision": "approve",
        "reviewer": "A. Reviewer",
        "submit_token": str(uuid4()),
        "inference_result_id": 1,
    }
    return {**review, **overrides}


def post(client: TestClient, study: str = "s", **overrides: Any) -> Any:
    return client.post(f"/api/studies/{study}/reviews", json=body(**overrides))


def test_a_review_is_saved_with_one_audit_row(setup: Setup) -> None:
    _, db, client = setup
    response = post(client, comment="  looks fine  ")
    assert response.status_code == 201, response.text
    saved = response.json()["review"]
    assert (saved["decision"], saved["reviewer"], saved["comment"]) == (
        "approve",
        "A. Reviewer",
        "looks fine",
    )
    assert [(a["event"], a["actor"], a["details"]["review_id"]) for a in db.audit_rows] == [
        ("review", "A. Reviewer", saved["id"])
    ]


def test_the_same_token_twice_is_saved_once(setup: Setup) -> None:
    _, db, client = setup
    token = str(uuid4())
    first, second = post(client, submit_token=token), post(client, submit_token=token)
    print(first.status_code, second.status_code, second.json())
    assert (first.status_code, second.status_code) == (201, 409)
    assert "already saved" in second.json()["detail"]
    assert len(db.review_rows) == 1 and len(db.audit_rows) == 1


def test_a_stale_result_is_409(setup: Setup) -> None:
    _, db, client = setup
    db.add_result("1.1", studies.CLASSIFIER, instances=[{"labels": {}}])  # id 2 is newer
    response = post(client, inference_result_id=1)
    assert response.status_code == 409 and "newer result exists" in response.json()["detail"]
    assert post(client, inference_result_id=2).status_code == 201
    assert post(client, inference_result_id=99).status_code == 422  # not this study's result


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"decision": "reject"}, "a comment is required to reject"),
        ({"decision": "reject", "comment": "   "}, "a comment is required to reject"),
        ({"reviewer": "   "}, "reviewer"),
        ({"reviewer": None}, "reviewer"),
        ({"decision": "maybe"}, "decision"),
        ({"submit_token": "not-a-uuid"}, "submit_token"),
        ({"inference_result_id": None}, "inference_result_id"),
        ({"extra": "field"}, "extra"),
    ],
)
def test_missing_or_invalid_fields_are_422(
    setup: Setup, overrides: dict[str, Any], message: str
) -> None:
    _, db, client = setup
    response = post(client, **overrides)
    print(response.json()["detail"])
    assert response.status_code == 422 and message in response.json()["detail"]
    assert db.review_rows == [] and db.audit_rows == []


def test_missing_fields_are_422(setup: Setup) -> None:
    _, _, client = setup
    response = client.post("/api/studies/s/reviews", json={"decision": "approve"})
    detail = response.json()["detail"]
    assert response.status_code == 422
    assert all(f in detail for f in ("reviewer", "submit_token", "inference_result_id"))


def test_reviews_of_an_unknown_study_are_404(setup: Setup) -> None:
    _, _, client = setup
    assert post(client, study="nope").status_code == 404
    assert client.get("/api/studies/nope/reviews").status_code == 404


def test_history_newest_first_and_the_newest_is_the_state(setup: Setup) -> None:
    _, _, client = setup
    post(client)
    post(client, decision="reject", comment="overlay outside the lungs")
    history = client.get("/api/studies/s/reviews").json()["reviews"]
    assert [r["decision"] for r in history] == ["reject", "approve"]
    [study] = client.get("/api/studies").json()["studies"]
    assert study["review_status"] == "rejected"


def test_audit_pages_newest_first(setup: Setup) -> None:
    _, _, client = setup
    for _ in range(5):
        post(client)
    first = client.get("/api/audit?page=1&page_size=2").json()
    third = client.get("/api/audit?page=3&page_size=2").json()
    assert first["total"] == 5 and [e["id"] for e in first["entries"]] == [5, 4]
    assert [e["id"] for e in third["entries"]] == [1]
    assert client.get("/api/audit?page=0").status_code == 422
    assert client.get("/api/audit?page_size=500").status_code == 422


def test_worklist_study_name_and_summary(setup: Setup) -> None:
    orthanc, db, client = setup
    thresholds = load_thresholds(THRESHOLDS)["labels"]
    scores = {
        label: {"score": (entry["threshold"] or 0) + (0.01 if i < 4 else -0.001)}
        for i, (label, entry) in enumerate(thresholds.items())
    }
    db.results.clear()
    db.add_result("1.1", studies.CLASSIFIER, instances=[{"labels": scores}])
    above = sum(
        1
        for i, (label, entry) in enumerate(thresholds.items())
        if i < 4 and entry["threshold"] is not None
    )

    named = dx(2)
    named.StudyDescription = "NIH ChestX-ray14 00000116_000.png"
    orthanc.add("named", "1.2", {"DX": [named]})
    orthanc.add("ct", "1.3", {"CT": [dx(3)]})
    db.add_result("1.3", studies.SEGMENTATION, series=[{"spleen_ml": 440.764875}])
    orthanc.add("up", "1.4", {"DX": [dx(4)]})
    db.uploads = {"up": "my_scan.dcm"}
    orthanc.state["studies"]["named"]["MainDicomTags"]["StudyDescription"] = (  # type: ignore[attr-defined]
        named.StudyDescription
    )

    found = {s["id"]: s for s in client.get("/api/studies").json()["studies"]}
    print({k: (v["description"], v["summary"]) for k, v in found.items()})
    assert found["s"]["summary"] == f"{above} of 13 above threshold"
    assert found["ct"]["summary"] == "Spleen 440.8 mL"
    assert found["named"]["description"] == "NIH ChestX-ray14 00000116_000.png"
    assert found["up"]["description"] == "my_scan.dcm" and found["up"]["source"] == "upload"
    assert found["named"]["summary"] is None  # no result yet
    assert len(LABELS) == 14 and sum(t["threshold"] is not None for t in thresholds.values()) == 13
