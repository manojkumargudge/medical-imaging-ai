"""The viewer API with a fake Orthanc and a fake results database (no network, no Docker)."""

import io
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset
from test_worker import dx

from src.api import studies
from src.api.app import Services, create_app, load_thresholds
from src.api.db import (
    ALREADY_SAVED,
    NEWER_RESULT,
    DatabaseUnavailableError,
    NewReview,
    NewUpload,
    ReviewConflictError,
    ReviewInvalidError,
)
from src.dicom.render import to_png
from src.service.orthanc import Orthanc, OrthancError

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = REPO_ROOT / "configs" / "viewer_thresholds.yaml"
LABELS = list(yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))["labels"])
T0 = datetime(2026, 11, 6, 5, 0, tzinfo=UTC)


def dicom_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


class FakeOrthanc(Orthanc):
    """The part of Orthanc's REST API the API reads; `down` makes every call fail."""

    def __init__(self, name: str = "aiml02-test") -> None:
        super().__init__("http://fake-orthanc", "u", "p")
        state: dict[str, Any] = {"name": name, "studies": {}, "files": {}, "down": None}
        object.__setattr__(self, "state", state)

    def add(
        self, study_id: str, uid: str, series: dict[str, list[Dataset]], stable: bool = True
    ) -> None:
        state = self.state  # type: ignore[attr-defined]
        listed = []
        for number, (modality, datasets) in enumerate(series.items()):
            instances = []
            for index, ds in enumerate(datasets):
                instance = f"{study_id}-{number}-{index}"
                state["files"][instance] = dicom_bytes(ds)
                instances.append(instance)
            listed.append(
                {
                    "ID": f"{study_id}-s{number}",
                    "ParentStudy": study_id,
                    "MainDicomTags": {"Modality": modality},
                    "Instances": instances,
                }
            )
        state["studies"][study_id] = {
            "ID": study_id,
            "IsStable": stable,
            "LastUpdate": "20261106T040000",
            "MainDicomTags": {"StudyInstanceUID": uid},
            "PatientMainDicomTags": {"PatientID": f"patient-{study_id}"},
            "series": listed,
        }

    def _check(self) -> None:
        down = self.state["down"]  # type: ignore[attr-defined]
        if down:
            raise down

    def get_json(self, path: str) -> Any:
        self._check()
        state = self.state  # type: ignore[attr-defined]
        found = state["studies"]
        public = [{k: v for k, v in s.items() if k != "series"} for s in found.values()]
        if path == "/system":
            return {"Name": state["name"]}
        if path == "/studies":
            return list(found)
        if path == "/studies?expand":
            return public
        if path == "/series?expand":
            return [x for s in found.values() for x in s["series"]]
        parts = path.strip("/").split("/")
        if parts[0] == "studies" and parts[1] in found:
            study = found[parts[1]]
            return (
                study["series"] if parts[2:] == ["series"] else public[list(found).index(parts[1])]
            )
        raise OrthancError(404, path, "Unknown resource")

    def get_bytes(self, path: str) -> bytes:
        self._check()
        files: dict[str, bytes] = self.state["files"]  # type: ignore[attr-defined]
        return files[path.split("/")[2]]

    def upload(self, dicom: bytes) -> dict[str, Any]:
        """Store one instance as a new study (the API gives every upload new UIDs)."""
        self._check()
        ds = dcmread(io.BytesIO(dicom))
        study_id = f"up-{len(self.state['studies'])}"  # type: ignore[attr-defined]
        self.add(study_id, str(ds.StudyInstanceUID), {str(ds.Modality): [ds]})
        self.state["studies"][study_id]["MainDicomTags"]["StudyDescription"] = str(  # type: ignore[attr-defined]
            ds.get("StudyDescription", "")
        )
        self.state["uploaded"] = dicom  # type: ignore[attr-defined]
        return {"ID": f"{study_id}-0-0", "ParentStudy": study_id, "Status": "Success"}


@dataclass
class FakeDb:
    """The results database's rules as the real one applies them (tested on Postgres in
    test_api_integration.py): newest result only, one row per submit token, audit per review."""

    results: list[dict[str, Any]] = field(default_factory=list)
    review_rows: list[dict[str, Any]] = field(default_factory=list)
    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    uploads: set[str] = field(default_factory=set)  # Orthanc study IDs or UIDs of uploads
    upload_rows: list[dict[str, Any]] = field(default_factory=list)
    next_upload_id: int = 1
    tokens: set[UUID] = field(default_factory=set)
    down: bool = False

    def _check(self) -> None:
        if self.down:
            raise DatabaseUnavailableError("the results database is not reachable: refused")

    def name(self) -> str:
        self._check()
        return "aiml02_results_test"

    def newest_results(self, study_uid: str | None = None) -> list[dict[str, Any]]:
        self._check()
        newest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(self.results, key=lambda r: r["id"]):
            newest[(row["study_uid"], row["model"])] = row
        return [r for r in newest.values() if study_uid in (None, r["study_uid"])]

    def newest_reviews(self, study_uid: str | None = None) -> dict[str, dict[str, Any]]:
        self._check()
        newest = {r["study_uid"]: r for r in sorted(self.review_rows, key=lambda r: r["id"])}
        return {k: v for k, v in newest.items() if study_uid in (None, k)}

    def uploaded(self) -> set[str]:
        self._check()
        rows = {v for r in self.upload_rows for v in (r["study_uid"], r["orthanc_study_id"])}
        return set(self.uploads) | rows

    def find_upload(self, sha256: str) -> dict[str, Any] | None:
        self._check()
        found = [r for r in self.upload_rows if r["sha256"] == sha256]
        return found[-1] if found else None

    def reserve_upload_id(self) -> int:
        self._check()
        self.next_upload_id += 1
        return self.next_upload_id - 1

    def add_upload(self, upload: NewUpload) -> dict[str, Any]:
        self._check()
        row = {**upload.__dict__, "created_at": T0}
        self.upload_rows.append(row)
        self.audit_rows.append(
            {
                "id": len(self.audit_rows) + 1,
                "created_at": T0,
                "actor": upload.uploader,
                "event": "upload",
                "study_uid": upload.study_uid,
                "details": {"upload_id": upload.id, "kind": upload.kind},
            }
        )
        return row

    def reviews(self, study_uid: str) -> list[dict[str, Any]]:
        self._check()
        return [r for r in reversed(self.review_rows) if r["study_uid"] == study_uid]

    def add_review(self, review: NewReview) -> dict[str, Any]:
        self._check()
        seen = [
            r
            for r in self.results
            if r["id"] == review.inference_result_id and r["study_uid"] == review.study_uid
        ]
        if not seen:
            raise ReviewInvalidError("inference_result_id is not a result of this study")
        newest = max(
            r["id"]
            for r in self.results
            if r["study_uid"] == review.study_uid and r["model"] == seen[0]["model"]
        )
        if newest != review.inference_result_id:
            raise ReviewConflictError(NEWER_RESULT)
        if review.submit_token in self.tokens:
            raise ReviewConflictError(ALREADY_SAVED)
        self.tokens.add(review.submit_token)
        row = {
            "id": len(self.review_rows) + 1,
            "study_uid": review.study_uid,
            "inference_result_id": review.inference_result_id,
            "decision": review.decision,
            "reviewer": review.reviewer,
            "comment": review.comment,
            "created_at": T0,
        }
        self.review_rows.append(row)
        self.audit_rows.append(
            {
                "id": len(self.audit_rows) + 1,
                "created_at": T0,
                "actor": review.reviewer,
                "event": "review",
                "study_uid": review.study_uid,
                "details": {"review_id": row["id"], "decision": review.decision},
            }
        )
        return row

    def audit(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        self._check()
        newest_first = list(reversed(self.audit_rows))
        return newest_first[offset : offset + limit], len(newest_first)

    def add_result(self, uid: str, model: str, status: str = "ok", **results: Any) -> None:
        self.results.append(
            {
                "id": len(self.results) + 1,
                "study_uid": uid,
                "model": model,
                "status": status,
                "created_at": T0,
                "results": results,
            }
        )


def classifier_results(cam_paths: dict[str, str]) -> dict[str, Any]:
    labels = {label: {"score": 0.5, "overlay": "o.png"} for label in LABELS}
    for label, path in cam_paths.items():
        labels[label]["cam"] = path
    return {"instances": [{"labels": labels}]}


@pytest.fixture
def setup(tmp_path: Path) -> tuple[FakeOrthanc, FakeDb, TestClient, Path]:
    orthanc, db = FakeOrthanc(), FakeDb()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    client = TestClient(create_app(Services(orthanc, db, results_dir, load_thresholds(THRESHOLDS))))
    return orthanc, db, client, results_dir


def statuses(client: TestClient) -> dict[str, tuple[str, str, str]]:
    response = client.get("/api/studies")
    assert response.status_code == 200
    return {
        s["id"]: (s["status"], s["review_status"], s["source"]) for s in response.json()["studies"]
    }


def test_worklist_status_review_and_source(
    setup: tuple[FakeOrthanc, FakeDb, TestClient, Path],
) -> None:
    orthanc, db, client, _ = setup
    orthanc.add("rx", "1.1", {"DX": [dx(1)]}, stable=False)
    orthanc.add("waiting", "1.2", {"DX": [dx(2)]})
    orthanc.add("done", "1.3", {"DX": [dx(3)]})
    orthanc.add("failed", "1.4", {"DX": [dx(4)]})
    orthanc.add("mr", "1.5", {"MR": [dx(5)]})
    orthanc.add("both", "1.6", {"DX": [dx(6)], "CT": [dx(7)]})
    db.add_result("1.3", studies.CLASSIFIER)
    db.add_result("1.4", studies.CLASSIFIER, "error")
    db.add_result("1.6", studies.CLASSIFIER)  # the CT model has not finished
    db.review_rows += [
        {"id": 1, "study_uid": "1.3", "decision": "approve", "created_at": T0},
        {"id": 2, "study_uid": "1.3", "decision": "reject", "created_at": T0},
        {"id": 3, "study_uid": "1.4", "decision": "approve", "created_at": T0},
    ]
    db.uploads = {"failed"}
    found = statuses(client)
    print(found)
    assert found == {
        "rx": ("receiving", "none", "unknown"),
        "waiting": ("processing", "none", "unknown"),
        "done": ("done", "rejected", "unknown"),  # the newest review is the current state
        "failed": ("failed", "approved", "upload"),
        "mr": ("no_model", "none", "unknown"),
        "both": ("processing", "none", "unknown"),
    }
    first = client.get("/api/studies").json()["studies"][0]
    assert first["patient_id"].startswith("patient-") and first["updated_at"].endswith("+00:00")


def test_a_failed_rerun_shows_the_newest_row(
    setup: tuple[FakeOrthanc, FakeDb, TestClient, Path],
) -> None:
    orthanc, db, client, _ = setup
    orthanc.add("s", "1.1", {"CT": [dx(1)]})
    db.add_result("1.1", studies.SEGMENTATION, "error")
    db.add_result("1.1", studies.SEGMENTATION, "ok")
    assert statuses(client)["s"][0] == "done"


def test_study_detail(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    orthanc, db, client, _ = setup
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    db.add_result("1.1", studies.CLASSIFIER, spleen="x")
    db.review_rows.append({"id": 1, "study_uid": "1.1", "decision": "approve", "created_at": T0})
    body = client.get("/api/studies/s").json()
    assert body["study"]["status"] == "done" and body["review"]["decision"] == "approve"
    assert body["results"][0]["results"] == {"spleen": "x"}
    assert "not calibrated probabilities" in body["score_note"]
    assert body["series"] == [{"id": "s-s0", "modality": "DX", "description": "", "instances": 1}]


def test_dx_image_is_the_dicom_pixels(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    orthanc, _, client, _ = setup
    ds = dx(8)
    orthanc.add("s", "1.1", {"DX": [ds]})
    orthanc.add("ct", "1.2", {"CT": [dx(9)]})
    response = client.get("/api/studies/s/image")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content == to_png(ds)
    with Image.open(io.BytesIO(response.content)) as image:
        assert np.array_equal(np.asarray(image), ds.pixel_array)
    assert client.get("/api/studies/ct/image").status_code == 404


def test_raw_cam(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    orthanc, db, client, results_dir = setup
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    (results_dir / "1.1").mkdir()
    (results_dir / "1.1" / "cam.png").write_bytes(b"\x89PNG fake")
    db.results.append(
        {
            "id": 1,
            "study_uid": "1.1",
            "model": studies.CLASSIFIER,
            "status": "ok",
            "created_at": T0,
            "results": classifier_results({"Cardiomegaly": "1.1/cam.png"}),
        }
    )
    response = client.get("/api/studies/s/cam/Cardiomegaly")
    assert response.status_code == 200 and response.content == b"\x89PNG fake"
    # A label of the 14 without a raw map (a result from before maps were stored).
    assert "no raw map" in client.get("/api/studies/s/cam/Hernia").json()["detail"]
    assert client.get("/api/studies/s/cam/Bogus").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/studies/s/cam/%2E%2E",  # reaches the server as label ".."
        "/api/studies/s/cam/%2e%2e%2f%2e%2e%2fsecret.txt",
        "/api/studies/%2E%2E/cam/Cardiomegaly",
        "/api/studies/s/cam/..%2F..%2Fsecret.txt",
        "/api/studies/s/cam/..%5C..%5Csecret.txt",
        "/api/studies/..%2F..%2Fsecret.txt/cam/Cardiomegaly",
        "/api/studies/../cam/Cardiomegaly",
        "/api/studies/s/../../secret.txt",
        "/api/studies/..%2Fs/image",
        "/api/studies/nope/cam/Cardiomegaly",
        "/api/studies/nope",
    ],
)
def test_path_traversal_and_unknown_ids_are_404(
    setup: tuple[FakeOrthanc, FakeDb, TestClient, Path], path: str
) -> None:
    orthanc, db, client, results_dir = setup
    (results_dir.parent / "secret.txt").write_text("secret")
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    db.add_result("1.1", studies.CLASSIFIER, instances=[{"labels": {}}])
    response = client.get(path)
    assert response.status_code == 404 and b"secret" not in response.content


def test_a_stored_path_outside_the_results_folder_is_not_served(
    setup: tuple[FakeOrthanc, FakeDb, TestClient, Path],
) -> None:
    orthanc, db, client, results_dir = setup
    (results_dir.parent / "secret.png").write_bytes(b"secret")
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    db.results.append(
        {
            "id": 1,
            "study_uid": "1.1",
            "model": studies.CLASSIFIER,
            "status": "ok",
            "created_at": T0,
            "results": classifier_results({"Mass": "../secret.png"}),
        }
    )
    response = client.get("/api/studies/s/cam/Mass")
    assert response.status_code == 404 and response.content != b"secret"


def test_thresholds(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    body = setup[2].get("/api/config/thresholds").json()
    assert list(body["labels"]) == LABELS and len(LABELS) == 14
    assert body["labels"]["Hernia"] == {
        "set": "score_only",
        "threshold": None,
        "low_reliability": False,
    }
    assert "not calibrated probabilities" in body["note"] and "education" in body["notice"]


def test_health(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    orthanc, db, client, _ = setup
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["orthanc"]["stack"] == "test"
    db.down = True
    response = client.get("/api/health")
    assert response.status_code == 503 and "not reachable" in response.json()["database"]["error"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), "not reachable"),
        (TimeoutError("timed out"), "not reachable"),
        (OrthancError(401, "/system", "Unauthorized"), "refused the API's credentials"),
    ],
)
def test_orthanc_down_is_503_with_a_clear_message(
    setup: tuple[FakeOrthanc, FakeDb, TestClient, Path], error: Exception, message: str
) -> None:
    orthanc, _, client, _ = setup
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    orthanc.state["down"] = error  # type: ignore[attr-defined]
    for path in ("/api/health", "/api/studies", "/api/studies/s", "/api/studies/s/image"):
        response = client.get(path)
        detail = str(response.json())
        print(path, response.status_code, detail)
        assert response.status_code == 503 and message in detail and "http://fake-orthanc" in detail


def test_database_down_is_503(setup: tuple[FakeOrthanc, FakeDb, TestClient, Path]) -> None:
    orthanc, db, client, _ = setup
    orthanc.add("s", "1.1", {"DX": [dx(1)]})
    db.down = True
    for path in ("/api/studies", "/api/studies/s"):
        response = client.get(path)
        assert response.status_code == 503 and "results database" in response.json()["detail"]


def test_model_names_match_the_worker() -> None:
    from src.service.analyzers import ClassifierAnalyzer, SegmentationAnalyzer

    assert ClassifierAnalyzer.name == studies.CLASSIFIER
    assert SegmentationAnalyzer.name == studies.SEGMENTATION
    assert set(ClassifierAnalyzer.modalities) | set(SegmentationAnalyzer.modalities) == set(
        studies.MODEL_FOR_MODALITY
    )


def test_the_api_has_no_torch() -> None:
    import subprocess
    import sys

    code = (
        "import sys, src.api.app; print(sorted({'torch', 'monai', 'nibabel'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
