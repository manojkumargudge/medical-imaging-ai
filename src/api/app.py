"""The viewer API: worklist, study details, DX image, raw Grad-CAM maps, thresholds, reviews,
uploads and the audit log. The writes are a review and an upload, each with its audit row in one
transaction.

- Orthanc is reached only from here, with the credentials in this process's environment
  (ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD); the browser never calls Orthanc.
- A study ID in a URL must be one Orthanc lists, and a label one of the 14 in
  configs/viewer_thresholds.yaml, before any file path is built; a file must also resolve
  inside RESULTS_DIR. Anything else is 404.
- Orthanc or the results database down: 503 with a message saying which and why.
- Reviews: 409 if the submit token was used already or a newer result exists; 422 for missing
  or invalid fields (reject needs a comment). Reviewer names are typed in and not verified.
- Uploads (src/api/uploads.py): one PNG or DX/CR DICOM file, at most 20 MB (413 above). A file
  already uploaded (same sha256, study still in Orthanc) answers 200 with that study and adds
  nothing. Otherwise the rows are written only after Orthanc has stored the study.

Environment: ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD, RESULTS_DB_URL, RESULTS_DIR (the
worker's output folder, read-only), VIEWER_THRESHOLDS. For education only, not for clinical use.

Run: uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import io
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from uuid import UUID

import yaml
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from pydicom import dcmread

from src.api import studies, uploads
from src.api.db import (
    Database,
    DatabaseUnavailableError,
    NewReview,
    NewUpload,
    PostgresDatabase,
    ReviewConflictError,
    ReviewInvalidError,
)
from src.dicom.render import to_png
from src.service.orthanc import Orthanc, OrthancError
from src.service.stack import DEMO, TEST

T = TypeVar("T")
ORTHANC_TIMEOUT = 10.0


class OrthancUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Services:
    orthanc: Orthanc
    db: Database
    results_dir: Path
    thresholds: dict[str, Any]

    @property
    def labels(self) -> list[str]:
        return list(self.thresholds["labels"])

    def call_orthanc(self, call: Callable[[str], T], path: str) -> T:
        try:
            return call(path)
        except OrthancError as error:
            if error.status in (401, 403):
                raise OrthancUnavailableError(
                    f"Orthanc at {self.orthanc.url} refused the API's credentials"
                ) from None
            raise
        except OSError as error:  # URLError (refused, DNS), timeouts
            reason = getattr(error, "reason", error)
            raise OrthancUnavailableError(
                f"Orthanc is not reachable at {self.orthanc.url}: {reason}"
            ) from None

    def get_json(self, path: str) -> Any:
        return self.call_orthanc(self.orthanc.get_json, path)

    def get_bytes(self, path: str) -> bytes:
        return self.call_orthanc(self.orthanc.get_bytes, path)

    def known_study(self, study_id: str) -> dict[str, Any]:
        """The Orthanc study, or 404 if Orthanc does not list this ID."""
        if study_id not in set(self.get_json("/studies")):
            raise HTTPException(404, "unknown study")
        study: dict[str, Any] = self.get_json(f"/studies/{study_id}")
        return study


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Comment = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]


class ReviewIn(BaseModel):
    """A review as the browser sends it. Names are not verified (there is no login)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    reviewer: Name
    comment: Comment | None = None
    submit_token: UUID
    inference_result_id: int

    @model_validator(mode="after")
    def reject_needs_comment(self) -> ReviewIn:
        if self.decision == "reject" and not self.comment:
            raise ValueError("a comment is required to reject")
        return self


def load_thresholds(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if len(data["labels"]) != 14:
        raise ValueError(f"{path}: expected 14 labels, found {len(data['labels'])}")
    return data


def services_from_env() -> Services:
    url = os.environ.get("RESULTS_DB_URL")
    if not url:
        raise ValueError("set RESULTS_DB_URL")
    return Services(
        replace(Orthanc.from_env(), timeout=ORTHANC_TIMEOUT),
        PostgresDatabase(url),
        Path(os.environ.get("RESULTS_DIR", "results/worker")),
        load_thresholds(
            Path(os.environ.get("VIEWER_THRESHOLDS", "configs/viewer_thresholds.yaml"))
        ),
    )


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def summary(
    study: dict[str, Any],
    series: list[dict[str, Any]],
    results: list[dict[str, Any]],
    review: dict[str, Any] | None,
    uploaded: set[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    study_uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
    modalities = {
        str(x["MainDicomTags"]["Modality"]) for x in series if x["MainDicomTags"].get("Modality")
    }
    times = [studies.orthanc_time(study["LastUpdate"])]
    times += [row["created_at"] for row in results]
    if review:
        times.append(review["created_at"])
    return {
        "id": study["ID"],
        "study_uid": study_uid,
        "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", ""),
        "description": studies.description(study, series),
        "modality": ",".join(sorted(modalities)),
        "status": studies.processing_status(
            bool(study["IsStable"]), modalities, {row["model"]: row["status"] for row in results}
        ),
        "review_status": studies.review_status(review["decision"] if review else None),
        "source": "upload" if {study["ID"], study_uid} & uploaded else "demo",
        "summary": studies.result_summary(results, thresholds["labels"]),
        "updated_at": iso(max(times)),
    }


def validation_errors(error: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": [str(p) for p in item.get("loc", ())], "msg": str(item.get("msg", ""))}
        for item in error.errors()
    ]


def readable(error: RequestValidationError) -> str:
    """FastAPI's validation errors as one sentence for the viewer."""
    parts = []
    for item in validation_errors(error):
        where = ".".join(p for p in item["loc"] if p != "body")
        message = item["msg"].removeprefix("Value error, ")
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts) or "invalid request"


def public_result(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "created_at": iso(row["created_at"])}


def public_review(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return {**row, "created_at": iso(row["created_at"])} if row else None


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="AIML-02 viewer API", description=studies.NOTICE)
    app.state.services = services

    def svc() -> Services:
        if app.state.services is None:
            app.state.services = services_from_env()
        found: Services = app.state.services
        return found

    @app.exception_handler(OrthancUnavailableError)
    async def orthanc_down(_: Request, error: OrthancUnavailableError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, error: RequestValidationError) -> JSONResponse:
        body = {"detail": readable(error), "errors": validation_errors(error)}
        return JSONResponse(body, status_code=422)

    @app.exception_handler(DatabaseUnavailableError)
    async def db_down(_: Request, error: DatabaseUnavailableError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=503)

    @app.get("/api/health")
    def health() -> JSONResponse:
        s = svc()
        checks: dict[str, Any] = {}
        try:
            checks["database"] = {"ok": True, "name": s.db.name()}
        except DatabaseUnavailableError as error:
            checks["database"] = {"ok": False, "error": str(error)}
        try:
            name = s.get_json("/system")["Name"]
            stack = {DEMO["orthanc_name"]: "demo", TEST["orthanc_name"]: "test"}.get(
                name, "unknown"
            )
            checks["orthanc"] = {"ok": True, "name": name, "stack": stack}
        except (OrthancUnavailableError, OrthancError) as error:
            checks["orthanc"] = {"ok": False, "error": str(error)}
        ok = all(check["ok"] for check in checks.values())
        body = {"status": "ok" if ok else "unavailable", **checks, "notice": studies.NOTICE}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/api/studies")
    def list_studies() -> dict[str, Any]:
        s = svc()
        orthanc_studies = s.get_json("/studies?expand")
        series_by_study: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for series in s.get_json("/series?expand"):
            series_by_study[series["ParentStudy"]].append(series)
        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in s.db.newest_results():
            results[row["study_uid"]].append(row)
        reviews, uploaded = s.db.newest_reviews(), s.db.uploaded()
        items = []
        for study in orthanc_studies:
            uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
            items.append(
                summary(
                    study,
                    series_by_study[study["ID"]],
                    results[uid],
                    reviews.get(uid),
                    uploaded,
                    s.thresholds,
                )
            )
        items.sort(key=lambda item: item["updated_at"] or "", reverse=True)
        return {"studies": items, "notice": studies.NOTICE}

    @app.get("/api/studies/{study_id}")
    def study_detail(study_id: str) -> dict[str, Any]:
        s = svc()
        study = s.known_study(study_id)
        series = s.get_json(f"/studies/{study_id}/series")
        uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
        results = s.db.newest_results(uid)
        review = s.db.newest_reviews(uid).get(uid)
        return {
            "study": summary(study, series, results, review, s.db.uploaded(), s.thresholds),
            "series": [
                {
                    "id": x["ID"],
                    "modality": x["MainDicomTags"].get("Modality"),
                    "description": x["MainDicomTags"].get("SeriesDescription", ""),
                    "instances": len(x["Instances"]),
                }
                for x in series
            ],
            "results": [public_result(row) for row in results],
            "review": public_review(review),
            "score_note": studies.SCORE_NOTE,
            "notice": studies.NOTICE,
        }

    @app.get("/api/studies/{study_id}/image")
    def dx_image(study_id: str) -> Response:
        s = svc()
        s.known_study(study_id)
        instances = [
            instance
            for series in s.get_json(f"/studies/{study_id}/series")
            if series["MainDicomTags"].get("Modality") in ("DX", "CR")
            for instance in series["Instances"]
        ]
        if not instances:
            raise HTTPException(404, "this study has no DX or CR image")
        ds = dcmread(io.BytesIO(s.get_bytes(f"/instances/{instances[0]}/file")))
        try:
            png = to_png(ds)
        except ValueError as error:
            raise HTTPException(422, f"cannot show this image: {error}") from None
        return Response(png, media_type="image/png")

    @app.get("/api/studies/{study_id}/cam/{label}")
    def raw_cam(study_id: str, label: str) -> Response:
        s = svc()
        if label not in s.labels:
            raise HTTPException(404, "unknown label")
        study = s.known_study(study_id)
        uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
        rows = [
            row
            for row in s.db.newest_results(uid)
            if row["model"] == studies.CLASSIFIER and row["status"] == "ok"
        ]
        if not rows:
            raise HTTPException(404, "no classifier result for this study yet")
        entry = rows[0]["results"]["instances"][0]["labels"].get(label, {})
        if "cam" not in entry:
            raise HTTPException(404, "this result has no raw map (made before maps were stored)")
        root = s.results_dir.resolve()
        path = (root / str(entry["cam"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "raw map file not found")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/api/studies/{study_id}/reviews")
    def list_reviews(study_id: str) -> dict[str, Any]:
        s = svc()
        uid = str(s.known_study(study_id)["MainDicomTags"].get("StudyInstanceUID", ""))
        rows = s.db.reviews(uid)
        return {"reviews": [public_review(row) for row in rows], "notice": studies.NOTICE}

    @app.post("/api/studies/{study_id}/reviews", status_code=201)
    def add_review(study_id: str, review: ReviewIn) -> dict[str, Any]:
        s = svc()
        uid = str(s.known_study(study_id)["MainDicomTags"].get("StudyInstanceUID", ""))
        try:
            row = s.db.add_review(
                NewReview(
                    study_uid=uid,
                    inference_result_id=review.inference_result_id,
                    decision=review.decision,
                    reviewer=review.reviewer,
                    comment=review.comment or None,
                    submit_token=review.submit_token,
                )
            )
        except ReviewConflictError as error:
            raise HTTPException(409, str(error)) from None
        except ReviewInvalidError as error:
            raise HTTPException(422, str(error)) from None
        return {"review": public_review(row)}

    @app.post("/api/uploads", status_code=201)
    def upload(file: UploadFile, uploader: Annotated[str, Form()]) -> JSONResponse:
        s = svc()
        name = uploader.strip()
        if not name or len(name) > 100:
            raise HTTPException(422, "uploader: a name of 1 to 100 characters is required")
        data = file.file.read(uploads.MAX_BYTES + 1)
        if len(data) > uploads.MAX_BYTES:
            raise HTTPException(413, "the file is larger than 20 MB")
        if not data:
            raise HTTPException(422, "the file is empty")
        sha256 = hashlib.sha256(data).hexdigest()

        earlier = s.db.find_upload(sha256)
        if earlier and earlier["orthanc_study_id"] in set(s.get_json("/studies")):
            body = {
                "duplicate": True,
                "upload_id": earlier["id"],
                "study_id": earlier["orthanc_study_id"],
                "message": "this file was uploaded before; nothing new was added",
            }
            return JSONResponse(body, status_code=200)

        try:
            checked = uploads.check(data)
        except uploads.UploadRejectedError as error:
            raise HTTPException(422, str(error)) from None
        upload_id = s.db.reserve_upload_id()
        ds = uploads.to_dicom(data, checked, upload_id, sha256)
        dicom = uploads.dicom_bytes(ds)
        try:
            stored = s.call_orthanc(lambda _: s.orthanc.upload(dicom), "/instances")
        except OrthancError as error:
            raise HTTPException(422, f"Orthanc refused the file (HTTP {error.status})") from None
        row = s.db.add_upload(
            NewUpload(
                id=upload_id,
                uploader=name,
                kind=checked.kind,
                filename=Path(file.filename or "").name[:255],
                size_bytes=len(data),
                sha256=sha256,
                study_uid=str(ds.StudyInstanceUID),
                orthanc_study_id=str(stored["ParentStudy"]),
            )
        )
        body = {
            "duplicate": False,
            "upload_id": row["id"],
            "study_id": row["orthanc_study_id"],
            "kind": row["kind"],
            "notice": studies.NOTICE,
            "scope_warning": uploads.SCOPE_WARNING,
        }
        return JSONResponse(body, status_code=201)

    @app.get("/api/audit")
    def audit(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        rows, total = svc().db.audit(page_size, (page - 1) * page_size)
        return {
            "entries": [{**row, "created_at": iso(row["created_at"])} for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "notice": studies.NOTICE,
        }

    @app.get("/api/config/thresholds")
    def thresholds() -> dict[str, Any]:
        data = svc().thresholds
        return {
            "source": data.get("source"),
            "labels": data["labels"],
            "note": studies.SCORE_NOTE,
            "notice": studies.NOTICE,
        }

    return app
