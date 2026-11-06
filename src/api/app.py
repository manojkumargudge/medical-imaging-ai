"""The viewer API: worklist, study details, DX image, raw Grad-CAM maps, thresholds. Read-only.

- Orthanc is reached only from here, with the credentials in this process's environment
  (ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD); the browser never calls Orthanc.
- A study ID in a URL must be one Orthanc lists, and a label one of the 14 in
  configs/viewer_thresholds.yaml, before any file path is built; a file must also resolve
  inside RESULTS_DIR. Anything else is 404.
- Orthanc or the results database down: 503 with a message saying which and why.

Environment: ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD, RESULTS_DB_URL, RESULTS_DIR (the
worker's output folder, read-only), VIEWER_THRESHOLDS. For education only, not for clinical use.

Run: uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydicom import dcmread

from src.api import studies
from src.api.db import Database, DatabaseUnavailableError, PostgresDatabase
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
    modalities: set[str],
    results: list[dict[str, Any]],
    review: dict[str, Any] | None,
    uploaded: set[str],
) -> dict[str, Any]:
    study_uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
    times = [studies.orthanc_time(study["LastUpdate"])]
    times += [row["created_at"] for row in results]
    if review:
        times.append(review["created_at"])
    return {
        "id": study["ID"],
        "study_uid": study_uid,
        "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", ""),
        "modality": ",".join(sorted(modalities)),
        "status": studies.processing_status(
            bool(study["IsStable"]), modalities, {row["model"]: row["status"] for row in results}
        ),
        "review_status": studies.review_status(review["decision"] if review else None),
        "source": "upload" if {study["ID"], study_uid} & uploaded else "demo",
        "updated_at": iso(max(times)),
    }


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
        modalities: dict[str, set[str]] = defaultdict(set)
        for series in s.get_json("/series?expand"):
            modality = series["MainDicomTags"].get("Modality")
            if modality:
                modalities[series["ParentStudy"]].add(str(modality))
        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in s.db.newest_results():
            results[row["study_uid"]].append(row)
        reviews, uploaded = s.db.newest_reviews(), s.db.uploaded()
        items = []
        for study in orthanc_studies:
            uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
            items.append(
                summary(study, modalities[study["ID"]], results[uid], reviews.get(uid), uploaded)
            )
        items.sort(key=lambda item: item["updated_at"] or "", reverse=True)
        return {"studies": items, "notice": studies.NOTICE}

    @app.get("/api/studies/{study_id}")
    def study_detail(study_id: str) -> dict[str, Any]:
        s = svc()
        study = s.known_study(study_id)
        series = s.get_json(f"/studies/{study_id}/series")
        modalities = {
            str(x["MainDicomTags"]["Modality"]) for x in series if "Modality" in x["MainDicomTags"]
        }
        uid = str(study["MainDicomTags"].get("StudyInstanceUID", ""))
        results = s.db.newest_results(uid)
        review = s.db.newest_reviews(uid).get(uid)
        return {
            "study": summary(study, modalities, results, review, s.db.uploaded()),
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
