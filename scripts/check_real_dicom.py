"""Run the real scanner chest X-rays (scripts/fetch_real_dicom.py) through the TEST stack.

For each file: upload to the test Orthanc, wait for the worker's result row, then check
- the worker's 14 scores against the same pipeline run here, outside Docker
  (ClassifierAnalyzer on this machine), within 1e-6 (as scripts/check_worker.py), so the
  container decoded the same pixels;
- that a raw Grad-CAM PNG exists for every label and that re-blending it gives the stored
  overlay within 1 grey level (scripts/check_worker.check_raw_cams);
- that the viewer API serves the image and a raw map as PNG, and labels the study "unknown"
  (so the viewer shows the model-scope warning).

Only the test stack (.env.test) is used: the Orthanc name and the database name are checked
first, and the demo stack is never contacted. RG1_J2KR and RG1_J2KI share all their UIDs (two
encodings of one image), so the J2KI copy gets new Study/Series/SOP UIDs derived from its hash;
its pixel data is sent unchanged. The checkpoint path is read from .env (CLASSIFIER_CHECKPOINT).
Writes reports/real_dicom_check.json. For education only, not for clinical use.

Run: python -m scripts.check_real_dicom
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import psycopg
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from scripts.check_worker import DX_TOLERANCE, check_raw_cams
from scripts.fetch_real_dicom import FILES, OUT
from src.service.analyzers import ClassifierAnalyzer
from src.service.orthanc import Orthanc, read_env
from src.service.stack import TEST, results_db_url

REPORT = Path("reports/real_dicom_check.json")
RESULTS_DIR = Path("results/worker-test")
CLASSIFIER = "densenet121-chestxray14"
NEW_UIDS = {"RG1_J2KI.dcm"}  # same UIDs as RG1_J2KR.dcm
TAGS = (
    "Modality",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "Rows",
    "Columns",
    "Manufacturer",
    "ViewPosition",
)


def with_new_uids(ds: Dataset, seed: str) -> Dataset:
    """New Study/Series/SOP Instance UIDs derived from `seed`; everything else unchanged."""
    for keyword in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        setattr(ds, keyword, generate_uid(entropy_srcs=[seed, keyword]))
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def as_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_db_url(values: dict[str, str]) -> str:
    url = results_db_url(values)
    with psycopg.connect(url) as connection:
        row = connection.execute("SELECT current_database()").fetchone()
    if row is None or row[0] != TEST["db_name"]:
        raise SystemExit(f"{row} is not the test database {TEST['db_name']!r}")
    return url


def wait_for_result(url: str, study_uid: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    with psycopg.connect(url, autocommit=True) as connection:
        while time.monotonic() < deadline:
            row = connection.execute(
                "SELECT status, results, seconds FROM inference_results "
                "WHERE study_uid = %s AND model = %s ORDER BY id DESC LIMIT 1",
                (study_uid, CLASSIFIER),
            ).fetchone()
            if row:
                return {"status": row[0], "results": row[1], "seconds": float(row[2])}
            time.sleep(5)
    raise SystemExit(f"no worker result for {study_uid} after {timeout:.0f} s")


def api_get(port: str, path: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=120) as response:  # noqa: S310
        return response.status, response.headers.get_content_type(), response.read()


def check_api(port: str, study_uid: str, label: str) -> dict[str, Any]:
    _, _, body = api_get(port, "/api/studies")
    study = next(s for s in json.loads(body)["studies"] if s["study_uid"] == study_uid)
    image = api_get(port, f"/api/studies/{study['id']}/image")
    cam = api_get(port, f"/api/studies/{study['id']}/cam/{label}")
    return {
        "study_id": study["id"],
        "status": study["status"],
        "source": study["source"],
        "image": [image[0], image[1], len(image[2])],
        "cam": [cam[0], cam[1], len(cam[2])],
        "ok": study["source"] == "unknown"
        and image[:2] == (200, "image/png")
        and cam[:2] == (200, "image/png"),
    }


def main() -> None:
    values = read_env(Path(TEST["env_file"]))
    orthanc = Orthanc.from_values(values)
    orthanc.require_name(TEST["orthanc_name"])
    url = test_db_url(values)
    checkpoint = Path(read_env(Path(".env"))["CLASSIFIER_CHECKPOINT"])
    local = ClassifierAnalyzer.from_checkpoint(checkpoint)
    report: list[dict[str, Any]] = []
    for item in FILES:
        path = OUT / item.name
        data = path.read_bytes()
        ds: Dataset = dcmread(io.BytesIO(data))
        if item.name in NEW_UIDS:
            ds = with_new_uids(ds, hashlib.sha256(data).hexdigest())
            data = as_bytes(ds)
        entry: dict[str, Any] = {
            "file": item.name,
            "source": item.source,
            "license": item.license,
            "transfer_syntax": f"{ds.file_meta.TransferSyntaxUID} "
            f"({ds.file_meta.TransferSyntaxUID.name})",
            **{tag: str(ds.get(tag, "")) for tag in TAGS},
            "new_uids": item.name in NEW_UIDS,
        }
        print(
            f"\n{item.name}: {entry['Modality']} {entry['PhotometricInterpretation']} "
            f"{entry['transfer_syntax']} bits {entry['BitsStored']}"
        )
        orthanc.upload(data)
        uid = str(ds.StudyInstanceUID)
        row = wait_for_result(url, uid)
        entry["worker_status"], entry["worker_seconds"] = row["status"], row["seconds"]
        if row["status"] != "ok":
            entry["error"] = row["results"].get("error")
            print(f"  worker FAILED: {entry['error']}")
            report.append(entry)
            continue
        labels = row["results"]["instances"][0]["labels"]
        with tempfile.TemporaryDirectory() as tmp:
            mine = local._instance(ds, Path(tmp), Path(tmp))["labels"]
        diff = max(abs(labels[k]["score"] - mine[k]["score"]) for k in labels)
        top = sorted(labels, key=lambda k: labels[k]["score"], reverse=True)[:3]
        entry["scores"] = {k: round(v["score"], 6) for k, v in labels.items()}
        entry["top3"] = top
        entry["above_threshold"] = sorted(k for k, v in labels.items() if v["above_threshold"])
        entry["max_score_diff_vs_local"] = diff
        entry["scores_match_local"] = bool(diff <= DX_TOLERANCE)
        entry["cams_ok"] = check_raw_cams(labels, RESULTS_DIR, ds, item.name)
        entry["api"] = check_api(values["API_PORT"], uid, top[0])
        shown = [(k, round(labels[k]["score"], 3)) for k in top]
        print(f"  worker ok in {row['seconds']:.1f} s; top3 {shown}")
        print(f"  max |worker - local| score diff {diff:.2e} (tolerance {DX_TOLERANCE})")
        print(f"  api {entry['api']}")
        report.append(entry)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    ok = all(
        e["worker_status"] == "ok" and e["scores_match_local"] and e["cams_ok"] and e["api"]["ok"]
        for e in report
    )
    print(f"\nwrote {REPORT}; {len(report)} files; all checks {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
