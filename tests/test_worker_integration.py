"""The worker against the real Orthanc and Postgres from docker-compose.yml, with fake models.

Runs only with AIML02_INTEGRATION=1 and the services up. Results go to session-only (TEMP)
tables, so the real results tables are never written to or cleaned up.
"""

import os
import time
import uuid
from pathlib import Path

import pytest
from pydicom.dataset import Dataset
from test_worker import classifier, ct_series, dx, segmenter

from scripts import upload_demo
from src.service.orthanc import Orthanc, load_env
from src.service.store import PostgresStore, Result
from src.service.worker import Worker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIML02_INTEGRATION") != "1", reason="set AIML02_INTEGRATION=1"
    ),
]


def results_db_url() -> str:
    load_env()
    port = os.environ.get("RESULTS_DB_PORT", "5434")
    password = os.environ["RESULTS_DB_PASSWORD"]
    return f"postgresql://aiml02:{password}@127.0.0.1:{port}/aiml02_results"


def tagged_dx(tag: str) -> Dataset:
    ds = dx(int(tag, 16) % 1000)
    for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        value = f"{getattr(ds, name)}.{int(tag, 16) % 10**6}"[:64]
        setattr(ds, name, value)
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def test_worker_processes_new_studies_once(tmp_path: Path) -> None:
    load_env()
    orthanc = Orthanc.from_env()
    store = PostgresStore(results_db_url(), temporary=True)
    store.set_last_seq(int(orthanc.get_json("/changes?last")["Last"]))
    start = store.last_seq()

    tag = uuid.uuid4().hex[:8]
    study_dx = upload_demo.upload_dx(orthanc, tagged_dx(tag))["study_uid"]
    ct = ct_series(tmp_path, f"it{tag}")
    for ds in ct:
        ds.PatientID = f"it-{tag}"
        ds.StudyInstanceUID = f"{ds.StudyInstanceUID}.{int(tag, 16) % 10**6}"[:64]
    study_ct = str(ct[0].StudyInstanceUID)
    for ds in ct:
        orthanc.upload(upload_demo.dicom_bytes(ds))

    worker = Worker(orthanc, store, [classifier(), segmenter()], tmp_path / "results")
    deadline = time.monotonic() + 90  # StableStudy comes 5 s after the last instance
    while time.monotonic() < deadline and len(store.rows()) < 2:
        worker.poll_once()
        time.sleep(1)
    rows = store.rows()
    print(f"rows {len(rows)}, seq {start} -> {store.last_seq()}")
    assert sorted((row["study_uid"], row["modality"], row["status"]) for row in rows) == sorted(
        [(study_dx, "DX", "ok"), (study_ct, "CT", "ok")]
    )
    assert store.last_seq() > start

    # A second pass over the same studies adds nothing, and the database refuses duplicates.
    for orthanc_id in orthanc.find_study(study_dx) + orthanc.find_study(study_ct):
        assert worker.process_study(orthanc_id) == ["skipped"]
    first = rows[0]
    duplicate = Result(
        first["study_uid"], "x", "DX", "m", "c", first["checkpoint_sha256"], "ok", {}, 0.0
    )
    assert store.insert(duplicate) is False
    after = store.rows()
    print(f"rows before rerun {len(rows)}, after {len(after)}")
    assert rows and after and len(after) == len(rows)
    store.close()
