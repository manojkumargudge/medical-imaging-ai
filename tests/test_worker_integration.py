"""The worker against the real Orthanc and Postgres from docker-compose.base.yml, with fake models.

Runs only with AIML02_INTEGRATION=1 against the test stack (.env.test, compose project
aiml02-test), never the demo; Orthanc and database names are checked first. Results go to
session-only (TEMP) tables as well.
"""

import os
import time
import uuid
from pathlib import Path

import pytest
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from stack_helpers import integration_db_url, integration_orthanc
from test_worker import classifier, ct_series, dx, segmenter

from scripts import upload_demo
from src.service.store import PostgresStore, Result
from src.service.worker import Worker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIML02_INTEGRATION") != "1", reason="set AIML02_INTEGRATION=1"
    ),
]


def tagged_dx(tag: str) -> Dataset:
    ds = dx(int(tag, 16) % 1000)
    # New UIDs derived from the tag. (Appending the tag and cutting to 64 characters lost it,
    # because dx()'s UIDs are already 64 long, so two runs could share a study UID.)
    for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        setattr(ds, name, generate_uid(entropy_srcs=[str(getattr(ds, name)), tag]))
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def test_worker_processes_new_studies_once(tmp_path: Path) -> None:
    orthanc = integration_orthanc()
    store = PostgresStore(integration_db_url(), temporary=True)
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
    ours = {study_dx, study_ct}
    while time.monotonic() < deadline and len(ours & {r["study_uid"] for r in store.rows()}) < 2:
        worker.poll_once()
        time.sleep(1)
    # Other studies that became stable meanwhile (other tests' uploads) are processed too.
    rows = [row for row in store.rows() if row["study_uid"] in ours]
    print(
        f"rows {len(rows)} of ours ({len(store.rows())} in all), seq {start} -> {store.last_seq()}"
    )
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
    after = [row for row in store.rows() if row["study_uid"] in ours]
    print(f"rows before rerun {len(rows)}, after {len(after)}")
    assert rows and after and len(after) == len(rows)
    store.close()
