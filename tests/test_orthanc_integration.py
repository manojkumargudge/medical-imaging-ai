"""Upload to the real Orthanc from docker-compose.yml, then find the studies there.

Runs only with AIML02_INTEGRATION=1 and the services up (CI job `integration`, or locally after
`docker compose up -d`). Synthetic images only.
"""

import os
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from scripts import upload_demo
from src.dicom import convert
from src.service.orthanc import Orthanc, OrthancError, load_env

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIML02_INTEGRATION") != "1", reason="set AIML02_INTEGRATION=1"
    ),
]


@pytest.fixture
def orthanc() -> Orthanc:
    load_env()
    return Orthanc.from_env()


def test_dx_and_ct_uploads_are_found(orthanc: Orthanc, tmp_path: Path) -> None:
    tag = uuid.uuid4().hex[:8]  # new UIDs on every run
    pixels = (np.add.outer(np.arange(64), np.arange(48) * 3) % 256).astype(np.uint8)
    Image.fromarray(pixels).save(tmp_path / "x.png")
    ds = convert.png_to_dx((tmp_path / "x.png").read_bytes(), f"it-{tag}", source=f"{tag}.png")
    dx = upload_demo.upload_dx(orthanc, ds)

    volume = np.random.default_rng(0).integers(-1000, 1000, (16, 12, 5)).astype(np.int16)
    nib.save(nib.Nifti1Image(volume, np.diag([0.8, 0.8, 2.5, 1])), tmp_path / "ct.nii.gz")
    ct = upload_demo.upload_ct(orthanc, tmp_path / "ct.nii.gz", f"it-{tag}")

    for upload, modality, instances in ((dx, "DX", 1), (ct, "CT", 5)):
        found = orthanc.find_study(upload["study_uid"])
        print(f"{modality}: uploaded {upload['instances']}, studies found {len(found)}")
        assert upload["instances"] == instances and len(found) == 1
        study = orthanc.get_json(f"/studies/{found[0]}")
        assert study["MainDicomTags"]["StudyInstanceUID"] == upload["study_uid"]
        series = orthanc.get_json(f"/studies/{found[0]}/series")
        assert [s["MainDicomTags"]["Modality"] for s in series] == [modality]
        assert len(series[0]["Instances"]) == instances

    # Uploading again stores nothing new.
    assert upload_demo.upload_dx(orthanc, ds)["orthanc"]["Status"] == "AlreadyStored"


def test_auth_is_required(orthanc: Orthanc) -> None:
    with pytest.raises(OrthancError) as error:
        Orthanc(orthanc.url, orthanc.user, "wrong-password").get_json("/system")
    assert error.value.status == 401
