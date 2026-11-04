"""The test stack's settings can never point at the demo stack."""

import pytest

from src.service.orthanc import Orthanc
from src.service.stack import DEMO, TEST, TEST_SETTINGS


def test_test_settings_never_match_the_demo() -> None:
    assert TEST["orthanc_name"] != DEMO["orthanc_name"] and TEST["db_name"] != DEMO["db_name"]
    assert TEST["env_file"] != DEMO["env_file"]
    ports = {TEST_SETTINGS[key] for key in ("ORTHANC_HTTP_PORT", "ORTHANC_DICOM_PORT")}
    ports.add(TEST_SETTINGS["RESULTS_DB_PORT"])
    assert not ports & {"8042", "4242", "5434"}
    assert TEST_SETTINGS["ORTHANC_URL"].endswith(":" + TEST_SETTINGS["ORTHANC_HTTP_PORT"])


def test_wrong_stack_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Orthanc, "get_json", lambda self, path: {"Name": DEMO["orthanc_name"]})
    with pytest.raises(RuntimeError, match="expected 'aiml02-test'"):
        Orthanc("http://127.0.0.1:1", "u", "p").require_name(TEST["orthanc_name"])
