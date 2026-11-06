"""The Orthanc client against a local fake server (no Docker), and the demo upload helpers."""

import base64
import io
import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from pydicom import dcmread

from scripts import upload_demo
from src.dicom import convert
from src.service.orthanc import Orthanc, OrthancError, load_env

USER, PASSWORD = "user", "secret"


class FakeOrthanc(BaseHTTPRequestHandler):
    stored: list[bytes] = []

    def log_message(self, *args: Any) -> None:
        pass

    def _authorized(self) -> bool:
        token = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
        if self.headers.get("Authorization") != f"Basic {token}":
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self._authorized():
            self._send(200, {"Version": "fake"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/instances":
            assert self.headers["Content-Type"] == "application/dicom"
            self.stored.append(body)
            self._send(200, {"ID": "i1", "ParentStudy": "s1", "Status": "Success"})
        elif self.path == "/tools/find":
            query = json.loads(body)["Query"]["StudyInstanceUID"]
            uids = {str(dcmread(io.BytesIO(data)).StudyInstanceUID) for data in self.stored}
            self._send(200, ["s1"] if query in uids else [])
        else:
            self._send(404, {})


@pytest.fixture
def server() -> Iterator[str]:
    FakeOrthanc.stored = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeOrthanc)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def png_file(path: Path) -> Path:
    pixels = (np.add.outer(np.arange(40) * 5, np.arange(30) * 2) % 256).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def test_upload_then_find_with_basic_auth(server: str, tmp_path: Path) -> None:
    orthanc = Orthanc(server, USER, PASSWORD)
    ds = convert.png_to_dx(png_file(tmp_path / "x.png").read_bytes(), "p1", source="x.png")
    result = upload_demo.upload_dx(orthanc, ds)
    assert result["orthanc"]["Status"] == "Success"
    assert orthanc.find_study(result["study_uid"]) == ["s1"]
    assert orthanc.find_study("1.2.3") == []
    stored = dcmread(io.BytesIO(FakeOrthanc.stored[0]))
    assert stored.Modality == "DX" and stored.file_meta.TransferSyntaxUID


def test_ct_series_upload_sends_every_slice(server: str, tmp_path: Path) -> None:
    volume = np.arange(6 * 5 * 4, dtype=np.int16).reshape(6, 5, 4)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "ct.nii.gz")
    result = upload_demo.upload_ct(Orthanc(server, USER, PASSWORD), tmp_path / "ct.nii.gz", "c1")
    assert result["instances"] == len(FakeOrthanc.stored) == 4
    assert {dcmread(io.BytesIO(data)).Modality for data in FakeOrthanc.stored} == {"CT"}


def test_wrong_password_raises(server: str) -> None:
    with pytest.raises(OrthancError) as error:
        Orthanc(server, USER, "wrong").get_json("/system")
    assert error.value.status == 401


def test_credentials_must_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORTHANC_USER", raising=False)
    monkeypatch.delenv("ORTHANC_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="ORTHANC_USER"):
        Orthanc.from_env()


def test_env_file_does_not_replace_set_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIML02_TEST_A", "kept")
    monkeypatch.delenv("AIML02_TEST_B", raising=False)
    (tmp_path / ".env").write_text("# comment\nAIML02_TEST_A=new\nAIML02_TEST_B='set'\n")
    load_env(tmp_path / ".env")
    assert os.environ["AIML02_TEST_A"] == "kept" and os.environ["AIML02_TEST_B"] == "set"
    monkeypatch.delenv("AIML02_TEST_B")


def test_compose_binds_localhost_and_keeps_secrets_out() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.base.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:${ORTHANC_HTTP_PORT:-8042}:8042"' in compose
    assert '"127.0.0.1:${ORTHANC_DICOM_PORT:-4242}:4242"' in compose
    assert '"127.0.0.1:${RESULTS_DB_PORT:-5434}:5432"' in compose
    assert 'ORTHANC__AUTHENTICATION_ENABLED: "true"' in compose
    assert "change-me" not in compose
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8")
