"""A small Orthanc REST client (standard library only) with HTTP basic auth.

Credentials come from the environment (ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD). `load_env`
reads a .env file into missing variables, so the scripts work with the same .env as Docker
Compose; `read_env` reads a file (such as .env.test) without touching the environment. Never
commit .env or .env.test; .env.example lists the names.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8042"


def read_env(path: Path) -> dict[str, str]:
    """KEY=value pairs from a file, without touching the process environment."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_env(path: Path = Path(".env")) -> None:
    """Set variables from a KEY=value file, without replacing ones already set."""
    if path.exists():
        for key, value in read_env(path).items():
            os.environ.setdefault(key, value)


class OrthancError(RuntimeError):
    def __init__(self, status: int, path: str, body: str) -> None:
        super().__init__(f"Orthanc {path}: HTTP {status}: {body[:200]}")
        self.status = status


@dataclass(frozen=True)
class Orthanc:
    url: str
    user: str
    password: str
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> Orthanc:
        return cls.from_values(os.environ)

    @classmethod
    def from_values(cls, values: Mapping[str, str]) -> Orthanc:
        user, password = values.get("ORTHANC_USER"), values.get("ORTHANC_PASSWORD")
        if not user or not password:
            raise ValueError("set ORTHANC_USER and ORTHANC_PASSWORD (see .env.example)")
        return cls(values.get("ORTHANC_URL", DEFAULT_URL).rstrip("/"), user, password)

    def require_name(self, expected: str) -> None:
        """Refuse to go on unless this Orthanc has the expected Name (demo vs test stack)."""
        name = self.get_json("/system")["Name"]
        if name != expected:
            raise RuntimeError(f"{self.url} is Orthanc {name!r}, expected {expected!r}")

    def _request(
        self, method: str, path: str, data: bytes | None = None, content_type: str = ""
    ) -> bytes:
        request = urllib.request.Request(f"{self.url}{path}", data=data, method=method)
        token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as error:
            raise OrthancError(error.code, path, error.read().decode(errors="replace")) from None

    def get_json(self, path: str) -> Any:
        return json.loads(self._request("GET", path))

    def get_bytes(self, path: str) -> bytes:
        return self._request("GET", path)

    def delete(self, path: str) -> None:
        self._request("DELETE", path)

    def post_json(self, path: str, payload: Any) -> Any:
        body = json.dumps(payload).encode()
        return json.loads(self._request("POST", path, body, "application/json"))

    def upload(self, dicom: bytes) -> dict[str, Any]:
        """Store one DICOM file; Status is "Success" or "AlreadyStored"."""
        result: dict[str, Any] = json.loads(
            self._request("POST", "/instances", dicom, "application/dicom")
        )
        return result

    def find_study(self, study_uid: str) -> list[str]:
        """Orthanc IDs of the studies with this StudyInstanceUID."""
        query = {"Level": "Study", "Query": {"StudyInstanceUID": study_uid}}
        found: list[str] = self.post_json("/tools/find", query)
        return found
