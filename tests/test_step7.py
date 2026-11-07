"""Step 7 packaging: the one-command demo stack, healthchecks, ports, pins, the worker heartbeat,
the timing summary and the notice. No Docker needed."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from test_api import THRESHOLDS, FakeDb, FakeOrthanc

from src.api.app import Services, create_app, load_thresholds
from src.api.studies import NOTICE
from src.service import timing
from src.service.store import MemoryStore
from src.service.worker import Worker

REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "docker-compose.base.yml",
    "docker-compose.worker.yml",
    "docker-compose.api.yml",
    "docker-compose.frontend.yml",
]
DOCKERFILES = {"worker": "Dockerfile", "api": "Dockerfile.api", "frontend": "frontend/Dockerfile"}


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((REPO_ROOT / name).read_text(encoding="utf-8"))
    return data


def services() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for name in FILES:
        for service, spec in load(name)["services"].items():
            merged.setdefault(service, {}).update(spec)
    return merged


def test_compose_yaml_is_the_whole_demo_stack() -> None:
    compose = load("compose.yaml")
    assert compose["name"] == "aiml02"
    assert compose["include"] == [{"path": FILES}]
    assert set(services()) == {"orthanc", "results-db", "worker", "api", "frontend"}


def test_every_service_has_a_healthcheck_and_waits_for_what_it_needs() -> None:
    found = services()
    for name, spec in found.items():
        dockerfile = DOCKERFILES.get(name)
        in_image = dockerfile and "HEALTHCHECK" in (REPO_ROOT / dockerfile).read_text("utf-8")
        assert "healthcheck" in spec or in_image, f"{name} has no healthcheck"
    needs = {
        "worker": {"orthanc", "results-db"},
        "api": {"orthanc", "results-db"},
        "frontend": {"api"},
    }
    for name, wanted in needs.items():
        depends = found[name]["depends_on"]
        assert set(depends) == wanted
        assert all(v["condition"] == "service_healthy" for v in depends.values())


def test_worker_and_api_use_the_same_database_name() -> None:
    """The test stack's database is aiml02_results_test (.env.test RESULTS_DB_NAME); a fixed
    name in the worker's URL left it unable to connect there."""
    found = services()
    urls = {name: found[name]["environment"]["RESULTS_DB_URL"] for name in ("worker", "api")}
    print(urls)
    assert all(url.endswith("/${RESULTS_DB_NAME:-aiml02_results}") for url in urls.values())


def test_ports_are_local_only_and_images_pinned() -> None:
    ports = [port for spec in services().values() for port in spec.get("ports", [])]
    print(ports)
    assert len(ports) == 5 and all(port.startswith("127.0.0.1:") for port in ports)
    for name, spec in services().items():
        if "build" in spec:
            dockerfile = (REPO_ROOT / DOCKERFILES[name]).read_text(encoding="utf-8")
            images = [
                line.split()[1] for line in dockerfile.splitlines() if line.startswith("FROM")
            ]
        else:
            images = [spec["image"]]
        # An exact release: x.y.z, or PostgreSQL's x.y (16.14); never a floating tag like :16.
        pinned = r":\d+\.\d+(\.\d+)?(-|$)"
        assert images and all(re.search(pinned, i) for i in images), (name, images)


def test_the_worker_touches_its_heartbeat_after_each_poll(tmp_path: Path) -> None:
    worker = Worker(FakeOrthanc(), MemoryStore(), [object()], tmp_path, log=lambda _: None)  # type: ignore[list-item]

    def stop() -> int:
        raise KeyboardInterrupt  # ends the loop after one poll

    worker.poll_once = stop  # type: ignore[method-assign,assignment]
    heartbeat = tmp_path / "heartbeat"
    with pytest.raises(KeyboardInterrupt):
        worker.run(0, heartbeat)
    assert heartbeat.exists()


def test_timing_drops_the_warm_up_runs() -> None:
    summary = timing.summarise([60.0, 20.0, 30.0, 25.0, 22.0, 40.0], warm_up=1)
    assert summary == {
        "runs": [60.0, 20.0, 30.0, 25.0, 22.0, 40.0],
        "warm_runs": 5,
        "median": 25.0,
        "min": 20.0,
        "max": 40.0,
    }
    with pytest.raises(ValueError, match="leave none"):
        timing.summarise([1.0], warm_up=1)


def test_the_notice_is_in_the_readme_health_and_openapi(tmp_path: Path) -> None:
    assert NOTICE == "For education only. Not for clinical use."
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[0] == f"**{NOTICE}**"
    client = TestClient(
        create_app(Services(FakeOrthanc(), FakeDb(), tmp_path, load_thresholds(THRESHOLDS)))
    )
    assert client.get("/api/health").json()["notice"] == NOTICE
    assert NOTICE in client.get("/openapi.json").json()["info"]["description"]


def test_docs_openapi_json_matches_the_api() -> None:
    from scripts import export_openapi

    stored = (REPO_ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert stored == export_openapi.document(), "run python -m scripts.export_openapi"
