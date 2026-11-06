"""The API image and Compose service: pins, what is copied, ports, mounts and credentials."""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

from src.service.stack import TEST_SETTINGS

REPO_ROOT = Path(__file__).resolve().parents[1]


def pins(lines: list[str]) -> dict[str, str]:
    found = {}
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.-]+)(\[[a-z,]+\])?==([^\s;]+)", line.strip())
        if match:
            found[match[1].lower()] = match[3]
    return found


def test_requirements_api_pins_match_pyproject() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = pins(project["dependencies"] + project["optional-dependencies"]["api"])
    image = pins((REPO_ROOT / "requirements-api.txt").read_text(encoding="utf-8").splitlines())
    print(f"requirements-api.txt: {image}")
    assert image and {"fastapi", "uvicorn"} <= set(image)
    assert {name: declared.get(name) for name in image} == image
    assert not {"torch", "torchvision", "monai", "nibabel"} & set(image)


def test_the_api_image_copies_code_and_configs_only() -> None:
    lines = (REPO_ROOT / "Dockerfile.api").read_text(encoding="utf-8").splitlines()
    copied = {line.split()[1] for line in lines if line.startswith("COPY")}
    assert copied == {"requirements-api.txt", "src", "configs"}
    ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ignore[0] == "*" and "!requirements-api.txt" in ignore and "**/*.pt" in ignore
    assert any(line.startswith("USER api") for line in lines)


def test_the_api_service_is_local_only_and_holds_the_orthanc_credentials() -> None:
    compose: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "docker-compose.api.yml").read_text(encoding="utf-8")
    )
    api = compose["services"]["api"]
    assert api["ports"] == ["127.0.0.1:${API_PORT:-8000}:8000"]
    assert api["volumes"] == ["${WORKER_RESULTS_DIR:-./results/worker}:/results:ro"]
    environment = api["environment"]
    # Credentials come from the env file at run time, never written into the file.
    assert environment["ORTHANC_USER"] == "${ORTHANC_USER}"
    assert environment["ORTHANC_PASSWORD"] == "${ORTHANC_PASSWORD}"
    assert "${RESULTS_DB_PASSWORD}" in environment["RESULTS_DB_URL"]
    assert environment["ORTHANC_URL"] == "http://orthanc:8042"


def test_the_frontend_service_is_local_only_and_has_no_credentials() -> None:
    compose: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "docker-compose.frontend.yml").read_text(encoding="utf-8")
    )
    frontend = compose["services"]["frontend"]
    assert frontend["ports"] == ["127.0.0.1:${FRONTEND_PORT:-8080}:8080"]
    assert "environment" not in frontend and "volumes" not in frontend
    nginx = (REPO_ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    assert "proxy_pass http://api:8000;" in nginx and "8042" not in nginx
    assert "Content-Security-Policy" in nginx


def test_the_frontend_image_pins_its_base_images() -> None:
    lines = (REPO_ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8").splitlines()
    images = [line.split()[1] for line in lines if line.startswith("FROM")]
    assert images and all(re.search(r":\d+\.\d+\.\d+-", image) for image in images)
    package = (REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8")
    assert not re.search(r'"[\^~]', package)  # exact versions only
    # One Node version: .nvmrc (CI reads it), the build image, and package.json engines.
    node = (REPO_ROOT / "frontend" / ".nvmrc").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", node)
    assert images[0].startswith(f"node:{node}-")
    major = int(node.split(".")[0])
    assert f'"node": ">={node} <{major + 1}"' in package


def test_the_test_stack_api_has_its_own_port_and_results_folder() -> None:
    assert TEST_SETTINGS["API_PORT"] != "8000"
    assert TEST_SETTINGS["FRONTEND_PORT"] != "8080"
    assert TEST_SETTINGS["WORKER_RESULTS_DIR"] != "./results/worker"
