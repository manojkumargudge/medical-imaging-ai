"""The two Docker Compose stacks: the demo (.env, project aiml02) and the integration-test stack
(.env.test, project aiml02-test). They have separate volumes, ports, Orthanc names and database
names. Integration tests check the names, so they can never write to the demo stack.
"""

from __future__ import annotations

from collections.abc import Mapping

DEMO = {"orthanc_name": "aiml02", "db_name": "aiml02_results", "env_file": ".env"}
TEST = {"orthanc_name": "aiml02-test", "db_name": "aiml02_results_test", "env_file": ".env.test"}

TEST_SETTINGS = {
    "ORTHANC_URL": "http://127.0.0.1:18042",
    "ORTHANC_HTTP_PORT": "18042",
    "ORTHANC_DICOM_PORT": "14242",
    "ORTHANC_NAME": TEST["orthanc_name"],
    "ORTHANC_USER": "aiml02-test",
    "RESULTS_DB_PORT": "15434",
    "RESULTS_DB_NAME": TEST["db_name"],
    "API_PORT": "18000",
    "FRONTEND_PORT": "18080",
    "WORKER_RESULTS_DIR": "./results/worker-test",
}


def results_db_url(values: Mapping[str, str], host: str = "127.0.0.1") -> str:
    port = values.get("RESULTS_DB_PORT", "5434")
    name = values.get("RESULTS_DB_NAME", DEMO["db_name"])
    return f"postgresql://aiml02:{values['RESULTS_DB_PASSWORD']}@{host}:{port}/{name}"
