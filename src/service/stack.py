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
    "API_CACHE_DIR": "./results/api-cache-test",
}


def results_db_url(values: Mapping[str, str], host: str = "127.0.0.1") -> str:
    port = values.get("RESULTS_DB_PORT", "5434")
    name = values.get("RESULTS_DB_NAME", DEMO["db_name"])
    return f"postgresql://aiml02:{values['RESULTS_DB_PASSWORD']}@{host}:{port}/{name}"


# The demo studies scripts/upload_demo.py uploads (and scripts/reset_demo.py restores). Their
# StudyInstanceUIDs are derived from these names (src/dicom/convert.py), so the API can tell a
# demo study from anything else without reading the files.
DEMO_NIH_IMAGES = ("00000116_000.png", "00000013_021.png")
DEMO_SPLEEN_CASE = "spleen_14"  # a test case in docs/results_seg.md


def demo_study_uids() -> frozenset[str]:
    from src.dicom.convert import uid_for

    nih = {uid_for("nih", image, "study") for image in DEMO_NIH_IMAGES}
    return frozenset(nih | {uid_for("ct", DEMO_SPLEEN_CASE, "study")})
