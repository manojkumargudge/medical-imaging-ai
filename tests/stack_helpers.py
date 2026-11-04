"""The integration tests' only way to the services: the test stack, checked by name.

`integration_orthanc()` reads .env.test (never .env or the process environment) and refuses to
return unless Orthanc reports the test stack's name; `integration_db_url()` likewise checks the
database name. The demo stack (.env) is never touched by tests.
"""

from pathlib import Path

from src.service.orthanc import Orthanc, read_env
from src.service.stack import TEST, results_db_url

REPO_ROOT = Path(__file__).resolve().parents[1]


def integration_values() -> dict[str, str]:
    return read_env(REPO_ROOT / TEST["env_file"])


def integration_orthanc() -> Orthanc:
    orthanc = Orthanc.from_values(integration_values())
    orthanc.require_name(TEST["orthanc_name"])
    return orthanc


def integration_db_url() -> str:
    import psycopg

    url = results_db_url(integration_values())
    with psycopg.connect(url) as connection:
        row = connection.execute("SELECT current_database()").fetchone()
    if row is None or row[0] != TEST["db_name"]:
        raise RuntimeError(f"{row} is not the test database {TEST['db_name']!r}")
    return url
