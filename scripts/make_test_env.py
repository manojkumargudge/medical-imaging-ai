"""Write .env.test for the integration-test stack: its own ports, names and random passwords.

Never overwrites an existing file. Then:
    docker compose -p aiml02-test --env-file .env.test up -d --wait
    AIML02_INTEGRATION=1 python -m pytest -m integration

Run: python -m scripts.make_test_env
"""

from __future__ import annotations

import argparse
import secrets
from collections.abc import Sequence
from pathlib import Path

from src.service.stack import TEST, TEST_SETTINGS


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(TEST["env_file"]))
    args = parser.parse_args(argv)
    values = {
        **TEST_SETTINGS,
        "ORTHANC_PASSWORD": secrets.token_urlsafe(18),
        "RESULTS_DB_PASSWORD": secrets.token_urlsafe(18),
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as file:  # "x": never overwrite
        file.write("# Integration-test stack only (project aiml02-test). Never commit.\n")
        file.writelines(f"{key}={value}\n" for key, value in values.items())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
