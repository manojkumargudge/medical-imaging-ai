"""Reset the demo stack to the scripts/upload_demo.py studies only.

Deletes, after showing the counts and asking for confirmation:

- every study in the demo Orthanc
- every row in the demo results table `inference_results`
- the worker's output files under results/

It keeps the viewer's tables (reviews, audit_log, uploads), and checks their row counts are the
same afterwards. A review refers to its result row, so the reset refuses, before deleting
anything, while any review exists.

Then it uploads the demo studies again with scripts/upload_demo.py.

The worker's place in Orthanc's change log is kept. Older changes refer to deleted studies,
which the worker skips, and the new uploads are processed as usual.

Only the demo stack: Orthanc must report the demo name and the database must be the demo
database. Stop the worker first, and start it again afterwards:
    docker compose -f docker-compose.yml -f docker-compose.worker.yml stop worker
    python -m scripts.reset_demo
    docker compose -f docker-compose.yml -f docker-compose.worker.yml start worker
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from scripts.upload_demo import upload_all
from src.service.orthanc import Orthanc, load_env
from src.service.stack import DEMO, results_db_url
from src.service.store import PostgresStore

RESULTS_ROOT = Path("results")


class ResultStore(Protocol):
    def database(self) -> str: ...

    def count_results(self) -> int: ...

    def delete_results(self) -> int: ...

    def count_kept(self) -> dict[str, int]: ...

    def reviewed_results(self) -> int: ...


def output_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.rglob("*") if path.is_file())


def check_results_dir(results_dir: Path, allowed_root: Path) -> None:
    resolved, root = results_dir.resolve(), allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{results_dir} is not a folder inside {allowed_root}; not deleting")


def reset(
    orthanc: Orthanc,
    store: ResultStore,
    results_dir: Path,
    upload: Callable[[Orthanc], list[str]],
    confirm: Callable[[str], bool],
    allowed_root: Path = RESULTS_ROOT,
) -> bool:
    """Returns False (and deletes nothing) if the confirmation is refused."""
    orthanc.require_name(DEMO["orthanc_name"])
    if store.database() != DEMO["db_name"]:
        raise RuntimeError(f"database {store.database()!r} is not {DEMO['db_name']!r}")
    check_results_dir(results_dir, allowed_root)
    reviewed = store.reviewed_results()
    if reviewed:
        raise RuntimeError(
            f"{reviewed} result rows have reviews, which are kept and refer to them; not resetting"
        )

    studies: list[str] = orthanc.get_json("/studies")
    rows, files = store.count_results(), output_files(results_dir) if results_dir.exists() else []
    summary = (
        f"Demo Orthanc {orthanc.url}: {len(studies)} studies; results database "
        f"{DEMO['db_name']}: {rows} rows; {results_dir}: {len(files)} files"
    )
    kept = store.count_kept()
    print(f"before: {summary}; kept: {kept}")
    if not confirm(f"Delete all of this and re-upload the demo studies? {summary}"):
        print("not confirmed; nothing deleted")
        return False

    for study in studies:
        orthanc.delete(f"/studies/{study}")
    deleted_rows = store.delete_results()
    if results_dir.exists():
        for child in results_dir.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    print(f"deleted {len(studies)} studies, {deleted_rows} rows, {len(files)} files")
    kept_after = store.count_kept()
    print(f"kept tables: before {kept}, after {kept_after}")
    if kept_after != kept:
        raise RuntimeError(f"kept tables changed: before {kept}, after {kept_after}")

    uploaded = upload(orthanc)
    after: list[str] = orthanc.get_json("/studies")
    print(f"after: before {len(studies)} studies, after {len(after)} (uploaded {len(uploaded)})")
    if not uploaded or len(after) != len(uploaded):
        raise RuntimeError(f"expected {len(uploaded)} studies after the reset, found {len(after)}")
    return True


def ask(question: str) -> bool:
    return input(f"{question}\nType 'wipe' to go on: ").strip() == "wipe"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="skip the question (already asked)")
    args = parser.parse_args(argv)
    load_env()
    results_dir = Path(os.environ.get("WORKER_RESULTS_DIR", "results/worker"))
    store = PostgresStore(results_db_url(os.environ))
    try:
        reset(
            Orthanc.from_env(),
            store,
            results_dir,
            upload_all,
            (lambda _: True) if args.yes else ask,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
