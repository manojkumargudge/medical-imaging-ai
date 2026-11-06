"""Apply the numbered SQL files in migrations/ to a results database, in order, once each.

Files are named `NNN_name.sql`, numbered from 001 without gaps. `schema_migrations` records each
applied file with its sha256. An applied file that has changed since is an error: add a new file
instead of editing an old one. All pending files run in one transaction, each with its
schema_migrations row (each file in a savepoint): if any file fails, none of them is applied.

001 is the worker's schema from before migrations existed (IF NOT EXISTS), so it is a no-op on a
database the worker already created. Migrations only add: nothing here drops or deletes data.

The command line reads the database settings from an env file (.env for the demo stack,
.env.test for the test stack), prints the row count of every table before and after, and checks
that no count changed. For education only, not for clinical use.

Run: python -m src.service.migrate [--env-file .env.test] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.service.orthanc import read_env
from src.service.stack import DEMO, TEST, results_db_url

if TYPE_CHECKING:
    import psycopg

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
FILE_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

TRACKING = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    sha256: str


def load(folder: Path = MIGRATIONS) -> list[Migration]:
    """The migration files in version order; the numbers must run 1, 2, 3, ... without gaps."""
    migrations = []
    for path in sorted(folder.glob("*.sql")):
        match = FILE_NAME.match(path.name)
        if not match:
            raise ValueError(f"{path.name}: expected NNN_name.sql")
        data = path.read_bytes()
        migrations.append(
            Migration(
                int(match[1]), path.name, data.decode("utf-8"), hashlib.sha256(data).hexdigest()
            )
        )
    versions = [migration.version for migration in migrations]
    if versions != list(range(1, len(migrations) + 1)):
        raise ValueError(f"migration numbers must be 1..{len(migrations)} without gaps: {versions}")
    return migrations


def to_apply(
    applied: Mapping[int, tuple[str, str]], migrations: Sequence[Migration]
) -> list[Migration]:
    """The migrations not applied yet. applied maps version -> (name, sha256) from the database."""
    known = {migration.version: migration for migration in migrations}
    for version, (name, sha256) in sorted(applied.items()):
        migration = known.get(version)
        if migration is None:
            raise ValueError(f"the database has migration {version} ({name}), which is not here")
        if (migration.name, migration.sha256) != (name, sha256):
            raise ValueError(
                f"{migration.name} has changed since it was applied as {name}; "
                "add a new migration instead of editing an applied one"
            )
    return [migration for migration in migrations if migration.version not in applied]


def applied_migrations(connection: psycopg.Connection) -> dict[int, tuple[str, str]]:
    with connection.transaction():
        connection.execute(TRACKING)
    rows = connection.execute("SELECT version, name, sha256 FROM schema_migrations").fetchall()
    return {int(version): (str(name), str(sha256)) for version, name, sha256 in rows}


def migrate(
    connection: psycopg.Connection, migrations: Sequence[Migration], dry_run: bool = False
) -> list[Migration]:
    """Apply the pending migrations and return them. Each file runs in a savepoint inside one
    open transaction, which the caller commits (main) or rolls back (the tests), so all pending
    files are applied together or not at all."""
    pending = to_apply(applied_migrations(connection), migrations)
    if dry_run:
        return pending
    for migration in pending:
        with connection.transaction():
            connection.execute(migration.sql.encode())
            connection.execute(
                "INSERT INTO schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.sha256),
            )
    return pending


def table_counts(connection: psycopg.Connection) -> dict[str, int]:
    """Rows per table in the current schema (schema_migrations left out)."""
    from psycopg import sql

    names = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE' "
        "AND table_name <> 'schema_migrations' ORDER BY table_name"
    ).fetchall()
    counts = {}
    for (name,) in names:
        query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(name))
        row = connection.execute(query).fetchone()
        counts[str(name)] = int(row[0]) if row else 0
    return counts


def check_counts_kept(before: Mapping[str, int], after: Mapping[str, int]) -> None:
    """Every table that existed before still has the same number of rows."""
    print(f"tables before: {len(before)}, after: {len(after)}")
    for name, count in before.items():
        print(f"  {name}: before {count} rows, after {after.get(name)} rows")
        if after.get(name) != count:
            raise RuntimeError(f"{name} had {count} rows before and {after.get(name)} after")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(DEMO["env_file"]))
    parser.add_argument("--dry-run", action="store_true", help="list pending, apply nothing")
    args = parser.parse_args(argv)
    import psycopg

    migrations = load()
    with psycopg.connect(results_db_url(read_env(args.env_file))) as connection:
        row = connection.execute("SELECT current_database()").fetchone()
        database = str(row[0]) if row else ""
        connection.commit()
        if database not in (DEMO["db_name"], TEST["db_name"]):
            raise RuntimeError(f"{database!r} is neither the demo nor the test results database")
        print(f"database {database} ({args.env_file})")
        before = table_counts(connection)
        connection.commit()
        pending = migrate(connection, migrations, args.dry_run)
        connection.commit()
        verb = "pending" if args.dry_run else "applied"
        print(f"{verb}: {[migration.name for migration in pending] or 'none'}")
        check_counts_kept(before, table_counts(connection))


if __name__ == "__main__":
    main()
