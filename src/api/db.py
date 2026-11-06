"""Read-only queries on the results database for the API (one short connection per call).

The tables are created by the migrations (src/service/migrate.py); a database that was not
migrated is reported as unavailable, with what to run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import psycopg
from psycopg.rows import dict_row

T = TypeVar("T")

RESULT_COLUMNS = (
    "id, study_uid, orthanc_study_id, modality, model, checkpoint_name, checkpoint_sha256, "
    "status, seconds, created_at"
)


class DatabaseUnavailableError(RuntimeError):
    pass


class Database(Protocol):
    def name(self) -> str: ...

    def newest_results(self, study_uid: str | None = None) -> list[dict[str, Any]]:
        """The newest row per (study, model); with study_uid, that study only and with the
        results JSON."""
        ...

    def newest_reviews(self, study_uid: str | None = None) -> dict[str, dict[str, Any]]:
        """study UID -> its newest review."""
        ...

    def uploaded(self) -> set[str]:
        """Orthanc study IDs and StudyInstanceUIDs recorded in uploads."""
        ...


@dataclass(frozen=True)
class PostgresDatabase:
    url: str
    connect_timeout: int = 5

    def _run(self, query: Callable[[psycopg.Connection[dict[str, Any]]], T]) -> T:
        try:
            with psycopg.connect(
                self.url, connect_timeout=self.connect_timeout, row_factory=dict_row
            ) as connection:
                return query(connection)
        except psycopg.errors.UndefinedTable as error:
            raise DatabaseUnavailableError(
                f"the results database is not migrated ({error.diag.message_primary}); "
                "run python -m src.service.migrate"
            ) from None
        except psycopg.OperationalError as error:
            reason = str(error).strip().splitlines()[-1] if str(error).strip() else "no reason"
            raise DatabaseUnavailableError(
                f"the results database is not reachable: {reason}"
            ) from None

    def name(self) -> str:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> str:
            row = connection.execute("SELECT current_database() AS name").fetchone()
            return str(row["name"]) if row else ""

        return self._run(query)

    def newest_results(self, study_uid: str | None = None) -> list[dict[str, Any]]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> list[dict[str, Any]]:
            if study_uid is None:
                return connection.execute(
                    f"SELECT DISTINCT ON (study_uid, model) {RESULT_COLUMNS} "
                    "FROM inference_results ORDER BY study_uid, model, id DESC"
                ).fetchall()
            return connection.execute(
                f"SELECT DISTINCT ON (model) {RESULT_COLUMNS}, results FROM inference_results "
                "WHERE study_uid = %s ORDER BY model, id DESC",
                (study_uid,),
            ).fetchall()

        return self._run(query)

    def newest_reviews(self, study_uid: str | None = None) -> dict[str, dict[str, Any]]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> list[dict[str, Any]]:
            select = (
                "SELECT DISTINCT ON (study_uid) id, study_uid, inference_result_id, decision, "
                "reviewer, comment, created_at FROM reviews"
            )
            if study_uid is None:
                return connection.execute(f"{select} ORDER BY study_uid, id DESC").fetchall()
            return connection.execute(
                f"{select} WHERE study_uid = %s ORDER BY study_uid, id DESC", (study_uid,)
            ).fetchall()

        return {row["study_uid"]: row for row in self._run(query)}

    def uploaded(self) -> set[str]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> list[dict[str, Any]]:
            return connection.execute("SELECT orthanc_study_id, study_uid FROM uploads").fetchall()

        found: set[str] = set()
        for row in self._run(query):
            found.update(value for value in row.values() if value)
        return found
