"""Queries on the results database for the API (one short connection per call).

Reads, plus one write: a review and its audit row, in one transaction. The tables are created by
the migrations (src/service/migrate.py); a database that was not migrated is reported as
unavailable, with what to run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

T = TypeVar("T")

RESULT_COLUMNS = (
    "id, study_uid, orthanc_study_id, modality, model, checkpoint_name, checkpoint_sha256, "
    "status, results, seconds, created_at"
)
REVIEW_COLUMNS = "id, study_uid, inference_result_id, decision, reviewer, comment, created_at"


class DatabaseUnavailableError(RuntimeError):
    pass


class ReviewConflictError(RuntimeError):
    """409: the submit token was used already, or a newer result exists."""


class ReviewInvalidError(ValueError):
    """422: the review breaks a database rule (e.g. reject without a comment)."""


@dataclass(frozen=True)
class NewReview:
    study_uid: str
    inference_result_id: int
    decision: str
    reviewer: str
    comment: str | None
    submit_token: UUID


@dataclass(frozen=True)
class NewUpload:
    id: int
    uploader: str
    kind: str
    filename: str  # kept in the database only; never in the DICOM or an API response
    size_bytes: int
    sha256: str
    study_uid: str
    orthanc_study_id: str


ALREADY_SAVED = "already saved: this review was submitted before (same submit token)"
NEWER_RESULT = "a newer result exists for this study; reload the page and review the newest one"


class Database(Protocol):
    def name(self) -> str: ...

    def newest_results(self, study_uid: str | None = None) -> list[dict[str, Any]]:
        """The newest row per (study, model), with the results JSON; or one study's."""
        ...

    def newest_reviews(self, study_uid: str | None = None) -> dict[str, dict[str, Any]]:
        """study UID -> its newest review."""
        ...

    def uploaded(self) -> set[str]:
        """Orthanc study IDs and StudyInstanceUIDs recorded in uploads."""
        ...

    def find_upload(self, sha256: str) -> dict[str, Any] | None:
        """The newest upload of a file with this sha256."""
        ...

    def reserve_upload_id(self) -> int:
        """The next uploads ID, taken before Orthanc is called (its UIDs are seeded by it)."""
        ...

    def add_upload(self, upload: NewUpload) -> dict[str, Any]:
        """Insert the uploads row and its audit row in one transaction; returns the upload."""
        ...

    def reviews(self, study_uid: str) -> list[dict[str, Any]]:
        """The study's reviews, newest first."""
        ...

    def add_review(self, review: NewReview) -> dict[str, Any]:
        """Insert the review and its audit row in one transaction; returns the review."""
        ...

    def audit(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """A page of the audit log, newest first, and the total number of rows."""
        ...


@dataclass(frozen=True)
class PostgresDatabase:
    url: str
    connect_timeout: int = 5

    def _run(self, query: Callable[[psycopg.Connection[dict[str, Any]]], T]) -> T:
        """Run in one transaction: committed if `query` returns, rolled back if it raises."""
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
                f"SELECT DISTINCT ON (model) {RESULT_COLUMNS} FROM inference_results "
                "WHERE study_uid = %s ORDER BY model, id DESC",
                (study_uid,),
            ).fetchall()

        return self._run(query)

    def newest_reviews(self, study_uid: str | None = None) -> dict[str, dict[str, Any]]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> list[dict[str, Any]]:
            select = f"SELECT DISTINCT ON (study_uid) {REVIEW_COLUMNS} FROM reviews"
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

    def find_upload(self, sha256: str) -> dict[str, Any] | None:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> dict[str, Any] | None:
            return connection.execute(
                "SELECT id, study_uid, orthanc_study_id FROM uploads WHERE sha256 = %s "
                "ORDER BY id DESC LIMIT 1",
                (sha256,),
            ).fetchone()

        return self._run(query)

    def reserve_upload_id(self) -> int:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> int:
            row = connection.execute(
                "SELECT nextval(pg_get_serial_sequence('uploads', 'id')) AS id"
            ).fetchone()
            if row is None:
                raise DatabaseUnavailableError("could not reserve an upload ID")
            return int(row["id"])

        return self._run(query)

    def add_upload(self, upload: NewUpload) -> dict[str, Any]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> dict[str, Any]:
            row = connection.execute(
                "INSERT INTO uploads (id, uploader, kind, filename, size_bytes, sha256, "
                "study_uid, orthanc_study_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, uploader, kind, size_bytes, sha256, study_uid, orthanc_study_id, "
                "created_at",
                (
                    upload.id,
                    upload.uploader,
                    upload.kind,
                    upload.filename,
                    upload.size_bytes,
                    upload.sha256,
                    upload.study_uid,
                    upload.orthanc_study_id,
                ),
            ).fetchone()
            if row is None:
                raise DatabaseUnavailableError("the upload was not recorded")
            details = {
                "upload_id": upload.id,
                "kind": upload.kind,
                "size_bytes": upload.size_bytes,
                "sha256": upload.sha256,
                "orthanc_study_id": upload.orthanc_study_id,
                "uploader_verified": False,
            }
            connection.execute(
                "INSERT INTO audit_log (actor, event, study_uid, details) "
                "VALUES (%s, 'upload', %s, %s)",
                (upload.uploader, upload.study_uid, json.dumps(details)),
            )
            return row

        return self._run(query)

    def reviews(self, study_uid: str) -> list[dict[str, Any]]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> list[dict[str, Any]]:
            return connection.execute(
                f"SELECT {REVIEW_COLUMNS} FROM reviews WHERE study_uid = %s ORDER BY id DESC",
                (study_uid,),
            ).fetchall()

        return self._run(query)

    def add_review(self, review: NewReview) -> dict[str, Any]:
        def query(connection: psycopg.Connection[dict[str, Any]]) -> dict[str, Any]:
            # The result the reviewer saw must be the newest of its model for this study (a
            # result the worker adds in the same instant is not seen; the review then names
            # the result it was made on, which stays true).
            seen = connection.execute(
                "SELECT model FROM inference_results WHERE id = %s AND study_uid = %s",
                (review.inference_result_id, review.study_uid),
            ).fetchone()
            if seen is None:
                raise ReviewInvalidError("inference_result_id is not a result of this study")
            newest = connection.execute(
                "SELECT id FROM inference_results WHERE study_uid = %s AND model = %s "
                "ORDER BY id DESC LIMIT 1",
                (review.study_uid, seen["model"]),
            ).fetchone()
            if newest is None or newest["id"] != review.inference_result_id:
                raise ReviewConflictError(NEWER_RESULT)
            try:
                with connection.transaction():
                    row = connection.execute(
                        "INSERT INTO reviews (study_uid, inference_result_id, decision, "
                        "reviewer, comment, submit_token) VALUES (%s, %s, %s, %s, %s, %s) "
                        f"ON CONFLICT (submit_token) DO NOTHING RETURNING {REVIEW_COLUMNS}",
                        (
                            review.study_uid,
                            review.inference_result_id,
                            review.decision,
                            review.reviewer,
                            review.comment,
                            review.submit_token,
                        ),
                    ).fetchone()
            except psycopg.errors.CheckViolation as error:
                raise ReviewInvalidError(
                    f"the database refused the review ({error.diag.constraint_name})"
                ) from None
            if row is None:
                raise ReviewConflictError(ALREADY_SAVED)
            details = {
                "review_id": row["id"],
                "decision": review.decision,
                "inference_result_id": review.inference_result_id,
                "comment": review.comment,
                "reviewer_verified": False,
            }
            connection.execute(
                "INSERT INTO audit_log (actor, event, study_uid, details) "
                "VALUES (%s, 'review', %s, %s)",
                (review.reviewer, review.study_uid, json.dumps(details)),
            )
            return row

        return self._run(query)

    def audit(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        def query(
            connection: psycopg.Connection[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], int]:
            rows = connection.execute(
                "SELECT id, created_at, actor, event, study_uid, details FROM audit_log "
                "ORDER BY id DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
            total = connection.execute("SELECT count(*) AS n FROM audit_log").fetchone()
            return rows, int(total["n"]) if total else 0

        return self._run(query)
