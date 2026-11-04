"""Where the worker keeps results and its place in Orthanc's change log.

One row per (StudyInstanceUID, checkpoint sha256): a rerun, or a second StableStudy for the same
study, inserts nothing new. The last processed change sequence is stored too, so a restarted
worker resumes where it stopped. PostgresStore is the real store; MemoryStore is for tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

SEQ_KEY = "orthanc_last_change_seq"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_results (
    id BIGSERIAL PRIMARY KEY,
    study_uid TEXT NOT NULL,
    orthanc_study_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    model TEXT NOT NULL,
    checkpoint_name TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    results JSONB NOT NULL,
    seconds DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (study_uid, checkpoint_sha256)
);
CREATE TABLE IF NOT EXISTS worker_state (
    key TEXT PRIMARY KEY,
    value BIGINT NOT NULL
);
"""


@dataclass(frozen=True)
class Result:
    study_uid: str
    orthanc_study_id: str
    modality: str
    model: str
    checkpoint_name: str
    checkpoint_sha256: str
    status: str  # "ok" or "error"
    results: dict[str, Any]
    seconds: float


class Store(Protocol):
    def last_seq(self) -> int: ...

    def set_last_seq(self, seq: int) -> None: ...

    def has(self, study_uid: str, checkpoint_sha256: str) -> bool: ...

    def insert(self, result: Result) -> bool:
        """True if a row was added, False if (study, checkpoint) was already there."""
        ...


@dataclass
class MemoryStore:
    seq: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def last_seq(self) -> int:
        return self.seq

    def set_last_seq(self, seq: int) -> None:
        self.seq = seq

    def has(self, study_uid: str, checkpoint_sha256: str) -> bool:
        return any(
            row["study_uid"] == study_uid and row["checkpoint_sha256"] == checkpoint_sha256
            for row in self.rows
        )

    def insert(self, result: Result) -> bool:
        if self.has(result.study_uid, result.checkpoint_sha256):
            return False
        # Through JSON, as in Postgres, so tests see what would be stored.
        row = json.loads(json.dumps(result.__dict__))
        self.rows.append({**row, "created_at": datetime.now(UTC).isoformat()})
        return True


class PostgresStore:
    def __init__(self, url: str, temporary: bool = False) -> None:
        """temporary=True uses session-only tables (tests): the real tables are not touched,
        and Postgres drops the temporary ones when the connection closes."""
        import psycopg

        self.connection = psycopg.connect(url, autocommit=True)
        schema = SCHEMA.replace("CREATE TABLE", "CREATE TEMP TABLE") if temporary else SCHEMA
        self.connection.execute(schema)

    def close(self) -> None:
        self.connection.close()

    def last_seq(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM worker_state WHERE key = %s", (SEQ_KEY,)
        ).fetchone()
        return int(row[0]) if row else 0

    def set_last_seq(self, seq: int) -> None:
        self.connection.execute(
            "INSERT INTO worker_state (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (SEQ_KEY, seq),
        )

    def has(self, study_uid: str, checkpoint_sha256: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM inference_results WHERE study_uid = %s AND checkpoint_sha256 = %s",
            (study_uid, checkpoint_sha256),
        ).fetchone()
        return row is not None

    def insert(self, result: Result) -> bool:
        cursor = self.connection.execute(
            "INSERT INTO inference_results (study_uid, orthanc_study_id, modality, model, "
            "checkpoint_name, checkpoint_sha256, status, results, seconds) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (study_uid, checkpoint_sha256) DO NOTHING",
            (
                result.study_uid,
                result.orthanc_study_id,
                result.modality,
                result.model,
                result.checkpoint_name,
                result.checkpoint_sha256,
                result.status,
                json.dumps(result.results),
                result.seconds,
            ),
        )
        return cursor.rowcount == 1

    def rows(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            "SELECT study_uid, modality, model, checkpoint_name, checkpoint_sha256, status, "
            "results, seconds, created_at FROM inference_results ORDER BY id"
        )
        names = [column.name for column in cursor.description or ()]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
