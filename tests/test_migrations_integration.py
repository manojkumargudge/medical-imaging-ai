"""The migrations on the real Postgres from docker-compose.base.yml: tables, rules and triggers.

Runs only with AIML02_INTEGRATION=1 against the test stack (.env.test, compose project
aiml02-test), never the demo; the database name is checked first. Each test works in a new
schema inside one transaction that is rolled back at the end, so nothing is kept and nothing
has to be dropped.
"""

import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from stack_helpers import integration_db_url

from src.service import migrate
from src.service.store import SCHEMA

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIML02_INTEGRATION") != "1", reason="set AIML02_INTEGRATION=1"
    ),
]


@pytest.fixture
def db() -> Iterator[psycopg.Connection[Any]]:
    connection = psycopg.connect(integration_db_url())  # not autocommit: one open transaction
    schema = f"it_{uuid.uuid4().hex[:12]}"
    connection.execute(f"CREATE SCHEMA {schema}".encode())
    connection.execute(f"SET LOCAL search_path TO {schema}".encode())
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def add_result(
    connection: psycopg.Connection[Any], study: str, status: str = "ok", sha: str = "sha"
) -> int | None:
    """Insert as the worker does (ON CONFLICT DO NOTHING); the new id, or None if it existed."""
    results: dict[str, Any] = (
        {"instances": []} if status == "ok" else {"error": "ValueError: bad pixels"}
    )
    row = connection.execute(
        "INSERT INTO inference_results (study_uid, orthanc_study_id, modality, model, "
        "checkpoint_name, checkpoint_sha256, status, results, seconds) "
        "VALUES (%s, 'o', 'DX', 'densenet121-chestxray14', 'c.pt', %s, %s, %s, 1.5) "
        "ON CONFLICT (study_uid, checkpoint_sha256) DO NOTHING RETURNING id",
        (study, sha, status, json.dumps(results)),
    ).fetchone()
    return int(row[0]) if row else None


def count(connection: psycopg.Connection[Any], table: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {table}".encode()).fetchone()
    return int(row[0]) if row else 0


def refused(
    connection: psycopg.Connection[Any], query: str, params: tuple[Any, ...], match: str
) -> None:
    with pytest.raises(psycopg.Error, match=match), connection.transaction():
        connection.execute(query.encode(), params)


def review(
    study: str,
    result_id: int,
    decision: str,
    reviewer: str | None = "A. Reviewer",
    comment: str | None = None,
    token: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    query = (
        "INSERT INTO reviews (study_uid, inference_result_id, decision, reviewer, comment, "
        "submit_token) VALUES (%s, %s, %s, %s, %s, %s)"
    )
    return query, (study, result_id, decision, reviewer, comment, token or str(uuid.uuid4()))


@pytest.fixture
def migrated(db: psycopg.Connection[Any]) -> psycopg.Connection[Any]:
    migrate.migrate(db, migrate.load())
    return db


def test_migrations_keep_an_existing_worker_database_and_apply_once(
    db: psycopg.Connection[Any],
) -> None:
    # A database as the worker created it before migrations existed, with rows in it.
    db.execute(SCHEMA.encode())
    add_result(db, "1.2.3")
    add_result(db, "1.2.4", "error")
    db.execute("INSERT INTO worker_state VALUES ('orthanc_last_change_seq', 234)")
    before = migrate.table_counts(db)

    applied = migrate.migrate(db, migrate.load())
    after = migrate.table_counts(db)
    print(f"applied {[m.name for m in applied]}; before {before}; after {after}")
    assert [m.name for m in applied] == ["001_initial.sql", "002_reviews_audit_uploads.sql"]
    assert before and after and before["inference_results"] == 2
    migrate.check_counts_kept(before, after)
    assert {"reviews", "audit_log", "uploads"} <= set(after)
    assert after["audit_log"] == 0  # the trigger audits new inserts only

    assert migrate.migrate(db, migrate.load()) == []  # a second run applies nothing
    assert count(db, "schema_migrations") == 2


def test_each_inference_insert_adds_one_audit_row(migrated: psycopg.Connection[Any]) -> None:
    done = add_result(migrated, "1.2.3")
    failed = add_result(migrated, "1.2.4", "error")
    assert add_result(migrated, "1.2.3") is None  # a rerun inserts nothing and audits nothing
    rows = migrated.execute(
        "SELECT actor, event, study_uid, details FROM audit_log ORDER BY id"
    ).fetchall()
    print(f"result rows {count(migrated, 'inference_results')}, audit rows {len(rows)}")
    assert [(actor, event, study) for actor, event, study, _ in rows] == [
        ("worker", "inference_done", "1.2.3"),
        ("worker", "inference_failed", "1.2.4"),
    ]
    assert rows[0][3]["inference_result_id"] == done and "error" not in rows[0][3]
    assert rows[1][3]["inference_result_id"] == failed
    assert rows[1][3]["error"] == "ValueError: bad pixels"


def test_review_rules(migrated: psycopg.Connection[Any]) -> None:
    result = add_result(migrated, "1.2.3")
    other = add_result(migrated, "9.9.9")
    assert result is not None and other is not None
    token = str(uuid.uuid4())
    migrated.execute(*review("1.2.3", result, "approve", token=token))

    for (query, params), match in [
        (review("1.2.3", result, "reject"), "reviews_reject_needs_comment"),
        (review("1.2.3", result, "reject", comment="   "), "reviews_reject_needs_comment"),
        (review("1.2.3", result, "approve", reviewer="  "), "reviews_reviewer_check"),
        (review("1.2.3", result, "approve", reviewer=None), "null value"),
        (review("1.2.3", result, "maybe"), "reviews_decision_check"),
        (review("1.2.3", other, "approve"), "is not a result of study"),
        (review("1.2.3", 10**9, "approve"), "is not a result of study"),
    ]:
        refused(migrated, query, params, match)
    # The same submit token twice (a double click, a resent form) is refused.
    refused(migrated, *review("1.2.3", result, "approve", token=token), "submit_token")

    migrated.execute(*review("1.2.3", result, "reject", comment="Overlay off the lung"))
    rows = migrated.execute("SELECT decision FROM reviews ORDER BY id").fetchall()
    latest = migrated.execute(
        "SELECT decision, comment FROM reviews WHERE study_uid = '1.2.3' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    print(f"reviews {[r[0] for r in rows]}, current state {latest}")
    assert [r[0] for r in rows] == ["approve", "reject"]
    assert latest == ("reject", "Overlay off the lung")


def test_reviews_and_audit_log_are_append_only(migrated: psycopg.Connection[Any]) -> None:
    result = add_result(migrated, "1.2.3")
    assert result is not None
    migrated.execute(*review("1.2.3", result, "approve"))
    before = {table: count(migrated, table) for table in ("reviews", "audit_log")}
    for table in before:
        refused(migrated, f"UPDATE {table} SET study_uid = 'x'", (), "UPDATE on .* refused")
        refused(migrated, f"DELETE FROM {table}", (), "DELETE on .* refused")
        refused(migrated, f"TRUNCATE {table}", (), "TRUNCATE on .* refused")
    after = {table: count(migrated, table) for table in before}
    print(f"rows before {before}, after {after}")
    assert all(before.values()) and before == after

    # A reviewed result cannot be deleted (scripts/reset_demo.py checks this first).
    refused(migrated, "DELETE FROM inference_results", (), "foreign key")


def test_uploads_need_a_name_and_a_size(migrated: psycopg.Connection[Any]) -> None:
    query = (
        "INSERT INTO uploads (uploader, kind, filename, size_bytes, sha256) "
        "VALUES (%s, %s, 'a.png', %s, 'x')"
    )
    migrated.execute(query.encode(), ("A. Uploader", "png", 10))
    refused(migrated, query, (" ", "png", 10), "uploads_uploader_check")
    refused(migrated, query, ("A. Uploader", "nifti", 10), "uploads_kind_check")
    refused(migrated, query, ("A. Uploader", "png", 0), "uploads_size_bytes_check")
    assert count(migrated, "uploads") == 1
