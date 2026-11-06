"""src/service/migrate.py without a database: file names, order, and changed-file detection."""

from pathlib import Path

import pytest

from src.service import migrate
from src.service.store import SCHEMA


def write(folder: Path, name: str, sql: str = "SELECT 1;") -> None:
    (folder / name).write_text(sql, encoding="utf-8")


def test_the_repository_migrations_load_in_order() -> None:
    migrations = migrate.load()
    assert [m.name for m in migrations][:2] == ["001_initial.sql", "002_reviews_audit_uploads.sql"]
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))


def test_001_is_the_worker_schema_so_it_is_a_no_op_on_the_demo_database() -> None:
    first = migrate.load()[0]
    assert SCHEMA.strip() in first.sql
    statements = [s.strip() for s in first.sql.split(";") if "CREATE" in s]
    assert statements and all("IF NOT EXISTS" in s for s in statements)


def test_migrations_never_drop_or_delete() -> None:
    for migration in migrate.load():
        code = "\n".join(
            line for line in migration.sql.upper().splitlines() if not line.strip().startswith("--")
        )
        for word in ("DROP ", "DELETE ", "TRUNCATE ", "ALTER TABLE INFERENCE_RESULTS"):
            # TRUNCATE appears only as a trigger event (BEFORE TRUNCATE ON ...).
            hits = [line for line in code.splitlines() if word in line and "BEFORE" not in line]
            assert not hits, f"{migration.name}: {hits}"


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["001_a.sql", "003_c.sql"], "without gaps"),
        (["002_a.sql"], "without gaps"),
        (["1_a.sql"], "NNN_name.sql"),
        (["001_A.sql"], "NNN_name.sql"),
    ],
)
def test_bad_file_names_or_gaps_are_refused(tmp_path: Path, names: list[str], message: str) -> None:
    for name in names:
        write(tmp_path, name)
    with pytest.raises(ValueError, match=message):
        migrate.load(tmp_path)


def test_only_unapplied_migrations_run(tmp_path: Path) -> None:
    write(tmp_path, "001_a.sql")
    write(tmp_path, "002_b.sql", "SELECT 2;")
    first, second = migrate.load(tmp_path)
    assert migrate.to_apply({}, [first, second]) == [first, second]
    assert migrate.to_apply({1: (first.name, first.sha256)}, [first, second]) == [second]
    applied = {1: (first.name, first.sha256), 2: (second.name, second.sha256)}
    assert migrate.to_apply(applied, [first, second]) == []


def test_an_edited_applied_migration_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "001_a.sql")
    [first] = migrate.load(tmp_path)
    with pytest.raises(ValueError, match="has changed since it was applied"):
        migrate.to_apply({1: (first.name, "0" * 64)}, [first])
    with pytest.raises(ValueError, match="not here"):
        migrate.to_apply({1: (first.name, first.sha256), 2: ("002_x.sql", "x")}, [first])


def test_changed_row_counts_are_reported() -> None:
    migrate.check_counts_kept({"inference_results": 3}, {"inference_results": 3, "reviews": 0})
    with pytest.raises(RuntimeError, match="had 3 rows before and 2 after"):
        migrate.check_counts_kept({"inference_results": 3}, {"inference_results": 2})
