"""Unit tests for scripts/archi_backup.py and scripts/archi_restore.py.

We don't have a live Postgres in CI, so the tests focus on the pure helpers
(encoding/decoding, manifest schema, ON CONFLICT SQL generation) plus a
round-trip via an in-process fake cursor.
"""

from __future__ import annotations

import base64
import gzip
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load(name: str):
    """Import a script module from the scripts/ directory by name."""
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------


def test_encode_round_trip_bytes_and_datetimes():
    backup = _load("archi_backup")
    restore = _load("archi_restore")

    raw_dt = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
    raw_bytes = b"\x00\x01encrypted\xff"
    row = {"id": "abc", "ts": raw_dt, "blob": raw_bytes, "n": 7, "text": "hi"}
    encoded = backup._encode_row(row)

    # JSON-roundtrip the encoded dict to mimic disk persistence.
    decoded = restore._decode_row(json.loads(json.dumps(encoded)))

    assert decoded["id"] == "abc"
    assert decoded["text"] == "hi"
    assert decoded["n"] == 7
    assert decoded["blob"] == raw_bytes
    assert decoded["ts"] == raw_dt


def test_encode_leaves_plain_values_alone():
    backup = _load("archi_backup")
    encoded = backup._encode_row({"a": 1, "b": None, "c": "text"})
    assert encoded == {"a": 1, "b": None, "c": "text"}


# ---------------------------------------------------------------------------
# Manifest round-trip via a fake cursor
# ---------------------------------------------------------------------------


class FakeCursor:
    """psycopg2-style cursor producing canned data for a single table."""

    def __init__(self, rows_by_query, *, dict_rows: bool):
        self._rows_by_query = rows_by_query
        self._dict_rows = dict_rows
        self._pending: List[Any] = []
        self._last_result = None

    def execute(self, sql: str, params=None) -> None:
        sql_norm = " ".join(sql.split())
        params = params or ()
        # information_schema lookups → "table exists?"
        if "information_schema.tables" in sql_norm:
            table = params[0] if params else ""
            self._last_result = (1,) if table in self._rows_by_query else None
            return
        if sql_norm.upper().startswith("SELECT MIGRATION_NAME"):
            self._last_result = ("0042_test_migration",)
            return
        if sql_norm.upper().startswith("SELECT * FROM "):
            table = sql_norm.split()[-1]
            rows = self._rows_by_query.get(table, [])
            self._pending = [dict(r) if self._dict_rows else tuple(r.values())
                             for r in rows]
            return
        raise AssertionError(f"FakeCursor doesn't handle: {sql_norm}")

    def fetchone(self):
        return self._last_result

    def fetchmany(self, size: int):
        if not self._pending:
            return []
        chunk = self._pending[:size]
        self._pending = self._pending[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, rows_by_query):
        self._rows_by_query = rows_by_query

    def cursor(self, *, cursor_factory=None):
        # If a dict-row cursor factory is requested (RealDictCursor) return dict rows.
        dict_rows = cursor_factory is not None and "RealDict" in repr(cursor_factory)
        return FakeCursor(self._rows_by_query, dict_rows=dict_rows)

    def close(self):
        pass


@pytest.fixture
def backup_rows():
    return {
        "users": [
            {"id": "u1", "display_name": "Alice", "email": "a@x"},
            {"id": "u2", "display_name": "Bob", "email": "b@x"},
        ],
        "conversation_metadata": [
            {"conversation_id": 1, "title": "Hello"},
        ],
        "conversations": [],
        # migration_state present (empty rows) so _table_exists returns True
        # and the schema-version lookup falls through to the canned response.
        "migration_state": [],
        # All other tables are absent — exercise the "skipped" branch.
    }


def test_run_backup_writes_manifest_and_jsonl(monkeypatch, tmp_path, backup_rows):
    backup = _load("archi_backup")

    def _fake_connect(**_kwargs):
        return FakeConn(backup_rows)

    monkeypatch.setattr(backup.psycopg2, "connect", _fake_connect)

    out_dir = tmp_path / "backup"
    result = backup.run_backup(tables=backup.DEFAULT_TABLES, out_dir=out_dir,
                               pg_config={"host": "x"})
    assert result == out_dir

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["format_version"] == "1"
    assert manifest["schema_version"] == "0042_test_migration"
    assert manifest["tables"]["users"]["rows"] == 2
    assert manifest["tables"]["conversation_metadata"]["rows"] == 1
    assert "conversations" not in manifest["tables"] or manifest["tables"]["conversations"]["rows"] == 0
    # Tables that the fake said weren't present should be absent from the manifest.
    assert "agent_traces" not in manifest["tables"]
    assert "tool_approvals" not in manifest["tables"]

    # The users.jsonl.gz file should contain two lines, each one a JSON row.
    with gzip.open(out_dir / "users.jsonl.gz", "rt") as fp:
        lines = [json.loads(ln) for ln in fp if ln.strip()]
    assert {row["id"] for row in lines} == {"u1", "u2"}


def test_default_tables_listed_in_dependency_order():
    backup = _load("archi_backup")
    # users must come before conversation_metadata, which must come before conversations.
    order = list(backup.DEFAULT_TABLES)
    assert order.index("users") < order.index("conversation_metadata")
    assert order.index("conversation_metadata") < order.index("conversations")
    # tool_approvals depends on conversation_metadata via conversation_id.
    assert order.index("conversation_metadata") < order.index("tool_approvals")


# ---------------------------------------------------------------------------
# Restore: ON CONFLICT SQL composition
# ---------------------------------------------------------------------------


class _FlushCapture:
    """Drop-in for psycopg2 cursor capturing the (sql, params) of each execute."""

    def __init__(self):
        self.calls: List[tuple[str, tuple]] = []
        self.rowcount = 1

    def execute(self, sql, params):
        self.calls.append((" ".join(sql.split()), params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FlushConn:
    def __init__(self, capture):
        self._capture = capture

    def cursor(self):
        return self._capture

    def commit(self):
        pass


def test_flush_batch_emits_on_conflict_do_nothing():
    restore = _load("archi_restore")
    cap = _FlushCapture()
    inserted = restore._flush_batch(
        _FlushConn(cap), "users", ("id",),
        [{"id": "u1", "display_name": "Alice"}],
    )
    assert inserted == 1
    sql, params = cap.calls[0]
    assert 'INSERT INTO users' in sql
    assert 'ON CONFLICT ("id") DO NOTHING' in sql
    assert params == ("u1", "Alice")


def test_flush_batch_rejects_unknown_table():
    restore = _load("archi_restore")
    # Unknown table → no PK entry → restore refuses.
    with pytest.raises(ValueError, match="Refusing to restore unknown table"):
        restore._restore_one_table(None, "totally_not_a_table", Path("/tmp/x"))
