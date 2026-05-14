"""Unit tests for UserActionService.

Uses a tiny in-process fake psycopg2 cursor so the lifecycle (record /
list_for_user / filter / since / limit) can be exercised without a live DB.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --- Fake psycopg2 cursor / pool ---------------------------------------------


class FakeCursor:
    def __init__(self, store):
        self._store = store
        self._last_result = None

    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        s = " ".join(sql.split()).upper()
        params = params or ()
        if s.startswith("CREATE TABLE") or s.startswith("CREATE INDEX"):
            return
        if s.startswith("INSERT INTO USER_ACTIONS"):
            (action_id, user_id, action_type, target_kind, target_id,
             payload_json, source, ts) = params
            import json as _json
            payload = _json.loads(payload_json) if isinstance(payload_json, str) else payload_json
            self._store.append({
                "action_id": action_id, "user_id": user_id,
                "action_type": action_type, "target_kind": target_kind,
                "target_id": target_id, "payload": payload,
                "source": source, "ts": ts,
            })
            return
        if "FROM USER_ACTIONS" in s and "ORDER BY TS DESC" in s:
            # params layout: user_id, [since], [action_types], limit
            user_id = params[0]
            since = None
            action_types = None
            idx = 1
            if "TS >= %S" in s:
                since = params[idx]
                idx += 1
            if "ACTION_TYPE = ANY" in s:
                action_types = params[idx]
                idx += 1
            limit = params[idx]
            rows = [r for r in self._store if r["user_id"] == user_id]
            if since:
                rows = [r for r in rows if r["ts"] >= since]
            if action_types:
                rows = [r for r in rows if r["action_type"] in action_types]
            rows.sort(key=lambda r: r["ts"], reverse=True)
            rows = rows[:limit]
            self._last_result = [
                (r["action_id"], r["user_id"], r["action_type"], r["target_kind"],
                 r["target_id"], r["payload"], r["source"], r["ts"])
                for r in rows
            ]
            return
        raise AssertionError(f"FakeCursor doesn't handle: {s[:120]}")

    def fetchone(self):
        if isinstance(self._last_result, list):
            return self._last_result[0] if self._last_result else None
        return self._last_result

    def fetchall(self):
        if isinstance(self._last_result, list):
            return self._last_result
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, store):
        self._store = store

    def cursor(self):
        return FakeCursor(self._store)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class FakePool:
    def __init__(self):
        self.store: List[dict] = []

    def get_connection(self):
        return FakeConn(self.store)

    def release_connection(self, _conn):
        pass


def _make_service():
    from src.utils.user_action_service import UserActionService

    pool = FakePool()
    return UserActionService(connection_pool=pool), pool


# --- Tests -------------------------------------------------------------------


def test_record_appends_row():
    svc, pool = _make_service()
    out = svc.record(
        user_id="alice", action_type="preferences_updated",
        target_kind="user", target_id="alice",
        payload={"theme": "dark"}, source="web",
    )
    assert out is not None
    assert out.user_id == "alice"
    assert out.payload == {"theme": "dark"}
    assert len(pool.store) == 1


def test_record_rejects_empty_action_type():
    svc, _ = _make_service()
    assert svc.record(user_id="u", action_type="") is None


def test_record_unknown_source_defaults_to_web(caplog):
    svc, pool = _make_service()
    out = svc.record(user_id="u", action_type="t", source="haxx")
    assert out is not None and out.source == "web"


def test_list_for_user_returns_newest_first():
    svc, _ = _make_service()
    older = svc.record(user_id="u", action_type="api_key_set")
    newer = svc.record(user_id="u", action_type="api_key_removed")
    rows = svc.list_for_user("u")
    assert [r.action_type for r in rows] == ["api_key_removed", "api_key_set"]


def test_list_for_user_filters_by_action_type():
    svc, _ = _make_service()
    svc.record(user_id="u", action_type="api_key_set")
    svc.record(user_id="u", action_type="preferences_updated")
    rows = svc.list_for_user("u", action_types=["api_key_set"])
    assert len(rows) == 1
    assert rows[0].action_type == "api_key_set"


def test_list_for_user_filters_by_since():
    from src.utils.user_action_service import UserActionService

    svc, pool = _make_service()
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    pool.store.append({
        "action_id": "a", "user_id": "u", "action_type": "old",
        "target_kind": None, "target_id": None, "payload": {},
        "source": "web", "ts": past,
    })
    pool.store.append({
        "action_id": "b", "user_id": "u", "action_type": "future",
        "target_kind": None, "target_id": None, "payload": {},
        "source": "web", "ts": future,
    })
    rows = svc.list_for_user("u", since=datetime.now(timezone.utc))
    assert [r.action_type for r in rows] == ["future"]


def test_list_limit_clamped():
    svc, _ = _make_service()
    for i in range(5):
        svc.record(user_id="u", action_type=f"t{i}")
    rows = svc.list_for_user("u", limit=2)
    assert len(rows) == 2


def test_record_isolates_failures(monkeypatch):
    svc, pool = _make_service()
    # Simulate the pool blowing up.
    monkeypatch.setattr(svc, "_get_connection", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert svc.record(user_id="u", action_type="t") is None
    # No exception propagated; store stays empty.
    assert pool.store == []


def test_only_returns_matching_user():
    svc, _ = _make_service()
    svc.record(user_id="alice", action_type="x")
    svc.record(user_id="bob", action_type="x")
    rows = svc.list_for_user("alice")
    assert {r.user_id for r in rows} == {"alice"}
