"""Unit tests for the MCP guardrail subsystem.

Covers:
  - classify_tool / is_auto_approved (pure functions)
  - parse_approval_command (text parser)
  - ToolApprovalService against an in-process fake psycopg2 cursor so we don't
    need a live database.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classifier_marks_safe_tool_safe():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    c = classify_tool("search_docs", "Search the knowledge base", {})
    assert c.sensitivity == "safe"
    assert c.requires_approval is False


def test_classifier_heuristic_write_from_name():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    c = classify_tool("delete_user", "Removes a user.", {})
    assert c.sensitivity == "write"
    assert c.requires_approval is True


def test_classifier_heuristic_execute_from_name():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    c = classify_tool("run_shell", "Execute a command on the host.", {})
    assert c.sensitivity == "execute"


def test_classifier_heuristic_write_from_description():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    c = classify_tool("foo", "Updates the user's profile.", {})
    assert c.sensitivity == "write"


def test_tool_overrides_take_precedence():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    # name would classify as "write"; override pins to "safe"
    cfg = {"tool_overrides": {"delete_temp": "safe"}}
    assert classify_tool("delete_temp", "removes temp files", cfg).sensitivity == "safe"

    # the reverse — a "safe" name forced to "execute"
    cfg = {"tool_overrides": {"hello": "execute"}}
    assert classify_tool("hello", "says hi", cfg).sensitivity == "execute"


def test_requires_approval_floor_lifts_safe_to_write():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    cfg = {"requires_approval": True}
    assert classify_tool("hello", "says hi", cfg).sensitivity == "write"


def test_requires_approval_floor_does_not_downgrade_execute():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    cfg = {"requires_approval": "write"}
    assert classify_tool("run_shell", "executes commands", cfg).sensitivity == "execute"


def test_invalid_override_value_falls_back_to_heuristic():
    from src.archi.pipelines.agents.tools.mcp_guardrails import classify_tool

    cfg = {"tool_overrides": {"delete_user": "nonsense"}}
    assert classify_tool("delete_user", "drop a user", cfg).sensitivity == "write"


def test_auto_approve_principals():
    from src.archi.pipelines.agents.tools.mcp_guardrails import (
        classify_tool, is_auto_approved,
    )

    cfg = {"requires_approval": True, "auto_approve_principals": ["admin-bot"]}
    c = classify_tool("delete_user", "drops user", cfg)
    assert is_auto_approved(c, "admin-bot") is True
    assert is_auto_approved(c, "alice") is False
    assert is_auto_approved(c, None) is False


# ---------------------------------------------------------------------------
# Approval-command parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("approve abc123def456", ("approved", "abc123def456")),
        ("Approve  abc123def456", ("approved", "abc123def456")),
        ("/approve abc123def456", ("approved", "abc123def456")),
        ("DENY abc123def456", ("denied", "abc123def456")),
        ("approved: abc123def456", ("approved", "abc123def456")),
        ("hello world", None),
        ("approve", None),
        ("approve short", None),
        ("", None),
    ],
)
def test_parse_approval_command(text, expected):
    from src.archi.pipelines.agents.tools.mcp_guardrails import parse_approval_command

    assert parse_approval_command(text) == expected


# ---------------------------------------------------------------------------
# Fake psycopg2 connection for ToolApprovalService tests
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, store):
        self._store = store
        self._last_result: Any = None
        self._last_rowcount = 0

    @property
    def rowcount(self) -> int:
        return self._last_rowcount

    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        sql_norm = " ".join(sql.split()).upper()
        params = params or ()
        # SELECT current row by full lookup
        if sql_norm.startswith("SELECT APPROVAL_ID, CONVERSATION_ID, MESSAGE_ID, USER_ID, SERVER_NAME, TOOL_NAME, TOOL_ARGS, ARGS_HASH, SENSITIVITY, STATUS, REQUESTED_AT, DECIDED_AT, DECIDED_BY, EXPIRES_AT, SOURCE FROM TOOL_APPROVALS WHERE CONVERSATION_ID IS NOT DISTINCT FROM"):
            conv_id, tool_name, args_hash = params
            now = datetime.now(timezone.utc)
            matches = [
                r for r in self._store
                if r["conversation_id"] == conv_id
                and r["tool_name"] == tool_name
                and r["args_hash"] == args_hash
                and r["status"] in ("approved", "denied", "pending")
                and (r["status"] != "pending" or r["expires_at"] > now)
            ]
            matches.sort(key=lambda r: r["requested_at"], reverse=True)
            self._last_result = _row_tuple(matches[0]) if matches else None
            return
        # SELECT by approval_id
        if sql_norm.startswith("SELECT APPROVAL_ID, CONVERSATION_ID, MESSAGE_ID, USER_ID, SERVER_NAME, TOOL_NAME, TOOL_ARGS, ARGS_HASH, SENSITIVITY, STATUS, REQUESTED_AT, DECIDED_AT, DECIDED_BY, EXPIRES_AT, SOURCE FROM TOOL_APPROVALS WHERE APPROVAL_ID"):
            (aid,) = params
            matches = [r for r in self._store if r["approval_id"] == aid]
            self._last_result = _row_tuple(matches[0]) if matches else None
            return
        if sql_norm.startswith("INSERT INTO TOOL_APPROVALS"):
            keys = [
                "approval_id", "conversation_id", "message_id", "user_id",
                "server_name", "tool_name", "tool_args", "args_hash",
                "sensitivity", "requested_at", "expires_at", "source",
            ]
            tool_args_raw = params[6]
            import json as _json
            tool_args = _json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
            row = dict(zip(keys, params))
            row["tool_args"] = tool_args
            row["status"] = "pending"
            row["decided_at"] = None
            row["decided_by"] = None
            self._store.append(row)
            self._last_result = None
            return
        if sql_norm.startswith("UPDATE TOOL_APPROVALS SET STATUS = 'EXPIRED'"):
            now = datetime.now(timezone.utc)
            count = 0
            for r in self._store:
                if r["status"] == "pending" and r["expires_at"] <= now:
                    r["status"] = "expired"
                    count += 1
            self._last_rowcount = count
            self._last_result = None
            return
        if sql_norm.startswith("UPDATE TOOL_APPROVALS SET STATUS"):
            decision, decided_at, decided_by, approval_id = params
            now = datetime.now(timezone.utc)
            updated = None
            for r in self._store:
                if (
                    r["approval_id"] == approval_id
                    and r["status"] == "pending"
                    and r["expires_at"] > now
                ):
                    r["status"] = decision
                    r["decided_at"] = decided_at
                    r["decided_by"] = decided_by
                    updated = r
                    break
            self._last_result = (approval_id,) if updated else None
            return
        if sql_norm.startswith("CREATE TABLE") or sql_norm.startswith("CREATE INDEX"):
            return
        raise AssertionError(f"FakeCursor doesn't know how to handle: {sql_norm[:100]}")

    def fetchone(self):
        return self._last_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _row_tuple(row: dict) -> tuple:
    return (
        row["approval_id"], row["conversation_id"], row["message_id"], row["user_id"],
        row["server_name"], row["tool_name"], row["tool_args"], row["args_hash"],
        row["sensitivity"], row["status"], row["requested_at"], row["decided_at"],
        row["decided_by"], row["expires_at"], row["source"],
    )


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
    """Minimum-viable pool that hands out FakeConn instances backed by a list."""

    def __init__(self):
        self.store: List[dict] = []

    def get_connection(self):
        return FakeConn(self.store)

    def release_connection(self, conn):
        pass


# ---------------------------------------------------------------------------
# ToolApprovalService end-to-end
# ---------------------------------------------------------------------------


def _make_service():
    from src.utils.tool_approval_service import ToolApprovalService

    pool = FakePool()
    svc = ToolApprovalService(connection_pool=pool)
    return svc, pool


def test_create_pending_returns_pending_approval():
    svc, pool = _make_service()
    approval = svc.create_pending(
        conversation_id=42, message_id=None, user_id="alice",
        server_name="github", tool_name="create_issue",
        tool_args={"title": "Bug"}, sensitivity="write",
    )
    assert approval.status == "pending"
    assert approval.tool_name == "create_issue"
    assert approval.expires_at > approval.requested_at
    assert len(pool.store) == 1


def test_find_decision_returns_recent_pending():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    found = svc.find_decision(
        conversation_id=1, tool_name="t", args_hash=a.args_hash,
    )
    assert found is not None
    assert found.approval_id == a.approval_id
    assert found.status == "pending"


def test_decide_approves_pending_row():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    decided = svc.decide(a.approval_id, decision="approved", decided_by="alice")
    assert decided is not None
    assert decided.status == "approved"
    assert decided.decided_by == "alice"
    assert decided.decided_at is not None


def test_decide_denies_pending_row():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    decided = svc.decide(a.approval_id, decision="denied", decided_by="alice")
    assert decided is not None
    assert decided.status == "denied"


def test_decide_rejects_already_decided_row():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    svc.decide(a.approval_id, decision="approved")
    # Second decide should no-op (returns None).
    again = svc.decide(a.approval_id, decision="denied")
    assert again is None


def test_decide_rejects_invalid_decision():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    with pytest.raises(ValueError):
        svc.decide(a.approval_id, decision="maybe")  # type: ignore[arg-type]


def test_find_decision_after_approve_returns_approved():
    svc, _ = _make_service()
    a = svc.create_pending(
        conversation_id=99, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={"x": 1},
        sensitivity="write",
    )
    svc.decide(a.approval_id, decision="approved")
    found = svc.find_decision(
        conversation_id=99, tool_name="t", args_hash=a.args_hash,
    )
    assert found is not None and found.status == "approved"


def test_hash_tool_args_stable_for_dict_order():
    from src.utils.tool_approval_service import hash_tool_args

    assert hash_tool_args({"a": 1, "b": 2}) == hash_tool_args({"b": 2, "a": 1})
    assert hash_tool_args({"a": 1}) != hash_tool_args({"a": 2})


def test_expire_stale_marks_overdue_pending():
    from src.utils.tool_approval_service import ToolApprovalService

    pool = FakePool()
    svc = ToolApprovalService(
        connection_pool=pool,
        default_ttl=timedelta(seconds=-1),  # already expired on create
    )
    svc.create_pending(
        conversation_id=1, message_id=None, user_id=None,
        server_name="srv", tool_name="t", tool_args={},
        sensitivity="write",
    )
    expired = svc.expire_stale()
    assert expired == 1
    assert pool.store[0]["status"] == "expired"
