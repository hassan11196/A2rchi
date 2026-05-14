"""Tool approval service — persistence layer for the MCP guardrail flow.

Each row in ``tool_approvals`` represents one (conversation, tool, args)
trigger of an approval prompt.  The lifecycle is::

    pending  --approve-->  approved
             --deny-->     denied
             --expire-->   expired   (after the TTL)

Once a row is ``approved`` or ``denied`` the agent's tool wrapper consults
it to decide whether to run the tool or short-circuit with a denial message.
Decisions are cached on the row so a re-asked identical call (same
conversation, tool, args hash) can reuse the prior choice.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import psycopg2

from src.utils.logging import get_logger

logger = get_logger(__name__)

ApprovalStatus = Literal["pending", "approved", "denied", "expired"]


# Default time-to-live for a pending approval before it expires.  Should be
# long enough that a user can reasonably react in the UI/Mattermost, short
# enough that abandoned prompts don't pile up.
DEFAULT_APPROVAL_TTL = timedelta(minutes=10)


SQL_CREATE_TOOL_APPROVALS_TABLE = """
CREATE TABLE IF NOT EXISTS tool_approvals (
    approval_id     VARCHAR(64) PRIMARY KEY,
    conversation_id INTEGER,
    message_id      INTEGER,
    user_id         VARCHAR(200),
    server_name     VARCHAR(200) NOT NULL,
    tool_name       VARCHAR(200) NOT NULL,
    tool_args       JSONB NOT NULL DEFAULT '{}'::jsonb,
    args_hash       VARCHAR(64) NOT NULL,
    sensitivity     VARCHAR(20) NOT NULL DEFAULT 'write',
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at      TIMESTAMPTZ,
    decided_by      VARCHAR(200),
    expires_at      TIMESTAMPTZ NOT NULL,
    source          VARCHAR(20) NOT NULL DEFAULT 'chat'
);
"""

SQL_INDEX_TOOL_APPROVALS_LOOKUP = (
    "CREATE INDEX IF NOT EXISTS idx_tool_approvals_lookup "
    "ON tool_approvals (conversation_id, tool_name, args_hash, status)"
)
SQL_INDEX_TOOL_APPROVALS_STATUS = (
    "CREATE INDEX IF NOT EXISTS idx_tool_approvals_status_expires "
    "ON tool_approvals (status, expires_at)"
)


@dataclass
class ToolApproval:
    """In-memory mirror of one ``tool_approvals`` row."""

    approval_id: str
    conversation_id: Optional[int]
    message_id: Optional[int]
    user_id: Optional[str]
    server_name: str
    tool_name: str
    tool_args: Dict[str, Any]
    args_hash: str
    sensitivity: str
    status: ApprovalStatus
    requested_at: datetime
    decided_at: Optional[datetime]
    decided_by: Optional[str]
    expires_at: datetime
    source: str

    @property
    def is_terminal(self) -> bool:
        return self.status in ("approved", "denied", "expired")


def hash_tool_args(tool_args: Any) -> str:
    """Deterministic hash for tool arguments, used to dedupe identical calls."""
    try:
        canonical = json.dumps(tool_args, sort_keys=True, default=str)
    except Exception:
        canonical = repr(tool_args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class ToolApprovalService:
    """CRUD helper for ``tool_approvals``.

    All methods accept either an injected connection pool (preferred for
    deployment use) or raw connection params (for tests / scripts).  The
    service is intentionally thin so it can be unit-tested with a fake
    cursor.
    """

    def __init__(
        self,
        connection_pool=None,
        connection_params: Optional[Dict[str, Any]] = None,
        *,
        default_ttl: timedelta = DEFAULT_APPROVAL_TTL,
    ):
        self._pool = connection_pool
        self._conn_params = connection_params
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------ conns
    def _get_connection(self):
        if self._pool:
            return self._pool.get_connection_direct()
        if self._conn_params:
            return psycopg2.connect(**self._conn_params)
        raise ValueError("ToolApprovalService needs a connection pool or params")

    def _release_connection(self, conn) -> None:
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> None:
        """Idempotently create the ``tool_approvals`` table and indexes."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(SQL_CREATE_TOOL_APPROVALS_TABLE)
                cur.execute(SQL_INDEX_TOOL_APPROVALS_LOOKUP)
                cur.execute(SQL_INDEX_TOOL_APPROVALS_STATUS)
            conn.commit()
        finally:
            self._release_connection(conn)

    # ------------------------------------------------------------------ lookups
    def find_decision(
        self,
        *,
        conversation_id: Optional[int],
        tool_name: str,
        args_hash: str,
    ) -> Optional[ToolApproval]:
        """Return the most recent non-expired terminal decision, if any."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT approval_id, conversation_id, message_id, user_id, server_name,
                           tool_name, tool_args, args_hash, sensitivity, status,
                           requested_at, decided_at, decided_by, expires_at, source
                    FROM tool_approvals
                    WHERE conversation_id IS NOT DISTINCT FROM %s
                      AND tool_name = %s
                      AND args_hash = %s
                      AND status IN ('approved','denied','pending')
                      AND (status != 'pending' OR expires_at > NOW())
                    ORDER BY requested_at DESC
                    LIMIT 1
                    """,
                    (conversation_id, tool_name, args_hash),
                )
                row = cur.fetchone()
                return _row_to_approval(row) if row else None
        finally:
            self._release_connection(conn)

    def get(self, approval_id: str) -> Optional[ToolApproval]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT approval_id, conversation_id, message_id, user_id, server_name,
                           tool_name, tool_args, args_hash, sensitivity, status,
                           requested_at, decided_at, decided_by, expires_at, source
                    FROM tool_approvals
                    WHERE approval_id = %s
                    """,
                    (approval_id,),
                )
                row = cur.fetchone()
                return _row_to_approval(row) if row else None
        finally:
            self._release_connection(conn)

    # ------------------------------------------------------------------ writes
    def create_pending(
        self,
        *,
        conversation_id: Optional[int],
        message_id: Optional[int],
        user_id: Optional[str],
        server_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        sensitivity: str,
        source: str = "chat",
        ttl: Optional[timedelta] = None,
    ) -> ToolApproval:
        """Insert a new ``pending`` row and return it."""
        approval_id = secrets.token_hex(16)
        now = datetime.now(timezone.utc)
        expires_at = now + (ttl or self._default_ttl)
        args_hash = hash_tool_args(tool_args)
        payload = json.dumps(tool_args or {})

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_approvals (
                        approval_id, conversation_id, message_id, user_id,
                        server_name, tool_name, tool_args, args_hash,
                        sensitivity, status, requested_at, expires_at, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
                            'pending', %s, %s, %s)
                    """,
                    (
                        approval_id, conversation_id, message_id, user_id,
                        server_name, tool_name, payload, args_hash,
                        sensitivity, now, expires_at, source,
                    ),
                )
            conn.commit()
        finally:
            self._release_connection(conn)

        return ToolApproval(
            approval_id=approval_id,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            server_name=server_name,
            tool_name=tool_name,
            tool_args=tool_args,
            args_hash=args_hash,
            sensitivity=sensitivity,
            status="pending",
            requested_at=now,
            decided_at=None,
            decided_by=None,
            expires_at=expires_at,
            source=source,
        )

    def decide(
        self,
        approval_id: str,
        *,
        decision: ApprovalStatus,
        decided_by: Optional[str] = None,
    ) -> Optional[ToolApproval]:
        """Mark a pending approval as ``approved`` or ``denied``."""
        if decision not in ("approved", "denied"):
            raise ValueError(f"Invalid decision: {decision!r}")
        now = datetime.now(timezone.utc)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tool_approvals
                    SET status = %s,
                        decided_at = %s,
                        decided_by = %s
                    WHERE approval_id = %s
                      AND status = 'pending'
                      AND expires_at > NOW()
                    RETURNING approval_id
                    """,
                    (decision, now, decided_by, approval_id),
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            self._release_connection(conn)
        if not row:
            return None
        return self.get(approval_id)

    def record_resolved(
        self,
        *,
        conversation_id: Optional[int],
        message_id: Optional[int],
        user_id: Optional[str],
        server_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        sensitivity: str,
        status: ApprovalStatus,
        decided_by: str,
        source: str = "auto",
        ttl: Optional[timedelta] = None,
    ) -> ToolApproval:
        """Insert a row that's already decided.

        Used when the guardrail short-circuits a write/execute tool call
        without an interactive prompt — e.g. ``acceptEdits`` /
        ``bypassPermissions`` / always-allow list / ``plan`` denial. The
        row is created with the terminal status (``approved`` or
        ``denied``) and ``decided_by`` should describe *why* it was
        decided automatically (e.g. ``"auto:bypassPermissions"``,
        ``"auto:always-allow"``, ``"auto:plan-denied"``). Required for the
        audit history view to show non-interactive decisions.
        """
        if status not in ("approved", "denied"):
            raise ValueError(f"record_resolved expects approved|denied, got {status!r}")
        approval_id = secrets.token_hex(16)
        now = datetime.now(timezone.utc)
        expires_at = now + (ttl or self._default_ttl)
        args_hash = hash_tool_args(tool_args)
        payload = json.dumps(tool_args or {})

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_approvals (
                        approval_id, conversation_id, message_id, user_id,
                        server_name, tool_name, tool_args, args_hash,
                        sensitivity, status, requested_at, decided_at,
                        decided_by, expires_at, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
                            %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        approval_id, conversation_id, message_id, user_id,
                        server_name, tool_name, payload, args_hash,
                        sensitivity, status, now, now,
                        decided_by, expires_at, source,
                    ),
                )
            conn.commit()
        finally:
            self._release_connection(conn)

        return ToolApproval(
            approval_id=approval_id,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            server_name=server_name,
            tool_name=tool_name,
            tool_args=tool_args,
            args_hash=args_hash,
            sensitivity=sensitivity,
            status=status,
            requested_at=now,
            decided_at=now,
            decided_by=decided_by,
            expires_at=expires_at,
            source=source,
        )

    def list_recent(
        self,
        *,
        user_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolApproval]:
        """Recent approvals (most-recent-first), optionally filtered.

        Powers the "Approval history" view in Settings. Keep the result
        bounded (default 100 rows) so a long-running user doesn't pull
        thousands of rows down the wire.
        """
        limit = max(1, min(int(limit or 100), 500))
        clauses: List[str] = []
        params: List[Any] = []
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        if conversation_id is not None:
            clauses.append("conversation_id = %s")
            params.append(conversation_id)
        if status:
            clauses.append("status = %s")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT approval_id, conversation_id, message_id, user_id, server_name,
                           tool_name, tool_args, args_hash, sensitivity, status,
                           requested_at, decided_at, decided_by, expires_at, source
                    FROM tool_approvals
                    {where}
                    ORDER BY requested_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall() or []
        finally:
            self._release_connection(conn)
        return [_row_to_approval(r) for r in rows]

    def expire_stale(self) -> int:
        """Mark every overdue pending row as ``expired``.  Returns the count."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tool_approvals
                    SET status = 'expired'
                    WHERE status = 'pending' AND expires_at <= NOW()
                    """,
                )
                count = cur.rowcount
            conn.commit()
        finally:
            self._release_connection(conn)
        return count


def _row_to_approval(row) -> ToolApproval:
    (approval_id, conversation_id, message_id, user_id, server_name,
     tool_name, tool_args, args_hash, sensitivity, status,
     requested_at, decided_at, decided_by, expires_at, source) = row
    return ToolApproval(
        approval_id=approval_id,
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        server_name=server_name,
        tool_name=tool_name,
        tool_args=tool_args if isinstance(tool_args, dict) else (json.loads(tool_args) if tool_args else {}),
        args_hash=args_hash,
        sensitivity=sensitivity,
        status=status,
        requested_at=requested_at,
        decided_at=decided_at,
        decided_by=decided_by,
        expires_at=expires_at,
        source=source,
    )
