"""UserActionService — write-operation audit log.

archi already has narrow audit trails (``config_audit`` for config edits,
``agent_tool_calls`` for tool invocations, ``rbac.audit`` for permission
checks).  But there's no single timeline an end user can look at to see
"what has been done on my behalf?" — change a preference here, set an API
key there, approve a write tool somewhere else, the user has to piece it
together themselves.

This service is the missing single source of truth.  Each write operation
performed *for* or *by* a user appends one row to ``user_actions``.  The
``/api/users/me/actions`` endpoint then renders that as a timeline.

Schema mirrors the other audit tables in the project: PK action_id (uuid
hex), foreign key to ``users`` only logically (we keep history for deleted
users), JSONB ``payload`` for opaque per-action context, indexed on
``(user_id, ts DESC)`` and ``(action_type, ts DESC)`` for typical queries.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2

from src.utils.logging import get_logger

logger = get_logger(__name__)


_VALID_SOURCES = {"web", "api", "mattermost", "agent", "system"}


SQL_CREATE_USER_ACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS user_actions (
    action_id   VARCHAR(64) PRIMARY KEY,
    user_id     VARCHAR(200),
    action_type VARCHAR(100) NOT NULL,
    target_kind VARCHAR(100),
    target_id   VARCHAR(200),
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    source      VARCHAR(20) NOT NULL DEFAULT 'web',
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SQL_INDEX_USER_ACTIONS_USER_TS = (
    "CREATE INDEX IF NOT EXISTS idx_user_actions_user_ts "
    "ON user_actions (user_id, ts DESC)"
)
SQL_INDEX_USER_ACTIONS_TYPE_TS = (
    "CREATE INDEX IF NOT EXISTS idx_user_actions_type_ts "
    "ON user_actions (action_type, ts DESC)"
)


@dataclass
class UserAction:
    action_id: str
    user_id: Optional[str]
    action_type: str
    target_kind: Optional[str]
    target_id: Optional[str]
    payload: Dict[str, Any]
    source: str
    ts: datetime


class UserActionService:
    """Thin CRUD wrapper over ``user_actions``.

    Designed to be cheap and fire-and-forget: callers should swallow any
    exception from ``record`` so an audit-write failure never breaks the
    user-facing operation it's auditing.
    """

    def __init__(
        self,
        connection_pool=None,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        self._pool = connection_pool
        self._conn_params = connection_params

    # ---------------------------------------------------------------- conns
    def _get_connection(self):
        if self._pool:
            return self._pool.get_connection_direct()
        if self._conn_params:
            return psycopg2.connect(**self._conn_params)
        raise ValueError("UserActionService needs a connection pool or params")

    def _release_connection(self, conn) -> None:
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()

    # ---------------------------------------------------------------- schema
    def ensure_schema(self) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(SQL_CREATE_USER_ACTIONS_TABLE)
                cur.execute(SQL_INDEX_USER_ACTIONS_USER_TS)
                cur.execute(SQL_INDEX_USER_ACTIONS_TYPE_TS)
            conn.commit()
        finally:
            self._release_connection(conn)

    # ---------------------------------------------------------------- writes
    def record(
        self,
        *,
        user_id: Optional[str],
        action_type: str,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "web",
        ts: Optional[datetime] = None,
    ) -> Optional[UserAction]:
        """Append a single row.  Returns the new ``UserAction`` or None on error.

        This method *intentionally* never raises: an audit write failing must
        not break the user-facing operation that triggered it.  Callers can
        check the return value (None == failed to record) if they care.
        """
        if not action_type:
            logger.warning("UserActionService.record called with empty action_type")
            return None
        if source not in _VALID_SOURCES:
            logger.warning("UserActionService: unknown source=%r; defaulting to 'web'", source)
            source = "web"
        action_id = secrets.token_hex(16)
        ts = ts or datetime.now(timezone.utc)
        payload_json = json.dumps(payload or {}, default=str)

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_actions
                            (action_id, user_id, action_type, target_kind,
                             target_id, payload, source, ts)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (action_id, user_id, action_type, target_kind,
                         target_id, payload_json, source, ts),
                    )
                conn.commit()
            finally:
                self._release_connection(conn)
        except Exception as exc:
            logger.warning(
                "UserActionService: failed to record %s for user=%s: %s",
                action_type, user_id, exc,
            )
            return None

        return UserAction(
            action_id=action_id, user_id=user_id, action_type=action_type,
            target_kind=target_kind, target_id=target_id,
            payload=payload or {}, source=source, ts=ts,
        )

    # ---------------------------------------------------------------- lookups
    def list_for_user(
        self,
        user_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        action_types: Optional[List[str]] = None,
    ) -> List[UserAction]:
        """Return up to *limit* most recent actions for *user_id*."""
        limit = max(1, min(int(limit), 1000))

        clauses = ["user_id = %s"]
        params: List[Any] = [user_id]
        if since:
            clauses.append("ts >= %s")
            params.append(since)
        if action_types:
            clauses.append("action_type = ANY(%s)")
            params.append(list(action_types))
        where = " AND ".join(clauses)
        params.append(limit)

        sql = f"""
            SELECT action_id, user_id, action_type, target_kind, target_id,
                   payload, source, ts
            FROM user_actions
            WHERE {where}
            ORDER BY ts DESC
            LIMIT %s
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        finally:
            self._release_connection(conn)
        return [_row_to_action(r) for r in rows]


def _row_to_action(row) -> UserAction:
    action_id, user_id, action_type, target_kind, target_id, payload, source, ts = row
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"_raw": payload}
    return UserAction(
        action_id=action_id, user_id=user_id, action_type=action_type,
        target_kind=target_kind, target_id=target_id,
        payload=payload or {}, source=source, ts=ts,
    )
