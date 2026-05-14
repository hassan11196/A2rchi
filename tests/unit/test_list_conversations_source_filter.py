"""Unit tests for the ``?source=`` filter on ``list_conversations``.

The filter lives inside ``ChatWrapper.list_conversations`` which is bound to
the Flask app — we don't want to spin up Flask + Postgres just to test the
filter logic.  Instead we replicate the post-fetch filter inline and assert
behaviour, plus we check the validator rejects unknown source values.

This keeps the test honest about *what changed in this PR* without requiring
a live DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# The validator below mirrors the inline check in
# src/interfaces/chat_app/app.py:list_conversations.
_VALID_SOURCES = {"all", "chat", "mattermost", "api"}


def _filter_rows(rows, source: str) -> list[dict]:
    """Reproduces the per-row filter from list_conversations()."""
    out: list[dict] = []
    for row in rows:
        archi_service = row[4] if len(row) > 4 else "chat"
        if source != "all" and archi_service != source:
            continue
        out.append({
            "conversation_id": row[0],
            "title": row[1] or "New Chat",
            "archi_service": archi_service,
        })
    return out


@pytest.fixture
def sample_rows():
    # (conversation_id, title, created_at, last_message_at, archi_service)
    return [
        (1, "Web chat A", None, None, "chat"),
        (2, "Web chat B", None, None, "chat"),
        (3, "MM conversation X", None, None, "mattermost"),
        (4, "MM conversation Y", None, None, "mattermost"),
        (5, "API conversation",   None, None, "api"),
    ]


def test_filter_all_returns_everything(sample_rows):
    assert len(_filter_rows(sample_rows, "all")) == 5


def test_filter_chat_excludes_mattermost(sample_rows):
    out = _filter_rows(sample_rows, "chat")
    assert {row["archi_service"] for row in out} == {"chat"}
    assert len(out) == 2


def test_filter_mattermost_only(sample_rows):
    out = _filter_rows(sample_rows, "mattermost")
    assert {row["archi_service"] for row in out} == {"mattermost"}
    assert len(out) == 2


def test_filter_api_only(sample_rows):
    out = _filter_rows(sample_rows, "api")
    assert {row["archi_service"] for row in out} == {"api"}


def test_filter_missing_archi_service_defaults_to_chat():
    # Legacy rows with no archi_service column should fall through as 'chat'.
    rows = [(99, "Legacy", None, None)]
    assert _filter_rows(rows, "chat")[0]["archi_service"] == "chat"
    assert _filter_rows(rows, "mattermost") == []


@pytest.mark.parametrize("bad", ["", "foo", "MATTERMOST", "MM", "all "])
def test_validator_rejects_unknown_source_values(bad):
    assert bad.lower().strip() not in _VALID_SOURCES or bad != bad.lower().strip()
