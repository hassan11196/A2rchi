"""Unit tests for ``POST /api/chat``.

We exercise the endpoint with Flask's test client and a fake ``chat_wrapper``
stub attached to ``app.chat_wrapper``.  No database, no real ChatWrapper, no
LLM — the test asserts request validation, error mapping, and the shape of
the success response.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def app(monkeypatch):
    flask = pytest.importorskip("flask")
    # Stub get_services so api.py doesn't attempt a real PostgresServiceFactory.
    from src.interfaces.chat_app import api as api_module

    monkeypatch.setattr(api_module, "get_services", lambda: SimpleNamespace())
    # Suppress the conversation_metadata UPDATE — no DB available.
    monkeypatch.setattr(api_module, "_stamp_archi_service_api", lambda *_a, **_kw: None)

    flask_app = flask.Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.secret_key = "test-secret"
    api_module.register_api(flask_app)
    return flask_app


def _install_chat_wrapper(app, *, return_value):
    """Attach a stub chat_wrapper that records its call and returns *return_value*."""
    calls: list[dict] = []

    def _stub(**kwargs):
        calls.append(kwargs)
        return return_value

    app.chat_wrapper = _stub
    return calls


def test_chat_endpoint_requires_message(app):
    client = app.test_client()
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400
    assert resp.json["error"] == "invalid_request"


def test_chat_endpoint_rejects_invalid_conversation_id(app):
    client = app.test_client()
    resp = client.post("/api/chat", json={
        "message": "hi", "conversation_id": "not-an-int",
    })
    assert resp.status_code == 400


def test_chat_endpoint_503_when_no_chat_wrapper(app):
    app.chat_wrapper = None  # explicitly absent
    client = app.test_client()
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 503
    assert resp.json["error"] == "chat_unavailable"


def test_chat_endpoint_happy_path(app):
    output = SimpleNamespace(answer="hello there")
    calls = _install_chat_wrapper(app, return_value=(output, 42, [101, 102], {}, None))

    client = app.test_client()
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json
    assert body["answer"] == "hello there"
    assert body["conversation_id"] == 42
    assert body["archi_service"] == "api"
    assert body["message_ids"] == [101, 102]
    assert len(calls) == 1
    invocation = calls[0]
    assert invocation["message"] == [("User", "hi")]
    assert invocation["conversation_id"] is None
    assert invocation["is_refresh"] is False
    # client_timeout default clamped between 1 and 300
    assert 1.0 <= invocation["client_timeout"] <= 300.0


def test_chat_endpoint_clamps_client_timeout(app):
    output = SimpleNamespace(answer="ok")
    calls = _install_chat_wrapper(app, return_value=(output, 1, [], {}, None))

    client = app.test_client()
    client.post("/api/chat", json={"message": "hi", "client_timeout": 99999})
    assert calls[0]["client_timeout"] == 300.0


def test_chat_endpoint_passes_through_error_code(app):
    output = None
    _install_chat_wrapper(app, return_value=(output, None, None, {}, 403))

    client = app.test_client()
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 403
    assert resp.json["error"] == "chat_error"


def test_chat_endpoint_502_on_empty_result(app):
    _install_chat_wrapper(app, return_value=(None, None, None, {}, None))

    client = app.test_client()
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 502


def test_chat_endpoint_forwards_conversation_id(app):
    output = SimpleNamespace(answer="ok")
    calls = _install_chat_wrapper(app, return_value=(output, 7, [], {}, None))

    client = app.test_client()
    resp = client.post("/api/chat", json={"message": "hi", "conversation_id": 7})
    assert resp.status_code == 200
    assert calls[0]["conversation_id"] == 7
