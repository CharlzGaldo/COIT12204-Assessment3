"""
Tests for the FastAPI HTTP layer using Starlette's TestClient.

These tests never call the real agent logic — they monkeypatch
app.main.classify_priority so we're purely testing the endpoint's
request validation, error handling, and response formatting (the
"Endpoint Integration" required feature), independent of agent
correctness (which is covered separately in test_agent.py).
"""
from fastapi.testclient import TestClient

from app.main import app

# One shared TestClient for all tests in this file — it doesn't start a
# real network server, it calls the FastAPI app directly in-process.
client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_priority_endpoint_success(monkeypatch):
    """
    Patch classify_priority (as imported into app.main) so this test
    doesn't depend on the real agent or any LLM. We're only checking
    that the endpoint calls it correctly and formats the response.
    """
    def fake_classify(task_description):
        assert task_description == "Fix the production outage"
        return {"priority": "High", "reasoning": "mocked reasoning", "fallback": False}

    # Important: patch the name as it exists in app.main's namespace
    # (it was imported there with `from app.agent import classify_priority`),
    # not app.agent.classify_priority directly.
    monkeypatch.setattr("app.main.classify_priority", fake_classify)

    resp = client.post(
        "/api/agent/classify_priority",
        json={"task_description": "Fix the production outage"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["priority"] == "High"
    assert body["fallback"] is False


def test_classify_priority_endpoint_blank_input_returns_422():
    """Whitespace-only input should be rejected by pydantic's field_validator."""
    resp = client.post("/api/agent/classify_priority", json={"task_description": "   "})
    assert resp.status_code == 422


def test_classify_priority_endpoint_missing_field_returns_422():
    """Missing the required task_description field entirely should also 422."""
    resp = client.post("/api/agent/classify_priority", json={})
    assert resp.status_code == 422


def test_classify_priority_endpoint_agent_error_returns_500(monkeypatch):
    """
    If classify_priority() somehow raises an unexpected exception (not the
    normal fallback path — a genuinely unhandled error), the endpoint
    should convert it into a 500 rather than leaking a stack trace.
    """
    def fake_classify(task_description):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr("app.main.classify_priority", fake_classify)

    resp = client.post(
        "/api/agent/classify_priority",
        json={"task_description": "anything"},
    )
    assert resp.status_code == 500


def test_state_endpoint_returns_dict():
    """Sanity check that /api/agent/state returns the expected shape."""
    resp = client.get("/api/agent/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "last_suggestion" in body
    assert "error_count" in body
    assert "history" in body
