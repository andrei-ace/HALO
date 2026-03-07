"""Integration tests for the cloud cognitive service against real Gemini endpoints.

Skipped when GOOGLE_API_KEY is not set.

Usage:
    GOOGLE_API_KEY=<key> uv run --project cloud_service pytest cloud_service/tests/test_integration.py -v -s
"""

from __future__ import annotations

import os

import httpx
import pytest
from halo.contracts.serde import snapshot_to_dict

from .conftest import idle_snapshot

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping Gemini integration tests",
)


@pytest.fixture
async def client():
    """In-process ASGI client with lifespan — no separate uvicorn process needed."""
    from cloud_service.app import app
    from cloud_service.deps import lifespan

    transport = httpx.ASGITransport(app=app)
    async with lifespan(app), httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_health_integration(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_decide_idle(client):
    """Idle snapshot + operator command → Gemini returns reasoning + commands."""
    snap = idle_snapshot()
    body = {
        "snapshot": snapshot_to_dict(snap),
        "operator_cmd": "What do you see?",
    }
    resp = await client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "reasoning" in data
    assert isinstance(data["reasoning"], str)
    assert len(data["reasoning"]) > 0
    assert "commands" in data
    assert isinstance(data["commands"], list)
    assert "msg_history" in data


async def test_decide_with_last_msg_id(client):
    """First decide (no last_msg_id) then second with last_msg_id verifies session continuity."""
    snap = idle_snapshot()
    # First call — fresh session
    body1 = {
        "snapshot": snapshot_to_dict(snap),
        "operator_cmd": "What do you see?",
    }
    resp1 = await client.post("/decide", json=body1)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "ok"
    assert "msg_history" in data1
    assert len(data1["msg_history"]) > 0

    # Extract last msg_id from the response
    last_msg_id = data1["msg_history"][-1]["msg_id"]

    # Second call — with last_msg_id for sync
    body2 = {
        "snapshot": snapshot_to_dict(snap),
        "operator_cmd": "Pick the red cube.",
        "last_msg_id": last_msg_id,
    }
    resp2 = await client.post("/decide", json=body2)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "ok"
    assert "reasoning" in data2
    assert isinstance(data2["commands"], list)
