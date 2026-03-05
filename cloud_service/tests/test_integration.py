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
    assert "nonce" in data


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


async def test_warm_up_and_decide(client):
    """Warm-up then decide verifies session continuity."""
    warm_body = {
        "arm_id": "arm0",
        "state": {
            "ts_ms": 100,
            "epoch": 1,
            "cursor": 0,
            "active_target_handle": None,
            "held_object_handle": None,
            "known_scene_handles": [],
            "last_scene_description": "A table with a red cube and a green cube.",
            "pending_operator_instruction": None,
            "recent_decisions": [],
            "last_snapshot_id": "snap-0",
            "last_arm_id": "arm0",
            "last_skill_phase": None,
            "last_skill_name": None,
            "last_outcome_state": None,
            "recent_event_summaries": [],
            "goal_summary": None,
        },
        "journal": [],
    }
    resp = await client.post("/warm-up", json=warm_body)
    assert resp.status_code == 200
    warm_data = resp.json()
    assert warm_data["readiness"] == "ready"

    # Now decide — session should be warm
    snap = idle_snapshot()
    body = {
        "snapshot": snapshot_to_dict(snap),
        "operator_cmd": "Pick the red cube.",
    }
    resp = await client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "reasoning" in data
    assert isinstance(data["commands"], list)
