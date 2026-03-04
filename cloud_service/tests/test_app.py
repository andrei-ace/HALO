"""Unit tests for the cloud cognitive service FastAPI app."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from halo.contracts.commands import CommandEnvelope, DescribeScenePayload
from halo.contracts.enums import CommandType
from halo.contracts.serde import snapshot_to_dict
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene

from .conftest import idle_snapshot


class _FakeSessionManager:
    """Minimal mock for SessionManager that works with endpoint code."""

    def __init__(self, agent, vlm_fn):
        self._agent = agent
        self._vlm_fn = vlm_fn
        self._sessions = {}

    @property
    def vlm_fn(self):
        return self._vlm_fn

    def get_or_create(self, arm_id):
        session = MagicMock()
        session.arm_id = arm_id
        session.agent = self._agent
        session.readiness = "ready"
        session.cursor = -1
        return session

    def get_session(self, arm_id):
        return None

    def warm_up_session(self, arm_id, state_dict, journal_dicts):
        session = MagicMock()
        session.readiness = "ready"
        session.cursor = len(journal_dicts) - 1 if journal_dicts else -1
        return session

    def reset_session(self, arm_id):
        self._agent.reset_loop_state()

    def reset_all(self):
        self._agent.reset_loop_state()


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.decide = AsyncMock(return_value=[])
    agent.last_reasoning = "test reasoning"
    agent.reset_loop_state = MagicMock()
    return agent


@pytest.fixture
def mock_vlm_fn():
    return AsyncMock(
        return_value=VlmScene(
            scene="A table with objects",
            detections=[
                VlmDetection(
                    handle="red_cube_01",
                    label="red cube",
                    bbox=(0.1, 0.2, 0.3, 0.4),
                    centroid=(0.2, 0.3),
                    is_graspable=True,
                ),
            ],
        )
    )


@pytest.fixture
def client(mock_agent, mock_vlm_fn):
    session_mgr = _FakeSessionManager(mock_agent, mock_vlm_fn)
    with (
        patch("cloud_service.deps._session_mgr", session_mgr),
        patch("cloud_service.deps._config", MagicMock(cloud_api_key="")),
    ):
        from cloud_service.app import app

        yield TestClient(app, raise_server_exceptions=True)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_decide_empty_commands(client, mock_agent):
    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap), "operator_cmd": "pick the red cube"}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["commands"] == []
    assert data["reasoning"] == "test reasoning"


def test_decide_with_commands(client, mock_agent):
    cmd = CommandEnvelope(
        command_id="cmd-1",
        arm_id="arm0",
        issued_at_ms=1000,
        type=CommandType.DESCRIBE_SCENE,
        payload=DescribeScenePayload(reason="initial"),
        precondition_snapshot_id=None,
    )
    mock_agent.decide = AsyncMock(return_value=[cmd])

    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap)}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["commands"]) == 1
    assert data["commands"][0]["type"] == "DESCRIBE_SCENE"
    assert data["commands"][0]["command_id"] == "cmd-1"


def test_decide_passes_epoch(client, mock_agent):
    """Epoch is extracted from request body and passed to agent.decide()."""
    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap), "epoch": 42}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    mock_agent.decide.assert_awaited_once()
    _, kwargs = mock_agent.decide.call_args
    assert kwargs["epoch"] == 42


def test_vlm_scene(client, mock_vlm_fn):
    # Create a minimal JPEG (1x1 pixel)
    import cv2
    import numpy as np

    img = np.zeros((1, 1, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    jpeg_bytes = buf.tobytes()

    metadata = json.dumps({"arm_id": "arm0", "known_handles": ["red_cube_01"], "target_handle": "red_cube_01"})
    resp = client.post(
        "/vlm/scene",
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        data={"metadata": metadata},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene"] == "A table with objects"
    assert len(data["detections"]) == 1
    assert data["detections"][0]["handle"] == "red_cube_01"


def test_warm_up(client):
    body = {
        "state": {
            "ts_ms": 100,
            "epoch": 1,
            "cursor": 1,
            "active_target_handle": "cube",
            "held_object_handle": None,
            "known_scene_handles": ["cube"],
            "last_scene_description": "table",
            "pending_operator_instruction": None,
            "recent_decisions": [],
            "last_snapshot_id": "snap-1",
            "last_arm_id": "arm0",
            "last_skill_phase": None,
            "last_skill_name": None,
            "last_outcome_state": None,
        },
        "journal": [
            {
                "cursor": 0,
                "ts_ms": 10,
                "epoch": 1,
                "backend": "local",
                "entry_type": "decision",
                "summary": "d0",
                "data": {},
            },
        ],
    }
    resp = client.post("/warm-up", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["readiness"] == "ready"
    assert "cursor" in data


def test_state_endpoint_nonexistent(client):
    resp = client.get("/state/arm99")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is False
    assert data["readiness"] == "cold"


def test_reset(client, mock_agent):
    resp = client.post("/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    mock_agent.reset_loop_state.assert_called_once()


def test_reset_arm(client, mock_agent):
    resp = client.post("/reset/arm0")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_auth_required():
    """When cloud_api_key is set, requests without auth are rejected."""
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()

    session_mgr = _FakeSessionManager(mock_agent, AsyncMock())

    with (
        patch("cloud_service.deps._session_mgr", session_mgr),
        patch("cloud_service.deps._config", MagicMock(cloud_api_key="secret-key")),
    ):
        from cloud_service.app import app

        c = TestClient(app, raise_server_exceptions=False)

        # No auth header
        resp = c.post("/decide", json={"snapshot": {}, "operator_cmd": None})
        assert resp.status_code == 401

        # Wrong key
        resp = c.post("/decide", json={"snapshot": {}, "operator_cmd": None}, headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

        # Correct key
        snap = idle_snapshot()
        body = {"snapshot": snapshot_to_dict(snap)}
        resp = c.post("/decide", json=body, headers={"Authorization": "Bearer secret-key"})
        assert resp.status_code == 200
