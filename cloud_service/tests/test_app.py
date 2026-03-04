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
        self._nonce = "test-nonce-abc123"

    @property
    def nonce(self):
        return self._nonce

    @property
    def active_arm_ids(self):
        return list(self._sessions.keys())

    @property
    def vlm_fn(self):
        return self._vlm_fn

    def get_or_create(self, arm_id):
        session = MagicMock()
        session.arm_id = arm_id
        session.agent = self._agent
        session.readiness = "ready"
        session.cursor = -1
        session.pending_handoff = None
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
    agent.inject_handoff_context = AsyncMock()
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
    data = resp.json()
    assert data["status"] == "ok"
    assert data["nonce"] == "test-nonce-abc123"
    assert isinstance(data["sessions"], list)


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


def _make_test_session_mgr():
    """Create a SessionManager with mocked PlannerAgent creation."""
    from pathlib import Path

    from cloud_service.session_manager import SessionManager

    mgr = SessionManager(
        model_name="test",
        prompts_dir=Path("/tmp"),
        vlm_fn_factory=lambda: MagicMock(),
    )
    # Patch get_or_create to use mock agents instead of real PlannerAgent
    _orig = mgr.get_or_create

    def _patched(arm_id):
        from cloud_service.session_manager import ArmSession

        if arm_id in mgr._sessions:
            session = mgr._sessions[arm_id]
            session.touch()
            return session
        agent = MagicMock()
        agent.reset_loop_state = MagicMock()
        session = ArmSession(arm_id=arm_id, agent=agent)
        session.touch()
        mgr._sessions[arm_id] = session
        return session

    mgr.get_or_create = _patched
    return mgr


def test_warm_up_routes_by_body_arm_id():
    """SessionManager receives correct arm_id (not hardcoded arm0)."""
    mgr = _make_test_session_mgr()
    session = mgr.warm_up_session("arm7", state_dict=None, journal_dicts=[])
    assert session.arm_id == "arm7"
    assert mgr.get_session("arm7") is not None
    assert mgr.get_session("arm0") is None


def test_warm_up_endpoint_reads_body_arm_id(client):
    """/warm-up endpoint passes body arm_id to session manager (not hardcoded arm0)."""
    # Patch the session manager's warm_up_session to capture the arm_id argument
    from unittest.mock import patch as _patch

    with _patch("cloud_service.deps._session_mgr") as patched_mgr:
        session_mock = MagicMock()
        session_mock.readiness = "ready"
        session_mock.cursor = -1
        patched_mgr.warm_up_session.return_value = session_mock

        body = {"arm_id": "arm5", "state": None, "journal": []}
        resp = client.post("/warm-up", json=body)
        assert resp.status_code == 200
        patched_mgr.warm_up_session.assert_called_once()
        call_args = patched_mgr.warm_up_session.call_args
        assert call_args[0][0] == "arm5" or call_args[1].get("arm_id") == "arm5"


def test_reset_clears_pending_handoff():
    """After reset, pending_handoff is cleared so stale context doesn't leak."""
    mgr = _make_test_session_mgr()
    state_dict = {
        "ts_ms": 100,
        "epoch": 1,
        "cursor": 0,
        "active_target_handle": None,
        "held_object_handle": None,
        "known_scene_handles": [],
        "last_scene_description": "table",
        "pending_operator_instruction": None,
        "recent_decisions": [],
        "last_snapshot_id": "snap-1",
        "last_arm_id": "arm0",
        "last_skill_phase": None,
        "last_skill_name": None,
        "last_outcome_state": None,
    }
    session = mgr.warm_up_session("arm0", state_dict=state_dict, journal_dicts=[])
    assert session.pending_handoff is not None

    # Reset should clear it
    mgr.reset_session("arm0")
    session_after = mgr.get_session("arm0")
    assert session_after.pending_handoff is None
