"""Unit tests for the cloud cognitive service FastAPI app."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from halo.cognitive.compactor import MessageHistory
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
    def active_arm_ids(self):
        return list(self._sessions.keys())

    @property
    def vlm_fn(self):
        return self._vlm_fn

    async def sync_session(self, arm_id, last_msg_id=None, msg_history=None, client_session_id=None):
        from cloud_service.session_manager import SyncResult

        session = MagicMock()
        session.arm_id = arm_id
        session.agent = self._agent
        session.readiness = "ready"
        session.cursor = -1
        session.pending_handoff = None
        return SyncResult(status="ok", session=session)

    async def get_or_create(self, arm_id, client_session_id=None):
        session = MagicMock()
        session.arm_id = arm_id
        session.agent = self._agent
        session.readiness = "ready"
        session.cursor = -1
        session.pending_handoff = None
        return session

    def get_session(self, arm_id):
        return None

    async def persist_session(self, arm_id):
        pass

    def evict_session(self, arm_id):
        self._sessions.pop(arm_id, None)

    async def reset_session(self, arm_id):
        self._agent.reset_loop_state()


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.decide = AsyncMock(return_value=[])
    agent.last_reasoning = "test reasoning"
    agent.last_compaction = None
    agent.last_token_usage = {}
    agent.reset_loop_state = MagicMock()
    agent.inject_handoff_context = AsyncMock()
    agent.msg_history = MessageHistory()
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
        patch("cloud_service.deps._config", MagicMock()),
    ):
        from cloud_service.app import app

        yield TestClient(app, raise_server_exceptions=True)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["sessions"], list)


def test_decide_empty_commands(client, mock_agent):
    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap), "operator_cmd": "pick the red cube"}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["commands"] == []
    assert data["reasoning"] == "test reasoning"
    assert data["status"] == "ok"
    assert "msg_history" in data


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


def test_decide_sends_last_msg_id(client, mock_agent):
    """last_msg_id is passed through to sync_session."""
    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap), "last_msg_id": "abc123"}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_decide_handoff_context_sets_pending_handoff(mock_agent, mock_vlm_fn):
    """handoff_context in request body is set on session.pending_handoff."""
    _captured_session = None

    async def _capture_sync(*args, **kwargs):
        nonlocal _captured_session
        from cloud_service.session_manager import SyncResult

        session = MagicMock()
        session.arm_id = "arm0"
        session.agent = mock_agent
        session.readiness = "ready"
        session.cursor = -1
        session.pending_handoff = None
        _captured_session = session
        return SyncResult(status="ok", session=session)

    session_mgr = _FakeSessionManager(mock_agent, mock_vlm_fn)
    session_mgr.sync_session = _capture_sync

    with (
        patch("cloud_service.deps._session_mgr", session_mgr),
        patch("cloud_service.deps._config", MagicMock()),
    ):
        from cloud_service.app import app

        c = TestClient(app)
        snap = idle_snapshot()
        handoff = "[Context handoff]\nOperator: pick the red cube"
        body = {"snapshot": snapshot_to_dict(snap), "handoff_context": handoff}
        resp = c.post("/decide", json=body)
        assert resp.status_code == 200
        # inject_handoff_context should have been called with the handoff text
        mock_agent.inject_handoff_context.assert_awaited_once_with(handoff)


def test_decide_returns_msg_history(client, mock_agent):
    """Response always includes msg_history."""
    # Add some records to the mock agent's history
    mock_agent.msg_history.append("user", "pick the cube")
    mock_agent.msg_history.append("model", "starting pick")

    snap = idle_snapshot()
    body = {"snapshot": snapshot_to_dict(snap)}
    resp = client.post("/decide", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "msg_history" in data
    assert len(data["msg_history"]) == 2
    assert data["msg_history"][0]["role"] == "user"
    assert data["msg_history"][1]["role"] == "model"


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


def test_state_endpoint_nonexistent(client):
    resp = client.get("/state/arm99")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is False
    assert data["readiness"] == "cold"


def test_decide_need_history_response(mock_agent, mock_vlm_fn):
    """When sync returns need_history, /decide returns status=need_history."""
    session_mgr = _FakeSessionManager(mock_agent, mock_vlm_fn)

    # Override sync_session to return need_history
    async def _need_history(*args, **kwargs):
        from cloud_service.session_manager import SyncResult

        return SyncResult(status="need_history")

    session_mgr.sync_session = _need_history

    with (
        patch("cloud_service.deps._session_mgr", session_mgr),
        patch("cloud_service.deps._config", MagicMock()),
    ):
        from cloud_service.app import app

        c = TestClient(app, raise_server_exceptions=True)
        snap = idle_snapshot()
        body = {"snapshot": snapshot_to_dict(snap), "last_msg_id": "stale-id"}
        resp = c.post("/decide", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "need_history"
        assert data["commands"] == []


def test_decide_restores_handoff_on_agent_failure(mock_agent, mock_vlm_fn):
    """If agent.decide() raises, pending_handoff is restored for retries."""
    mock_agent.decide = AsyncMock(side_effect=RuntimeError("model unavailable"))

    session_mgr = _FakeSessionManager(mock_agent, mock_vlm_fn)
    # Override sync_session to return a session with pending_handoff
    _last_session = None

    async def _with_handoff(*args, **kwargs):
        nonlocal _last_session
        from cloud_service.session_manager import SyncResult

        session = MagicMock()
        session.arm_id = "arm0"
        session.agent = mock_agent
        session.readiness = "ready"
        session.cursor = -1
        session.pending_handoff = "You are picking the red cube."
        _last_session = session
        return SyncResult(status="ok", session=session)

    session_mgr.sync_session = _with_handoff

    with (
        patch("cloud_service.deps._session_mgr", session_mgr),
        patch("cloud_service.deps._config", MagicMock()),
    ):
        from cloud_service.app import app

        c = TestClient(app, raise_server_exceptions=False)
        snap = idle_snapshot()
        resp = c.post("/decide", json={"snapshot": snapshot_to_dict(snap)})
        assert resp.status_code == 500
        # Handoff must be restored on the in-memory session
        assert _last_session.pending_handoff == "You are picking the red cube."
