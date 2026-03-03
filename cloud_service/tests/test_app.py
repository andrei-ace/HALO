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
    # Patch the singletons in deps before importing app
    with (
        patch("cloud_service.deps._agent", mock_agent),
        patch("cloud_service.deps._vlm_fn", mock_vlm_fn),
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


def test_reset(client, mock_agent):
    resp = client.post("/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    mock_agent.reset_loop_state.assert_called_once()


def test_auth_required():
    """When cloud_api_key is set, requests without auth are rejected."""
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()

    with (
        patch("cloud_service.deps._agent", mock_agent),
        patch("cloud_service.deps._vlm_fn", AsyncMock()),
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
