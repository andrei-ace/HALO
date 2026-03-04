"""Unit tests for CognitiveBackend protocol, LocalCognitiveBackend, CloudCognitiveBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from halo.cognitive.backend import CognitiveBackend, WarmableBackend
from halo.cognitive.cloud_backend import CloudCognitiveBackend
from halo.cognitive.config import BackendReadiness, BackendType, CloudConfig, CognitiveConfig, LocalConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry
from halo.cognitive.local_backend import LocalCognitiveBackend
from halo.contracts.enums import (
    ActStatus,
    CommandType,
    PerceptionFailureCode,
    SafetyState,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    TargetInfo,
)
from halo.services.target_perception_service.vlm_parser import VlmScene


def _idle_snap() -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-001",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle=None,
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
        held_object_handle=None,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CognitiveConfig()
    assert cfg.active == BackendType.LOCAL
    assert cfg.local.planner_model == "gpt-oss:20b"
    assert cfg.cloud.planner_model == "gemini-2.5-flash"
    assert cfg.enable_failover is False


def test_backend_type_enum():
    assert BackendType.LOCAL == "local"
    assert BackendType.CLOUD == "cloud"


def test_cloud_config_service_url():
    cfg = CloudConfig(service_url="https://example.com", api_key="test-key")
    assert cfg.service_url == "https://example.com"
    assert cfg.api_key == "test-key"
    assert cfg.request_timeout_s == 30.0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_local_backend_is_cognitive_backend():
    """LocalCognitiveBackend satisfies the CognitiveBackend protocol."""
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(return_value=[]), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch(
            "halo.cognitive.local_backend.make_ollama_vlm_fn",
            return_value=AsyncMock(return_value=VlmScene(scene="", detections=[])),
        ),
    ):
        backend = LocalCognitiveBackend()
    assert isinstance(backend, CognitiveBackend)


def test_cloud_backend_is_cognitive_backend():
    """CloudCognitiveBackend satisfies the CognitiveBackend protocol."""
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://localhost:8080"))
    assert isinstance(backend, CognitiveBackend)


# ---------------------------------------------------------------------------
# LocalCognitiveBackend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_backend_decide():
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = "thinking..."
    mock_agent.reset_loop_state = MagicMock()

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch(
            "halo.cognitive.local_backend.make_ollama_vlm_fn",
            return_value=AsyncMock(),
        ),
    ):
        backend = LocalCognitiveBackend()

    snap = _idle_snap()
    cmds = await backend.decide(snap, operator_cmd="pick cube")
    assert cmds == []
    mock_agent.decide.assert_awaited_once_with(snap, operator_cmd="pick cube", epoch=None)
    assert backend.last_reasoning == "thinking..."
    assert backend.backend_type == "local"


@pytest.mark.asyncio
async def test_local_backend_vlm_scene():
    mock_vlm = AsyncMock(return_value=VlmScene(scene="table", detections=[]))

    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch("halo.cognitive.local_backend.make_ollama_vlm_fn", return_value=mock_vlm),
    ):
        backend = LocalCognitiveBackend()

    scene = await backend.vlm_scene("arm0", b"fake-image")
    assert scene.scene == "table"


@pytest.mark.asyncio
async def test_local_backend_health_check_failure():
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch("halo.cognitive.local_backend.make_ollama_vlm_fn", return_value=AsyncMock()),
    ):
        backend = LocalCognitiveBackend(config=LocalConfig(base_url="http://localhost:99999"))

    healthy = await backend.health_check()
    assert healthy is False


# ---------------------------------------------------------------------------
# CloudCognitiveBackend (HTTP client)
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, json_data: dict) -> httpx.Response:
    """Create an httpx.Response with a request attached (required for raise_for_status)."""
    return httpx.Response(status_code, json=json_data, request=httpx.Request("POST", "http://test:8080"))


@pytest.mark.asyncio
async def test_cloud_backend_decide():
    """Cloud backend posts to /decide and returns deserialized commands."""
    response_json = {
        "commands": [
            {
                "command_id": "cmd-1",
                "arm_id": "arm0",
                "issued_at_ms": 1000,
                "type": "DESCRIBE_SCENE",
                "payload": {"reason": "initial"},
                "precondition_snapshot_id": None,
            }
        ],
        "reasoning": "cloud thinking",
    }

    mock_response = _mock_response(200, response_json)
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        snap = _idle_snap()
        cmds = await backend.decide(snap, operator_cmd="scan")

    assert len(cmds) == 1
    assert cmds[0].type == CommandType.DESCRIBE_SCENE
    assert cmds[0].command_id == "cmd-1"
    assert backend.last_reasoning == "cloud thinking"
    assert backend.backend_type == "cloud"


@pytest.mark.asyncio
async def test_cloud_backend_vlm_scene():
    """Cloud backend posts JPEG + metadata to /vlm/scene."""
    response_json = {
        "scene": "A table with cubes",
        "detections": [
            {
                "handle": "red_cube_01",
                "label": "red cube",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "centroid": [0.2, 0.3],
                "is_graspable": True,
            }
        ],
    }

    mock_response = _mock_response(200, response_json)
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        scene = await backend.vlm_scene("arm0", b"\xff\xd8\xff\xe0fake-jpeg")

    assert scene.scene == "A table with cubes"
    assert len(scene.detections) == 1
    assert scene.detections[0].handle == "red_cube_01"


@pytest.mark.asyncio
async def test_cloud_backend_health_check_ok():
    mock_response = httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", "http://test:8080"))
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
        healthy = await backend.health_check()

    assert healthy is True


@pytest.mark.asyncio
async def test_cloud_backend_health_check_failure():
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
        healthy = await backend.health_check()

    assert healthy is False


@pytest.mark.asyncio
async def test_cloud_backend_reset_loop_state():
    """reset_loop_state fires a best-effort POST /reset."""
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))
    # Should not raise even without a running event loop context
    backend.reset_loop_state()


# ---------------------------------------------------------------------------
# WarmableBackend compliance
# ---------------------------------------------------------------------------


def test_local_backend_is_warmable():
    """LocalCognitiveBackend satisfies the WarmableBackend protocol."""
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(return_value=[]), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch(
            "halo.cognitive.local_backend.make_ollama_vlm_fn",
            return_value=AsyncMock(return_value=VlmScene(scene="", detections=[])),
        ),
    ):
        backend = LocalCognitiveBackend()
    assert isinstance(backend, WarmableBackend)
    assert backend.readiness == BackendReadiness.READY
    assert backend.caught_up_cursor == -1


def test_cloud_backend_is_warmable():
    """CloudCognitiveBackend satisfies the WarmableBackend protocol."""
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))
    assert isinstance(backend, WarmableBackend)
    assert backend.readiness == BackendReadiness.COLD
    assert backend.caught_up_cursor == -1


@pytest.mark.asyncio
async def test_local_backend_warm_up_always_ready():
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch("halo.cognitive.local_backend.make_ollama_vlm_fn", return_value=AsyncMock()),
    ):
        backend = LocalCognitiveBackend()

    ready = await backend.warm_up(state=None, journal_entries=[])
    assert ready is True
    assert backend.readiness == BackendReadiness.READY


@pytest.mark.asyncio
async def test_cloud_backend_warm_up_success():
    response_json = {"readiness": "ready", "cursor": 10}
    mock_response = _mock_response(200, response_json)
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    state = CognitiveState(
        ts_ms=100,
        epoch=1,
        cursor=10,
        active_target_handle="cube",
        held_object_handle=None,
        known_scene_handles=["cube"],
        last_scene_description="table",
        pending_operator_instruction=None,
        recent_decisions=["d1"],
        last_snapshot_id="snap-1",
        last_arm_id="arm0",
        last_skill_phase=None,
        last_skill_name=None,
        last_outcome_state=None,
    )
    entries = [
        ContextEntry(cursor=9, ts_ms=90, epoch=1, backend="local", entry_type="decision", summary="d0"),
        ContextEntry(cursor=10, ts_ms=100, epoch=1, backend="local", entry_type="decision", summary="d1"),
    ]

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        ready = await backend.warm_up(state=state, journal_entries=entries)

    assert ready is True
    assert backend.readiness == BackendReadiness.READY
    assert backend.caught_up_cursor == 10


@pytest.mark.asyncio
async def test_cloud_backend_warm_up_warming():
    """Cloud service returns warming (not ready yet)."""
    response_json = {"readiness": "warming", "cursor": 5}
    mock_response = _mock_response(200, response_json)
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        ready = await backend.warm_up(state=None, journal_entries=[])

    assert ready is False
    assert backend.readiness == BackendReadiness.WARMING
    assert backend.caught_up_cursor == 5


@pytest.mark.asyncio
async def test_cloud_backend_warm_up_failure():
    """warm_up handles HTTP errors gracefully."""
    backend = CloudCognitiveBackend(config=CloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
        ready = await backend.warm_up(state=None, journal_entries=[])

    assert ready is False
    assert backend.readiness == BackendReadiness.FAILED
