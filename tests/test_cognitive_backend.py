"""Unit tests for CognitiveBackend protocol, LocalCognitiveBackend, and RemoteCognitiveBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from halo.cognitive import CognitiveStack, make_cognitive_stack
from halo.cognitive.backend import CognitiveBackend
from halo.cognitive.config import (
    BackendType,
    CloudConfig,
    CognitiveConfig,
    LocalConfig,
    RemoteCloudConfig,
)
from halo.cognitive.local_backend import LocalCognitiveBackend
from halo.cognitive.remote_backend import RemoteCognitiveBackend
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


class _StubCloudBackend:
    def __init__(self) -> None:
        self.decide = AsyncMock(return_value=[])
        self.vlm_scene = AsyncMock(return_value=VlmScene(scene="", detections=[]))
        self.health_check = AsyncMock(return_value=False)
        self.reset_loop_state = MagicMock()
        self._last_reasoning = ""

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CognitiveConfig()
    assert cfg.active == BackendType.LOCAL
    assert cfg.local.planner_model == "gpt-oss:20b"
    assert cfg.cloud.planner_model == "gemini-3.1-flash-lite-preview"
    assert cfg.enable_failover is False


def test_backend_type_enum():
    assert BackendType.LOCAL == "local"
    assert BackendType.CLOUD == "cloud"


def test_cloud_config_defaults():
    cfg = CloudConfig()
    assert cfg.planner_model == "gemini-3.1-flash-lite-preview"
    assert cfg.vlm_model == "gemini-3.1-flash-lite-preview"
    assert cfg.audio_enabled is True
    assert cfg.input_sample_rate == 16000
    assert cfg.output_sample_rate == 24000
    assert cfg.voice_name == "Kore"
    assert cfg.session_resumption is True
    assert cfg.context_compression is True
    assert cfg.response_modalities == ("AUDIO",)
    assert cfg.enable_transcription is True


def test_remote_cloud_config_defaults():
    cfg = RemoteCloudConfig(service_url="https://example.com", use_iam_auth=True)
    assert cfg.service_url == "https://example.com"
    assert cfg.use_iam_auth is True
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
            "halo.cognitive.local_backend.make_vlm_fn",
            return_value=AsyncMock(return_value=VlmScene(scene="", detections=[])),
        ),
    ):
        backend = LocalCognitiveBackend()
    assert isinstance(backend, CognitiveBackend)


def test_remote_backend_is_cognitive_backend():
    """RemoteCognitiveBackend satisfies the CognitiveBackend protocol."""
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://localhost:8080"))
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
            "halo.cognitive.local_backend.make_vlm_fn",
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
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=mock_vlm),
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
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        backend = LocalCognitiveBackend(config=LocalConfig(base_url="http://localhost:99999"))

    healthy = await backend.health_check()
    assert healthy is False


# ---------------------------------------------------------------------------
# RemoteCognitiveBackend (HTTP client)
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, json_data: dict) -> httpx.Response:
    """Create an httpx.Response with a request attached (required for raise_for_status)."""
    return httpx.Response(status_code, json=json_data, request=httpx.Request("POST", "http://test:8080"))


@pytest.mark.asyncio
async def test_remote_backend_decide():
    """Remote backend posts to /decide and returns deserialized commands."""
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
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        snap = _idle_snap()
        cmds = await backend.decide(snap, operator_cmd="scan")

    assert len(cmds) == 1
    assert cmds[0].type == CommandType.DESCRIBE_SCENE
    assert cmds[0].command_id == "cmd-1"
    assert backend.last_reasoning == "cloud thinking"
    assert backend.backend_type == "cloud"


@pytest.mark.asyncio
async def test_remote_backend_vlm_scene():
    """Remote backend posts JPEG + metadata to /vlm/scene."""
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
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        scene = await backend.vlm_scene("arm0", b"\xff\xd8\xff\xe0fake-jpeg")

    assert scene.scene == "A table with cubes"
    assert len(scene.detections) == 1
    assert scene.detections[0].handle == "red_cube_01"


@pytest.mark.asyncio
async def test_remote_backend_health_check_ok():
    mock_response = httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", "http://test:8080"))
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
        healthy = await backend.health_check()

    assert healthy is True


@pytest.mark.asyncio
async def test_remote_backend_health_check_failure():
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://test:8080"))

    with patch.object(backend._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
        healthy = await backend.health_check()

    assert healthy is False


@pytest.mark.asyncio
async def test_remote_backend_reset_loop_state_clears_reasoning():
    """reset_loop_state clears reasoning but preserves session state."""
    backend = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://test:8080"))
    backend._last_reasoning = "some reasoning"

    old_session_id = backend._session_id
    backend.reset_loop_state()

    assert backend.last_reasoning == ""
    # Session ID must stay stable so the server can rehydrate the same session.
    assert backend._session_id == old_session_id


# ---------------------------------------------------------------------------
# CognitiveStack factory
# ---------------------------------------------------------------------------


def test_make_cognitive_stack_creates_all_components():
    """make_cognitive_stack() wires up all components correctly."""
    cloud = _StubCloudBackend()
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        stack = make_cognitive_stack(cloud_backend=cloud)

    assert isinstance(stack, CognitiveStack)
    assert isinstance(stack.local, CognitiveBackend)
    assert isinstance(stack.cloud, CognitiveBackend)
    assert stack.switchboard is not None
    assert stack.context_store is not None
    assert stack.lease_manager is not None
    assert stack.config.active == BackendType.LOCAL


def test_make_cognitive_stack_with_cloud_active():
    """Factory respects active backend in config."""
    cfg = CognitiveConfig(
        active=BackendType.CLOUD,
        enable_failover=True,
    )
    cloud = _StubCloudBackend()
    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        stack = make_cognitive_stack(config=cfg, cloud_backend=cloud)

    assert stack.switchboard.active_type == BackendType.CLOUD
    assert stack.config.enable_failover is True


def test_make_cognitive_stack_cloud_active_without_backend_raises():
    """Factory rejects active=CLOUD without a cloud_backend."""
    cfg = CognitiveConfig(active=BackendType.CLOUD)
    with pytest.raises(ValueError, match="cloud_backend"):
        with (
            patch(
                "halo.cognitive.local_backend.PlannerAgent",
                return_value=MagicMock(decide=AsyncMock(), last_reasoning="", reset_loop_state=MagicMock()),
            ),
            patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
        ):
            make_cognitive_stack(config=cfg)
