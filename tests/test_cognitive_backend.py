"""Unit tests for CognitiveBackend protocol, LocalCognitiveBackend, CloudCognitiveBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from halo.cognitive.backend import CognitiveBackend
from halo.cognitive.cloud_backend import CloudCognitiveBackend
from halo.cognitive.config import BackendType, CloudConfig, CognitiveConfig, LocalConfig
from halo.cognitive.local_backend import LocalCognitiveBackend
from halo.contracts.enums import (
    ActStatus,
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
    with (
        patch(
            "halo.cognitive.cloud_backend.PlannerAgent",
            return_value=MagicMock(decide=AsyncMock(return_value=[]), last_reasoning="", reset_loop_state=MagicMock()),
        ),
        patch(
            "halo.cognitive.cloud_backend.make_gemini_vlm_fn",
            return_value=AsyncMock(return_value=VlmScene(scene="", detections=[])),
        ),
    ):
        backend = CloudCognitiveBackend(config=CloudConfig(api_key="test"))
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
    mock_agent.decide.assert_awaited_once_with(snap, operator_cmd="pick cube")
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
# CloudCognitiveBackend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_backend_decide():
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = "cloud thinking"
    mock_agent.reset_loop_state = MagicMock()

    with (
        patch("halo.cognitive.cloud_backend.PlannerAgent", return_value=mock_agent),
        patch(
            "halo.cognitive.cloud_backend.make_gemini_vlm_fn",
            return_value=AsyncMock(),
        ),
    ):
        backend = CloudCognitiveBackend(config=CloudConfig(api_key="test"))

    snap = _idle_snap()
    cmds = await backend.decide(snap)
    assert cmds == []
    assert backend.last_reasoning == "cloud thinking"
    assert backend.backend_type == "cloud"


@pytest.mark.asyncio
async def test_cloud_backend_reset_loop_state():
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()

    with (
        patch("halo.cognitive.cloud_backend.PlannerAgent", return_value=mock_agent),
        patch(
            "halo.cognitive.cloud_backend.make_gemini_vlm_fn",
            return_value=AsyncMock(),
        ),
    ):
        backend = CloudCognitiveBackend(config=CloudConfig(api_key="test"))

    backend.reset_loop_state()
    mock_agent.reset_loop_state.assert_called_once()
