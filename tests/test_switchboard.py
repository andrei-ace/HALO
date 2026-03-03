"""Unit tests for Switchboard — failover, failback, context handoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from halo.cognitive.config import BackendType, CognitiveConfig
from halo.cognitive.switchboard import CONSECUTIVE_FAILURES_BEFORE_SWITCH, Switchboard
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


def _make_mock_backend(backend_type: str = "local") -> MagicMock:
    backend = MagicMock()
    backend.backend_type = backend_type
    backend.decide = AsyncMock(return_value=[])
    backend.vlm_scene = AsyncMock(return_value=VlmScene(scene="table", detections=[]))
    backend.health_check = AsyncMock(return_value=True)
    type(backend).last_reasoning = PropertyMock(return_value="mock reasoning")
    backend.reset_loop_state = MagicMock()
    return backend


def _make_switchboard(
    enable_failover: bool = False,
    active: BackendType = BackendType.LOCAL,
) -> tuple[Switchboard, MagicMock, MagicMock]:
    local = _make_mock_backend("local")
    cloud = _make_mock_backend("cloud")
    config = CognitiveConfig(active=active, enable_failover=enable_failover)
    sb = Switchboard(config=config, local=local, cloud=cloud)
    return sb, local, cloud


# ---------------------------------------------------------------------------
# Basic delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_delegates_to_active():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    snap = _idle_snap()
    await sb.decide(snap, operator_cmd="pick cube")
    local.decide.assert_awaited_once_with(snap, operator_cmd="pick cube")
    cloud.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_vlm_scene_delegates_to_active():
    sb, local, cloud = _make_switchboard(active=BackendType.CLOUD)
    await sb.vlm_scene("arm0", b"img")
    cloud.vlm_scene.assert_awaited_once()
    local.vlm_scene.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_backend_property():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    assert sb.active_backend is local
    assert sb.active_type == BackendType.LOCAL


# ---------------------------------------------------------------------------
# Manual switching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_to():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    await sb.switch_to(BackendType.CLOUD, reason="manual switch")
    assert sb.active_type == BackendType.CLOUD
    assert sb.active_backend is cloud
    cloud.reset_loop_state.assert_called_once()


@pytest.mark.asyncio
async def test_switch_to_same_noop():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    initial_epoch = sb.lease_manager.current_epoch
    await sb.switch_to(BackendType.LOCAL, reason="no-op")
    # Epoch should not change
    assert sb.lease_manager.current_epoch == initial_epoch


@pytest.mark.asyncio
async def test_switch_rotates_lease():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    epoch_before = sb.lease_manager.current_epoch
    await sb.switch_to(BackendType.CLOUD, reason="test")
    epoch_after = sb.lease_manager.current_epoch
    assert epoch_after > epoch_before
    assert sb.lease_manager.current_lease.holder == "cloud"


# ---------------------------------------------------------------------------
# Context handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_records_in_context_store():
    sb, local, cloud = _make_switchboard()
    snap = _idle_snap()
    await sb.decide(snap, operator_cmd="pick the red cube")
    assert len(sb.context_store) >= 1


@pytest.mark.asyncio
async def test_vlm_scene_records_in_context_store():
    sb, local, cloud = _make_switchboard()
    await sb.vlm_scene("arm0", b"img")
    # "table" scene should be recorded
    entries = sb.context_store.get_entries_after(-1)
    assert any(e.entry_type == "scene" for e in entries)


@pytest.mark.asyncio
async def test_handoff_context_includes_decisions():
    sb, local, cloud = _make_switchboard()
    snap = _idle_snap()
    await sb.decide(snap)
    ctx = sb.get_handoff_context()
    assert "[Context handoff" in ctx


# ---------------------------------------------------------------------------
# Automatic failover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failover_after_consecutive_failures():
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    local.decide = AsyncMock(side_effect=RuntimeError("Ollama down"))
    snap = _idle_snap()

    for _ in range(CONSECUTIVE_FAILURES_BEFORE_SWITCH):
        await sb.decide(snap)

    # Should have switched to cloud
    assert sb.active_type == BackendType.CLOUD


@pytest.mark.asyncio
async def test_no_failover_when_disabled():
    sb, local, cloud = _make_switchboard(enable_failover=False, active=BackendType.LOCAL)
    local.decide = AsyncMock(side_effect=RuntimeError("Ollama down"))
    snap = _idle_snap()

    for _ in range(CONSECUTIVE_FAILURES_BEFORE_SWITCH + 2):
        await sb.decide(snap)

    # Should NOT have switched (failover disabled)
    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_success_resets_failure_counter():
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    snap = _idle_snap()

    # Fail twice
    local.decide = AsyncMock(side_effect=RuntimeError("fail"))
    await sb.decide(snap)
    await sb.decide(snap)

    # Then succeed
    local.decide = AsyncMock(return_value=[])
    await sb.decide(snap)

    # Fail twice more — should not trigger switch (counter was reset)
    local.decide = AsyncMock(side_effect=RuntimeError("fail"))
    await sb.decide(snap)
    await sb.decide(snap)
    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_vlm_failure_also_counts():
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    local.vlm_scene = AsyncMock(side_effect=RuntimeError("VLM down"))

    for _ in range(CONSECUTIVE_FAILURES_BEFORE_SWITCH):
        await sb.vlm_scene("arm0", b"img")

    assert sb.active_type == BackendType.CLOUD


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_publishes_event():
    bus = MagicMock()
    bus.publish = AsyncMock()
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    sb._bus = bus

    await sb.switch_to(BackendType.CLOUD, reason="test event")
    bus.publish.assert_awaited_once()
    event = bus.publish.call_args[0][0]
    assert event.type.value == "BACKEND_SWITCHED"
    assert event.data["from"] == BackendType.LOCAL
    assert event.data["to"] == BackendType.CLOUD


# ---------------------------------------------------------------------------
# Reset loop state
# ---------------------------------------------------------------------------


def test_reset_loop_state_delegates():
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    sb.reset_loop_state()
    local.reset_loop_state.assert_called_once()
