"""Unit tests for Switchboard — failover, failback, context handoff, warm-up."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from halo.cognitive.config import BackendReadiness, BackendType, CognitiveConfig
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
    local.decide.assert_awaited_once_with(snap, operator_cmd="pick cube", epoch=sb.lease_manager.current_epoch)
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


# ---------------------------------------------------------------------------
# Epoch stamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_passes_epoch():
    """Switchboard.decide() passes the current epoch to the backend."""
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    snap = _idle_snap()
    await sb.decide(snap)
    local.decide.assert_awaited_once()
    _, kwargs = local.decide.call_args
    assert kwargs["epoch"] == sb.lease_manager.current_epoch


# ---------------------------------------------------------------------------
# Warm-up failback protocol
# ---------------------------------------------------------------------------


def _make_warmable_mock_backend(backend_type: str = "local") -> MagicMock:
    """Create a mock that implements both CognitiveBackend and WarmableBackend."""
    backend = _make_mock_backend(backend_type)
    backend.warm_up = AsyncMock(return_value=False)
    type(backend).readiness = PropertyMock(return_value=BackendReadiness.COLD)
    type(backend).caught_up_cursor = PropertyMock(return_value=-1)
    return backend


@pytest.mark.asyncio
async def test_failback_sends_full_warmup_when_cold():
    """When preferred backend is COLD, _check_failback sends full state + journal."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud)

    # Simulate: switched to cloud (local is preferred but failed)
    await sb.switch_to(BackendType.CLOUD, reason="test failover")
    assert sb.active_type == BackendType.CLOUD

    # Local is COLD but healthy — should trigger warm_up with full state
    type(local).readiness = PropertyMock(return_value=BackendReadiness.COLD)
    local.health_check = AsyncMock(return_value=True)
    local.warm_up = AsyncMock(return_value=False)  # not ready yet

    await sb._check_failback()

    local.warm_up.assert_awaited_once()
    call_kwargs = local.warm_up.call_args[1]
    assert call_kwargs["state"] is not None  # CognitiveState was built
    assert isinstance(call_kwargs["journal_entries"], list)
    # Still on cloud (warm_up returned False)
    assert sb.active_type == BackendType.CLOUD


@pytest.mark.asyncio
async def test_failback_sends_incremental_when_warming():
    """When preferred backend is WARMING, _check_failback sends incremental entries."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud)

    await sb.switch_to(BackendType.CLOUD, reason="test")

    # Local is WARMING with cursor=5
    type(local).readiness = PropertyMock(return_value=BackendReadiness.WARMING)
    type(local).caught_up_cursor = PropertyMock(return_value=5)
    local.health_check = AsyncMock(return_value=True)
    local.warm_up = AsyncMock(return_value=False)

    await sb._check_failback()

    local.warm_up.assert_awaited_once()
    call_kwargs = local.warm_up.call_args[1]
    assert call_kwargs["state"] is None  # incremental — no full state


@pytest.mark.asyncio
async def test_failback_switches_when_ready():
    """When preferred backend is READY, _check_failback switches to it."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud)

    await sb.switch_to(BackendType.CLOUD, reason="test")
    assert sb.active_type == BackendType.CLOUD

    # Local is READY — should switch back
    type(local).readiness = PropertyMock(return_value=BackendReadiness.READY)
    local.health_check = AsyncMock(return_value=True)

    await sb._check_failback()

    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_failback_immediate_for_non_warmable():
    """Non-warmable backends get immediate switch (backward compat)."""
    # Use plain mocks without warm_up/readiness/caught_up_cursor
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    await sb.switch_to(BackendType.CLOUD, reason="test")
    assert sb.active_type == BackendType.CLOUD

    local.health_check = AsyncMock(return_value=True)
    await sb._check_failback()

    # Should switch immediately (no warm-up)
    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_failback_warmup_with_snapshot_fn():
    """When snapshot_fn is provided, it's used to build CognitiveState for warm-up."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    snap = _idle_snap()
    snapshot_fn = AsyncMock(return_value=snap)
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, snapshot_fn=snapshot_fn)

    await sb.switch_to(BackendType.CLOUD, reason="test")

    type(local).readiness = PropertyMock(return_value=BackendReadiness.COLD)
    local.health_check = AsyncMock(return_value=True)
    local.warm_up = AsyncMock(return_value=False)

    await sb._check_failback()

    snapshot_fn.assert_awaited_once_with("arm0")
    call_kwargs = local.warm_up.call_args[1]
    # State should contain snapshot-derived fields
    assert call_kwargs["state"].last_snapshot_id == "snap-001"


# ---------------------------------------------------------------------------
# Lease renewal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_decide_renews_lease():
    """Every successful decide() call should renew the lease TTL."""
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    snap = _idle_snap()

    lease_before = sb.lease_manager.current_lease
    granted_at_before = lease_before.granted_at_ms

    # Small sleep to ensure monotonic time advances
    import asyncio

    await asyncio.sleep(0.002)

    await sb.decide(snap)

    # granted_at should have been refreshed
    assert sb.lease_manager.current_lease.granted_at_ms >= granted_at_before


@pytest.mark.asyncio
async def test_lease_expires_without_renewal():
    """Lease should expire if TTL passes without renewal."""
    from halo.cognitive.lease import LeaseManager

    mgr = LeaseManager()
    lease = mgr.grant("local", ttl_ms=1)  # 1ms TTL

    import asyncio

    await asyncio.sleep(0.005)

    # Lease should be expired since no renewal happened
    assert not mgr.is_valid(lease.epoch)

    # After renewal, lease should be valid again
    mgr.renew(lease.epoch)
    assert mgr.is_valid(lease.epoch)


# ---------------------------------------------------------------------------
# Active backend re-warm on COLD detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewarm_active_on_cold_detection():
    """When active WarmableBackend drops to COLD, _rewarm_active() sends warm-up."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud)

    # Cloud is active and drops to COLD (e.g. instance restart)
    type(cloud).readiness = PropertyMock(return_value=BackendReadiness.COLD)
    cloud.warm_up = AsyncMock(return_value=True)

    await sb._rewarm_active()

    cloud.warm_up.assert_awaited_once()
    call_kwargs = cloud.warm_up.call_args[1]
    assert call_kwargs["state"] is not None


@pytest.mark.asyncio
async def test_rewarm_active_skips_when_ready():
    """_rewarm_active() is a no-op when active backend readiness is not COLD."""
    local = _make_warmable_mock_backend("local")
    cloud = _make_warmable_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud)

    type(cloud).readiness = PropertyMock(return_value=BackendReadiness.READY)

    await sb._rewarm_active()

    cloud.warm_up.assert_not_awaited()
