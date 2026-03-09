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
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))
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
    # Only the OLD backend gets reset_loop_state
    local.reset_loop_state.assert_called_once()
    cloud.reset_loop_state.assert_not_called()


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
# Handoff injection on switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_to_local_resets_stale_session_then_injects_handoff():
    """switch_to(LOCAL) resets a stale session (no synced history) then injects handoff."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.reset_session = AsyncMock()
    mock_agent.inject_handoff_context = AsyncMock()
    mock_agent._pending_handoff = None
    mock_agent.msg_history = MessageHistory()  # empty — no compaction sync

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    # Add context so handoff isn't empty
    snap = _idle_snap()
    await sb.decide(snap, operator_cmd="pick the red cube")

    # Switch to LOCAL
    await sb.switch_to(BackendType.LOCAL, reason="test handoff")

    # No synced history → reset_session must be called before inject
    mock_agent.reset_session.assert_awaited_once()
    mock_agent.inject_handoff_context.assert_awaited_once()

    handoff = mock_agent.inject_handoff_context.call_args[0][0]
    assert "[Context handoff" in handoff
    assert "Recent decisions:" in handoff


@pytest.mark.asyncio
async def test_switch_to_local_preserves_compaction_synced_history():
    """switch_to(LOCAL) skips reset when local agent has compaction-synced history."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.reset_session = AsyncMock()
    mock_agent.inject_handoff_context = AsyncMock()
    mock_agent.inject_compaction_state = AsyncMock()
    mock_agent._pending_handoff = None
    # Simulate compaction-synced history (non-empty)
    synced_history = MessageHistory()
    synced_history.append("model", "Session summary: picked red cube")
    synced_history.append("user", "snapshot...")
    mock_agent.msg_history = synced_history

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    snap = _idle_snap()
    await sb.decide(snap, operator_cmd="pick the red cube")

    # Switch to LOCAL — should NOT reset the synced session
    await sb.switch_to(BackendType.LOCAL, reason="test compaction preserve")

    mock_agent.reset_session.assert_not_awaited()  # reset happens inside inject_compaction_state
    mock_agent.inject_compaction_state.assert_awaited_once()
    mock_agent.inject_handoff_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_switch_to_cloud_preserves_sync_state_and_forwards_handoff():
    """switch_to(CLOUD) preserves sync state for need_history protocol and forwards handoff."""
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    remote = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    remote._last_reasoning = ""
    remote._last_msg_id = "old-msg-id"
    remote._msg_history = [{"msg_id": "old-msg-id", "role": "user", "text": "old"}]
    remote._on_compaction = None
    remote._last_token_usage = {}
    remote._session_id = "test"
    remote._config = MagicMock()
    remote._pending_handoff = None

    local = _make_mock_backend("local")
    local.decide = AsyncMock(return_value=[])
    local.last_reasoning = "I will pick the cube"
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=remote, max_retries=1, retry_delays=(0.0,))

    # Add context so handoff is non-empty
    snap = _idle_snap()
    await sb.decide(snap, operator_cmd="pick the red cube")

    # Switch to CLOUD
    await sb.switch_to(BackendType.CLOUD, reason="test reset")

    # Sync state preserved so need_history protocol works on server restart
    assert remote._last_msg_id == "old-msg-id"
    assert remote._msg_history == [{"msg_id": "old-msg-id", "role": "user", "text": "old"}]

    # Handoff context must be forwarded
    assert remote._pending_handoff is not None
    assert "[Context handoff" in remote._pending_handoff


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

    # Succeed — should stay on local
    local.decide = AsyncMock(return_value=[])
    await sb.decide(snap)
    assert sb.active_type == BackendType.LOCAL
    assert sb._consecutive_failures == 0

    # One failure triggers immediate failover (retries exhausted → switch)
    local.decide = AsyncMock(side_effect=RuntimeError("fail"))
    await sb.decide(snap)
    assert sb.active_type == BackendType.CLOUD


@pytest.mark.asyncio
async def test_vlm_failure_also_counts():
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    local.vlm_scene = AsyncMock(side_effect=RuntimeError("VLM down"))

    # Single call with retries exhausted triggers immediate failover
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


@pytest.mark.asyncio
async def test_cloud_switch_ts_set_before_event_published():
    """cloud_switch_ts_ms must be set before BACKEND_SWITCHED is published,
    so consumers reading it directly always see a fresh gate."""
    ts_at_publish: list[int] = []

    async def capture_ts(event):
        # At the moment publish() is called, cloud_switch_ts_ms must already be set
        ts_at_publish.append(sb.cloud_switch_ts_ms)

    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=capture_ts)
    sb, local, cloud = _make_switchboard(active=BackendType.LOCAL)
    sb._bus = bus

    assert sb.cloud_switch_ts_ms == 0
    await sb.switch_to(BackendType.CLOUD, reason="test")

    assert sb.cloud_switch_ts_ms > 0
    assert len(ts_at_publish) == 1
    assert ts_at_publish[0] == sb.cloud_switch_ts_ms  # was set before publish


@pytest.mark.asyncio
async def test_cloud_switch_ts_not_set_on_local_switch():
    """Switching to LOCAL should not update cloud_switch_ts_ms."""
    sb, local, cloud = _make_switchboard(active=BackendType.CLOUD)
    sb.cloud_switch_ts_ms = 0

    await sb.switch_to(BackendType.LOCAL, reason="test")
    assert sb.cloud_switch_ts_ms == 0


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
# Failback — immediate switch when preferred backend recovers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failback_switches_when_healthy():
    """When preferred backend is healthy, _check_failback switches to it."""
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    await sb.switch_to(BackendType.CLOUD, reason="test")
    assert sb.active_type == BackendType.CLOUD

    # Local is healthy — should switch back
    local.health_check = AsyncMock(return_value=True)
    await sb._check_failback()
    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_failback_no_switch_when_unhealthy():
    """When preferred backend is unhealthy, _check_failback does not switch."""
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    await sb.switch_to(BackendType.CLOUD, reason="test")
    assert sb.active_type == BackendType.CLOUD

    local.health_check = AsyncMock(return_value=False)
    await sb._check_failback()
    assert sb.active_type == BackendType.CLOUD


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
# Failure reason tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_reason_included_in_switch_event():
    """BACKEND_SWITCHED event includes the descriptive failure reason."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    sb._bus = bus

    local.decide = AsyncMock(side_effect=RuntimeError("Ollama timeout after 30s"))
    snap = _idle_snap()

    await sb.decide(snap)

    # Should have switched — find the BACKEND_SWITCHED event
    switch_calls = [c for c in bus.publish.call_args_list if c[0][0].type.value == "BACKEND_SWITCHED"]
    assert len(switch_calls) == 1
    event = switch_calls[0][0][0]
    assert "Ollama timeout after 30s" in event.data["reason"]


@pytest.mark.asyncio
async def test_empty_response_triggers_failure():
    """Empty decide() response (no reasoning, no commands) counts as failure."""
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    local.decide = AsyncMock(return_value=[])
    type(local).last_reasoning = PropertyMock(return_value="")
    snap = _idle_snap()

    result = await sb.decide(snap)

    assert result == []
    assert sb._consecutive_failures == 1
    assert "empty response" in sb._last_failure_reason


@pytest.mark.asyncio
async def test_three_empty_responses_trigger_failover():
    """Three consecutive empty responses trigger backend switch."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    sb._bus = bus

    local.decide = AsyncMock(return_value=[])
    type(local).last_reasoning = PropertyMock(return_value="")
    snap = _idle_snap()

    for _ in range(CONSECUTIVE_FAILURES_BEFORE_SWITCH):
        await sb.decide(snap)

    assert sb.active_type == BackendType.CLOUD
    switch_calls = [c for c in bus.publish.call_args_list if c[0][0].type.value == "BACKEND_SWITCHED"]
    assert len(switch_calls) == 1
    assert "empty response" in switch_calls[0][0][0].data["reason"]


@pytest.mark.asyncio
async def test_non_empty_reasoning_resets_failure_counter():
    """A decide() with reasoning but no commands is a success (not empty)."""
    sb, local, cloud = _make_switchboard(enable_failover=True, active=BackendType.LOCAL)
    local.decide = AsyncMock(return_value=[])
    type(local).last_reasoning = PropertyMock(return_value="Waiting for tracking to stabilize")
    snap = _idle_snap()

    await sb.decide(snap)

    assert sb._consecutive_failures == 0


# ---------------------------------------------------------------------------
# BACKEND_SWITCHED filtering from planner snapshots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_switched_filtered_from_snapshot():
    """BACKEND_SWITCHED events should not appear in planner snapshot recent_events."""
    from halo.contracts.events import EventEnvelope, EventType
    from halo.runtime.runtime import HALORuntime

    rt = HALORuntime()
    rt.register_arm("arm0")

    # Publish a BACKEND_SWITCHED event and a normal event
    switch_event = EventEnvelope(
        event_id="switch-1",
        type=EventType.BACKEND_SWITCHED,
        ts_ms=1000,
        arm_id="arm0",
        data={"from": "local", "to": "cloud", "reason": "test"},
    )
    normal_event = EventEnvelope(
        event_id="skill-ok",
        type=EventType.SKILL_SUCCEEDED,
        ts_ms=1001,
        arm_id="arm0",
        data={"skill_name": "pick"},
    )
    await rt.bus.publish(switch_event)
    await rt.bus.publish(normal_event)

    snap = await rt.get_latest_runtime_snapshot("arm0")

    event_types = [e.type for e in snap.recent_events]
    assert EventType.BACKEND_SWITCHED not in event_types
    assert EventType.SKILL_SUCCEEDED in event_types


# ---------------------------------------------------------------------------
# History mirroring (cloud → local)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_history_after_cloud_decide():
    """Local msg_history is updated after a successful cloud decide."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    # Set up local with real MessageHistory
    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.msg_history = MessageHistory()

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    # Set up remote with msg_history containing records
    remote = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    remote._last_reasoning = "I will pick"
    remote._on_compaction = None
    remote._last_token_usage = {}
    remote._session_id = "test"
    remote._config = MagicMock()
    remote._pending_handoff = None
    remote._last_msg_id = "msg-2"
    remote._msg_history = [
        {"msg_id": "msg-1", "role": "user", "text": "snap1", "ts_ms": 1, "is_summary": False},
        {"msg_id": "msg-2", "role": "model", "text": "reply1", "ts_ms": 2, "is_summary": False},
    ]

    # Mock HTTP response
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "commands": [],
        "reasoning": "I will pick the cube",
        "msg_history": remote._msg_history,
    }
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    remote._client = mock_client

    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=remote, max_retries=1, retry_delays=(0.0,))

    snap = _idle_snap()
    await sb.decide(snap)

    # Local MessageHistory should now have the mirrored records
    records = local.agent.msg_history.get_all()
    assert len(records) == 2
    assert records[0].role == "user"
    assert records[0].text == "snap1"
    assert records[1].role == "model"
    assert records[1].text == "reply1"


@pytest.mark.asyncio
async def test_mirror_history_noop_when_local_active():
    """No mirror occurs when local is the active backend."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = "ok"
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.msg_history = MessageHistory()

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    snap = _idle_snap()
    await sb.decide(snap)

    # Local history should stay empty — no mirroring when local is active
    assert local.agent.msg_history.count() == 0


@pytest.mark.asyncio
async def test_mirror_error_does_not_break_decide():
    """Mirror exception is swallowed — decide() still returns commands."""

    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    # Set up remote that will return commands
    remote = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    remote._last_reasoning = "picking"
    remote._on_compaction = None
    remote._last_token_usage = {}
    remote._session_id = "test"
    remote._config = MagicMock()
    remote._pending_handoff = None
    remote._last_msg_id = None
    remote._msg_history = None

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"commands": [], "reasoning": "ok"}
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    remote._client = mock_client

    # Local backend whose msg_history.replace_all raises
    local = _make_mock_backend("local")

    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=remote, max_retries=1, retry_delays=(0.0,))

    # Force the mirror to fail — local is a MagicMock, no real agent
    snap = _idle_snap()
    # Should NOT raise despite mirroring failure
    result = await sb.decide(snap)
    assert result == []  # empty commands but no exception


# ---------------------------------------------------------------------------
# Local decide preemption on cloud failback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_decide_preempted_by_cloud_failback():
    """When cloud recovers mid-decide, the slow local inference is cancelled
    and the call is retried on cloud.  Local message history is rolled back."""
    import asyncio
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    # Local backend that takes a long time (simulated via Event)
    local_started = asyncio.Event()

    async def slow_local_decide(snap, operator_cmd=None, epoch=None):
        local_started.set()
        await asyncio.sleep(10)  # "slow" local inference
        return []

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(side_effect=slow_local_decide)
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.reset_session = AsyncMock()
    mock_agent.inject_handoff_context = AsyncMock()
    mock_agent._pending_handoff = None
    mock_agent.msg_history = MessageHistory()
    # Simulate that decide() appends to history before blocking
    mock_agent.msg_history.append("user", "pre-existing snapshot")

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    cloud.decide = AsyncMock(return_value=[])
    type(cloud).last_reasoning = PropertyMock(return_value="cloud reasoning")

    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    snap = _idle_snap()
    history_count_before = local.agent.msg_history.count()

    async def trigger_failback():
        """Simulate cloud recovery while local decide is in-flight."""
        await local_started.wait()
        await asyncio.sleep(0.01)  # let decide race start
        await sb.switch_to(BackendType.CLOUD, reason="cloud recovered")

    # Run decide and failback concurrently
    decide_task = asyncio.create_task(sb.decide(snap))
    failback_task = asyncio.create_task(trigger_failback())

    await decide_task
    await failback_task

    # Should have ended up on cloud
    assert sb.active_type == BackendType.CLOUD
    # Cloud decide should have been called (replay after preemption)
    cloud.decide.assert_awaited()
    # Local message history should be rolled back to pre-decide state
    assert local.agent.msg_history.count() == history_count_before


@pytest.mark.asyncio
async def test_cloud_decide_not_preemptable():
    """Cloud decide() is not raced against preemption — only local is."""
    sb, local, cloud = _make_switchboard(active=BackendType.CLOUD)
    snap = _idle_snap()

    # If preempt was erroneously checked for cloud, this would fail
    sb._decide_preempt.set()  # should be ignored for cloud
    result = await sb.decide(snap)
    assert result == []
    cloud.decide.assert_awaited_once()


@pytest.mark.asyncio
async def test_preempt_already_set_before_decide():
    """If switch_to(CLOUD) fires before decide() enters, local is skipped immediately."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = ""
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.reset_session = AsyncMock()
    mock_agent.inject_handoff_context = AsyncMock()
    mock_agent._pending_handoff = None
    mock_agent.msg_history = MessageHistory()

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    cloud.decide = AsyncMock(return_value=[])
    type(cloud).last_reasoning = PropertyMock(return_value="cloud ok")

    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    # Simulate: switch_to(CLOUD) already happened before decide() is called
    await sb.switch_to(BackendType.CLOUD, reason="cloud recovered")

    # Now call decide — it should go to cloud, not local
    snap = _idle_snap()
    await sb.decide(snap)

    # Local decide should NOT have been called
    mock_agent.decide.assert_not_awaited()
    # Cloud decide should have been called
    cloud.decide.assert_awaited()


@pytest.mark.asyncio
async def test_preempt_cleared_after_round_trip():
    """LOCAL→CLOUD→LOCAL round-trip must not leave a stale preempt latch."""
    from unittest.mock import patch

    from halo.cognitive.compactor import MessageHistory
    from halo.cognitive.local_backend import LocalCognitiveBackend

    mock_agent = MagicMock()
    mock_agent.decide = AsyncMock(return_value=[])
    mock_agent.last_reasoning = "local ok"
    mock_agent.reset_loop_state = MagicMock()
    mock_agent.reset_session = AsyncMock()
    mock_agent.inject_handoff_context = AsyncMock()
    mock_agent.inject_compaction_state = AsyncMock()
    mock_agent._pending_handoff = None
    mock_agent.msg_history = MessageHistory()

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        local = LocalCognitiveBackend()

    cloud = _make_mock_backend("cloud")
    config = CognitiveConfig(active=BackendType.LOCAL, enable_failover=True)
    sb = Switchboard(config=config, local=local, cloud=cloud, max_retries=1, retry_delays=(0.0,))

    # Round-trip: LOCAL → CLOUD → LOCAL (no decide in-flight)
    await sb.switch_to(BackendType.CLOUD, reason="cloud ok")
    assert not sb._decide_preempt.is_set(), "latch should be clear after switch completes"

    await sb.switch_to(BackendType.LOCAL, reason="prefer local")
    assert sb.active_type == BackendType.LOCAL
    assert not sb._decide_preempt.is_set(), "latch should still be clear"

    # First local decide after round-trip should run normally on LOCAL
    snap = _idle_snap()
    await sb.decide(snap)
    mock_agent.decide.assert_awaited_once()
