"""Component tests for Switchboard warm-up / failback protocol.

Wires real Switchboard + ContextStore + LeaseManager + EventBus + CommandRouter
(via HALORuntime). Only the LLM backends are mocked via ControllableMockBackend.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from halo.cognitive.config import BackendReadiness, BackendType, CognitiveConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry, ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import _CATCHUP_BATCH_SIZE, Switchboard
from halo.contracts.commands import CommandEnvelope, DescribeScenePayload
from halo.contracts.enums import (
    ActStatus,
    CommandAckStatus,
    CommandType,
    PerceptionFailureCode,
    SafetyState,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventEnvelope as EventEnv
from halo.contracts.events import EventType
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    TargetInfo,
)
from halo.runtime.runtime import HALORuntime
from halo.services.target_perception_service.vlm_parser import VlmScene

# ---------------------------------------------------------------------------
# ControllableMockBackend
# ---------------------------------------------------------------------------


class ControllableMockBackend:
    """Stateful mock that satisfies both CognitiveBackend and WarmableBackend protocols."""

    def __init__(self, backend_type: str = "local") -> None:
        self._backend_type = backend_type
        # Health
        self._healthy = True
        # Decide
        self._decide_fn: callable | None = None
        self._decide_raise: Exception | None = None
        self._last_reasoning: str = "mock reasoning"
        # Warm-up
        self._readiness: str = BackendReadiness.COLD
        self._caught_up_cursor: int = -1
        self._warmup_calls: list[dict] = []
        self._warmup_raise: Exception | None = None
        self._readiness_schedule: list[str] | None = None
        self._readiness_idx = 0
        # Reset
        self._reset_calls: int = 0

    # -- CognitiveBackend protocol --

    @property
    def backend_type(self) -> str:
        return self._backend_type

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        if self._decide_raise is not None:
            raise self._decide_raise
        if self._decide_fn is not None:
            return self._decide_fn(snap, operator_cmd, epoch)
        return []

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        return VlmScene(scene="mock scene", detections=[])

    async def health_check(self) -> bool:
        return self._healthy

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    def reset_loop_state(self) -> None:
        self._reset_calls += 1
        self._readiness = BackendReadiness.COLD
        self._caught_up_cursor = -1

    # -- WarmableBackend protocol --

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        if self._warmup_raise is not None:
            raise self._warmup_raise
        self._warmup_calls.append({"state": state, "entries": list(journal_entries)})
        # Advance cursor from journal entries
        if journal_entries:
            self._caught_up_cursor = max(e.cursor for e in journal_entries)
        # Progress readiness via schedule
        if self._readiness_schedule and self._readiness_idx < len(self._readiness_schedule):
            self._readiness = self._readiness_schedule[self._readiness_idx]
            self._readiness_idx += 1
        return self._readiness == BackendReadiness.READY

    @property
    def readiness(self) -> str:
        return self._readiness

    @property
    def caught_up_cursor(self) -> int:
        return self._caught_up_cursor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_command(epoch: int, token: str, cmd_id: str | None = None) -> CommandEnvelope:
    return CommandEnvelope(
        command_id=cmd_id or uuid.uuid4().hex,
        arm_id="arm0",
        issued_at_ms=int(time.time() * 1000),
        type=CommandType.DESCRIBE_SCENE,
        payload=DescribeScenePayload(reason="test"),
        precondition_snapshot_id=None,
        epoch=epoch,
        lease_token=token,
    )


async def _trigger_failover_to_local(sb: Switchboard, cloud: ControllableMockBackend) -> None:
    """Make cloud.decide fail repeatedly to trigger failover to LOCAL."""
    cloud._decide_raise = RuntimeError("cloud down")
    snap = _idle_snap()
    await sb.decide(snap)
    # With max_retries=1, a single call exhausts retries and triggers immediate failover
    assert sb.active_type == BackendType.LOCAL
    cloud._decide_raise = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lease_mgr() -> LeaseManager:
    return LeaseManager()


@pytest.fixture
def context_store() -> ContextStore:
    return ContextStore()


@pytest.fixture
def runtime(lease_mgr: LeaseManager) -> HALORuntime:
    rt = HALORuntime(lease_manager=lease_mgr)
    rt.register_arm("arm0")
    return rt


@pytest.fixture
def local_backend() -> ControllableMockBackend:
    return ControllableMockBackend("local")


@pytest.fixture
def cloud_backend() -> ControllableMockBackend:
    return ControllableMockBackend("cloud")


@pytest.fixture
def switchboard(
    local_backend: ControllableMockBackend,
    cloud_backend: ControllableMockBackend,
    lease_mgr: LeaseManager,
    context_store: ContextStore,
    runtime: HALORuntime,
) -> Switchboard:
    config = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)
    return Switchboard(
        config=config,
        local=local_backend,
        cloud=cloud_backend,
        lease_mgr=lease_mgr,
        context_store=context_store,
        bus=runtime.bus,
        snapshot_fn=runtime.get_latest_runtime_snapshot,
        arm_id="arm0",
        max_retries=1,
        retry_delays=(0.0,),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_during_warmup_journaled_and_caught_up(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    runtime: HALORuntime,
) -> None:
    """Events published during warm-up are journaled and sent to cloud via warm_up."""
    # Failover: cloud fails → LOCAL active
    await _trigger_failover_to_local(switchboard, cloud_backend)

    # Start event journal loop
    await switchboard.start()
    await asyncio.sleep(0.01)

    # Cloud recovers, starts COLD
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._readiness_schedule = [BackendReadiness.WARMING, BackendReadiness.READY]

    # First failback check → COLD → sends full warm-up → transitions to WARMING
    await switchboard._check_failback()
    assert len(cloud_backend._warmup_calls) == 1

    # Publish events while warming
    ev1 = EventEnv(
        event_id="ev-1", type=EventType.SKILL_SUCCEEDED, ts_ms=2000, arm_id="arm0", data={"skill_name": "pick"}
    )
    ev2 = EventEnv(
        event_id="ev-2", type=EventType.TARGET_ACQUIRED, ts_ms=2001, arm_id="arm0", data={"target_handle": "cube"}
    )
    await runtime.bus.publish(ev1)
    await runtime.bus.publish(ev2)
    await asyncio.sleep(0.05)  # let journal loop process

    # Second check → WARMING → sends incremental entries
    await switchboard._check_failback()
    assert len(cloud_backend._warmup_calls) == 2
    second_entries = cloud_backend._warmup_calls[1]["entries"]
    assert second_entries  # should have the event entries
    # Cloud cursor should have advanced
    assert cloud_backend.caught_up_cursor >= 0

    await switchboard.stop()


@pytest.mark.asyncio
async def test_local_commands_valid_during_warmup_rejected_after_switch(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    runtime: HALORuntime,
    lease_mgr: LeaseManager,
) -> None:
    """Commands valid during LOCAL epoch are rejected after failback to CLOUD."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    local_epoch = lease_mgr.current_epoch
    local_token = lease_mgr.current_token

    # Command with LOCAL epoch → accepted
    cmd1 = _make_command(local_epoch, local_token)
    ack1 = await runtime.submit_command(cmd1)
    assert ack1.status == CommandAckStatus.ACCEPTED

    # Failback to CLOUD
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.READY
    cloud_backend._caught_up_cursor = switchboard.context_store.latest_cursor
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.CLOUD
    new_epoch = lease_mgr.current_epoch
    new_token = lease_mgr.current_token
    assert new_epoch > local_epoch

    # Old epoch → rejected
    cmd2 = _make_command(local_epoch, local_token)
    ack2 = await runtime.submit_command(cmd2)
    assert ack2.status == CommandAckStatus.REJECTED_WRONG_EPOCH

    # New epoch → accepted
    cmd3 = _make_command(new_epoch, new_token)
    ack3 = await runtime.submit_command(cmd3)
    assert ack3.status == CommandAckStatus.ACCEPTED


@pytest.mark.asyncio
async def test_operator_instruction_during_warmup_forwarded(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """Operator instruction given during LOCAL is forwarded in warm-up journal entries."""
    await _trigger_failover_to_local(switchboard, cloud_backend)

    # Record operator instruction in context store (as switchboard.decide would)
    # Note: decide() records "operator" then "decision" — the decision clears
    # pending_operator_instruction. So we record the operator entry directly to
    # verify it appears in journal entries sent during warm-up.
    switchboard.context_store.append(
        epoch=switchboard.lease_manager.current_epoch,
        backend=switchboard.active_type,
        entry_type="operator",
        summary="pick the green cube",
    )

    # Cloud recovers COLD
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._readiness_schedule = [BackendReadiness.WARMING]
    await switchboard._check_failback()

    assert len(cloud_backend._warmup_calls) == 1
    state = cloud_backend._warmup_calls[0]["state"]
    assert state is not None
    assert state.pending_operator_instruction == "pick the green cube"
    # Journal entries should include operator entry
    entries = cloud_backend._warmup_calls[0]["entries"]
    assert any(e.entry_type == "operator" for e in entries)


@pytest.mark.asyncio
async def test_rapid_events_bounded_catchup_batch(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """Warm-up catch-up batches are bounded to _CATCHUP_BATCH_SIZE."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    cs = switchboard.context_store

    # Append 30+ entries directly to switchboard's context store
    for i in range(30):
        cs.append(
            epoch=switchboard.lease_manager.current_epoch,
            backend="local",
            entry_type="event",
            summary=f"event-{i}",
        )

    total_entries = len(cs.get_entries_after(-1))
    assert total_entries > _CATCHUP_BATCH_SIZE  # sanity check

    # Cloud starts WARMING with cursor far behind
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.WARMING
    cloud_backend._caught_up_cursor = -1

    await switchboard._check_failback()

    assert len(cloud_backend._warmup_calls) == 1
    first_batch = cloud_backend._warmup_calls[0]["entries"]
    assert len(first_batch) == _CATCHUP_BATCH_SIZE  # bounded to 20

    # Second call sends remaining entries
    await switchboard._check_failback()
    assert len(cloud_backend._warmup_calls) == 2
    second_batch = cloud_backend._warmup_calls[1]["entries"]
    assert len(second_batch) > 0
    assert len(second_batch) <= _CATCHUP_BATCH_SIZE


@pytest.mark.asyncio
async def test_failover_during_failback(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    lease_mgr: LeaseManager,
) -> None:
    """If LOCAL fails while cloud is warming, failover back to CLOUD works cleanly."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    assert switchboard.active_type == BackendType.LOCAL

    # Cloud starts warming
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._readiness_schedule = [BackendReadiness.WARMING]
    await switchboard._check_failback()

    # Now LOCAL starts failing
    local_backend._decide_raise = RuntimeError("local crashed")
    snap = _idle_snap()
    await switchboard.decide(snap)

    # Should have failed over to CLOUD
    assert switchboard.active_type == BackendType.CLOUD
    # Both backends should have had reset_loop_state called
    assert local_backend._reset_calls >= 1
    assert cloud_backend._reset_calls >= 1


@pytest.mark.asyncio
async def test_warmup_failure_prevents_switch(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """If warm_up raises, switchboard stays on LOCAL without crash."""
    await _trigger_failover_to_local(switchboard, cloud_backend)

    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._warmup_raise = RuntimeError("warm-up connection failed")

    await switchboard._check_failback()

    assert switchboard.active_type == BackendType.LOCAL
    cloud_backend._warmup_raise = None


@pytest.mark.asyncio
async def test_multi_cycle_cold_warming_ready(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """Full warm-up lifecycle: COLD → WARMING → READY → switch over 3 cycles."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    epoch_local = switchboard.lease_manager.current_epoch

    # Add some context
    switchboard.context_store.append(epoch=epoch_local, backend="local", entry_type="decision", summary="pick red cube")

    cloud_backend._healthy = True
    cloud_backend._readiness_schedule = [BackendReadiness.WARMING, BackendReadiness.READY]

    # Cycle 1: COLD → warm_up → transitions to WARMING
    cloud_backend._readiness = BackendReadiness.COLD
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.LOCAL

    # Cycle 2: WARMING → warm_up → transitions to READY
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.LOCAL

    # Cycle 3: READY + cursor caught up → switch
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.CLOUD
    assert switchboard.lease_manager.current_epoch > epoch_local


@pytest.mark.asyncio
async def test_inflight_commands_rejected_after_switch(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    runtime: HALORuntime,
    lease_mgr: LeaseManager,
) -> None:
    """Commands created before failback are rejected after the switch."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    old_epoch = lease_mgr.current_epoch
    old_token = lease_mgr.current_token

    # Create command with LOCAL credentials but don't submit yet
    inflight_cmd = _make_command(old_epoch, old_token)

    # Failback to CLOUD
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.READY
    cloud_backend._caught_up_cursor = switchboard.context_store.latest_cursor
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.CLOUD

    # Submit the stale command → rejected
    ack = await runtime.submit_command(inflight_cmd)
    assert ack.status == CommandAckStatus.REJECTED_WRONG_EPOCH


@pytest.mark.asyncio
async def test_context_completeness_after_switch(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """All context types (operator, scene, decision, event) are present in warm-up."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    cs = switchboard.context_store
    epoch = switchboard.lease_manager.current_epoch

    # Accumulate diverse context via switchboard APIs
    snap = _idle_snap()
    await switchboard.decide(snap, operator_cmd="pick the red cube")
    await switchboard.vlm_scene("arm0", b"img")

    # Append an event entry directly to switchboard's context store
    cs.append(
        epoch=epoch,
        backend="local",
        entry_type="event",
        summary="SKILL_SUCCEEDED: {'skill_name': 'pick'}",
        data={"event_type": "SKILL_SUCCEEDED"},
    )

    # Cloud recovers through COLD → READY
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._readiness_schedule = [BackendReadiness.READY]
    await switchboard._check_failback()

    assert len(cloud_backend._warmup_calls) == 1
    state = cloud_backend._warmup_calls[0]["state"]
    entries = cloud_backend._warmup_calls[0]["entries"]

    # State should have data from all context types
    assert state.recent_decisions  # from decide()
    assert state.known_scene_handles is not None  # from vlm_scene (may be empty list)
    assert state.last_scene_description  # from vlm_scene
    assert state.recent_event_summaries  # from event

    # Journal entries should contain all types
    entry_types = {e.entry_type for e in entries}
    assert "operator" in entry_types
    assert "decision" in entry_types
    assert "scene" in entry_types
    assert "event" in entry_types


@pytest.mark.asyncio
async def test_lease_token_validation_end_to_end(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    runtime: HALORuntime,
    lease_mgr: LeaseManager,
) -> None:
    """Full epoch lifecycle: CLOUD(1) → LOCAL(2) → CLOUD(3); only latest token accepted."""
    # Start: CLOUD active (epoch 1)
    epoch1 = lease_mgr.current_epoch
    token1 = lease_mgr.current_token

    # Failover to LOCAL (epoch 2)
    await _trigger_failover_to_local(switchboard, cloud_backend)
    epoch2 = lease_mgr.current_epoch
    token2 = lease_mgr.current_token
    assert epoch2 > epoch1

    # Failback to CLOUD (epoch 3)
    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.READY
    cloud_backend._caught_up_cursor = switchboard.context_store.latest_cursor
    await switchboard._check_failback()
    epoch3 = lease_mgr.current_epoch
    token3 = lease_mgr.current_token
    assert epoch3 > epoch2

    # Only (epoch3, token3) should be accepted
    ack_old1 = await runtime.submit_command(_make_command(epoch1, token1))
    assert ack_old1.status == CommandAckStatus.REJECTED_WRONG_EPOCH

    ack_old2 = await runtime.submit_command(_make_command(epoch2, token2))
    assert ack_old2.status == CommandAckStatus.REJECTED_WRONG_EPOCH

    ack_valid = await runtime.submit_command(_make_command(epoch3, token3))
    assert ack_valid.status == CommandAckStatus.ACCEPTED

    # Wrong token with correct epoch → rejected
    ack_bad_token = await runtime.submit_command(_make_command(epoch3, "wrong-token"))
    assert ack_bad_token.status == CommandAckStatus.REJECTED_WRONG_EPOCH

    # None token → rejected
    cmd_no_token = CommandEnvelope(
        command_id=uuid.uuid4().hex,
        arm_id="arm0",
        issued_at_ms=int(time.time() * 1000),
        type=CommandType.DESCRIBE_SCENE,
        payload=DescribeScenePayload(reason="test"),
        precondition_snapshot_id=None,
        epoch=epoch3,
        lease_token=None,
    )
    ack_no_token = await runtime.submit_command(cmd_no_token)
    assert ack_no_token.status == CommandAckStatus.REJECTED_WRONG_EPOCH


@pytest.mark.asyncio
async def test_second_failback_requires_full_warmup(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """After CLOUD→LOCAL→CLOUD→LOCAL→CLOUD, the second failback must do a full
    COLD warm-up — not skip to READY with stale session history.

    Bug: reset_loop_state() didn't reset readiness/cursor, so after the first
    failback cycle cloud.readiness stayed READY, causing the second failback to
    skip the full warm-up and reuse old session state.
    """
    # --- First cycle: CLOUD → LOCAL → CLOUD ---
    await _trigger_failover_to_local(switchboard, cloud_backend)

    cloud_backend._healthy = True
    cloud_backend._readiness = BackendReadiness.COLD
    cloud_backend._readiness_schedule = [BackendReadiness.READY]
    await switchboard._check_failback()  # COLD → warm_up → READY
    await switchboard._check_failback()  # READY + caught up → switch to CLOUD
    assert switchboard.active_type == BackendType.CLOUD
    first_cycle_warmup_count = len(cloud_backend._warmup_calls)
    assert first_cycle_warmup_count >= 1

    # At this point cloud._readiness is READY and _caught_up_cursor is set.
    # switch_to() called reset_loop_state() — it MUST reset readiness to COLD.

    # --- Second cycle: CLOUD fails again → LOCAL → CLOUD ---
    cloud_backend._decide_raise = RuntimeError("cloud down again")
    await switchboard.decide(_idle_snap())
    assert switchboard.active_type == BackendType.LOCAL
    cloud_backend._decide_raise = None

    # Add new context during second LOCAL stint
    switchboard.context_store.append(
        epoch=switchboard.lease_manager.current_epoch,
        backend="local",
        entry_type="decision",
        summary="new decision during second local stint",
    )

    # Cloud recovers — readiness MUST be COLD (not stale READY)
    cloud_backend._healthy = True
    assert cloud_backend.readiness == BackendReadiness.COLD, (
        f"Expected COLD after reset_loop_state(), got {cloud_backend.readiness}"
    )

    # Failback should go through full COLD warm-up (with CognitiveState)
    cloud_backend._readiness_schedule = [BackendReadiness.READY]
    await switchboard._check_failback()  # COLD → full warm_up

    # Find the warm_up call from the second cycle — it must have state (full warm-up)
    second_cycle_calls = cloud_backend._warmup_calls[first_cycle_warmup_count:]
    assert len(second_cycle_calls) >= 1
    assert second_cycle_calls[0]["state"] is not None, (
        "Second failback must do full COLD warm-up with CognitiveState, not incremental"
    )
