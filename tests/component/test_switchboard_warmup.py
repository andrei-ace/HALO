"""Component tests for Switchboard failback protocol (simplified — no warm-up).

Wires real Switchboard + ContextStore + LeaseManager + EventBus + CommandRouter
(via HALORuntime). Only the LLM backends are mocked via ControllableMockBackend.
"""

from __future__ import annotations

import time
import uuid

import pytest

from halo.cognitive.config import BackendType, CognitiveConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import Switchboard
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
    """Stateful mock that satisfies the CognitiveBackend protocol."""

    def __init__(self, backend_type: str = "local") -> None:
        self._backend_type = backend_type
        # Health
        self._healthy = True
        # Decide
        self._decide_fn: callable | None = None
        self._decide_raise: Exception | None = None
        self._last_reasoning: str = "mock reasoning"
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
async def test_failback_immediate_when_preferred_healthy(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """When preferred backend recovers, _check_failback switches immediately."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    assert switchboard.active_type == BackendType.LOCAL

    # Cloud recovers
    cloud_backend._healthy = True
    await switchboard._check_failback()

    assert switchboard.active_type == BackendType.CLOUD


@pytest.mark.asyncio
async def test_local_commands_valid_during_failover_rejected_after_failback(
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
async def test_failover_during_failback(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
    lease_mgr: LeaseManager,
) -> None:
    """If LOCAL fails while on it, failover back to CLOUD works cleanly."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    assert switchboard.active_type == BackendType.LOCAL

    # Now LOCAL starts failing
    local_backend._decide_raise = RuntimeError("local crashed")
    cloud_backend._healthy = True
    snap = _idle_snap()
    await switchboard.decide(snap)

    # Should have failed over to CLOUD
    assert switchboard.active_type == BackendType.CLOUD
    assert local_backend._reset_calls >= 1


@pytest.mark.asyncio
async def test_context_recorded_during_local_stint(
    switchboard: Switchboard,
    cloud_backend: ControllableMockBackend,
    local_backend: ControllableMockBackend,
) -> None:
    """Context recorded during LOCAL stint is preserved in context store."""
    await _trigger_failover_to_local(switchboard, cloud_backend)
    cs = switchboard.context_store

    # Accumulate diverse context via switchboard APIs
    snap = _idle_snap()
    await switchboard.decide(snap, operator_cmd="pick the red cube")
    await switchboard.vlm_scene("arm0", b"img")

    # Append an event entry
    cs.append(
        epoch=switchboard.lease_manager.current_epoch,
        backend="local",
        entry_type="event",
        summary="SKILL_SUCCEEDED: {'skill_name': 'pick'}",
        data={"event_type": "SKILL_SUCCEEDED"},
    )

    entries = cs.get_entries_after(-1)
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
    await switchboard._check_failback()
    assert switchboard.active_type == BackendType.CLOUD

    # Submit the stale command → rejected
    ack = await runtime.submit_command(inflight_cmd)
    assert ack.status == CommandAckStatus.REJECTED_WRONG_EPOCH
