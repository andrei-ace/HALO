"""Tests for RuntimeStateStore."""

import asyncio

import pytest

from halo.contracts.commands import CommandAck
from halo.contracts.enums import (
    ActStatus,
    CommandAckStatus,
    PerceptionFailureCode,
    PhaseId,
    SafetyState,
    SkillName,
    TrackingStatus,
)
from halo.contracts.snapshots import (
    ActInfo,
    PerceptionInfo,
    SafetyInfo,
    SkillInfo,
    TargetInfo,
)
from halo.runtime.state_store import RuntimeStateStore

ARM = "arm0"
ARM2 = "arm1"


@pytest.fixture
def store() -> RuntimeStateStore:
    s = RuntimeStateStore()
    s.register_arm(ARM)
    return s


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_arm_is_idempotent():
    s = RuntimeStateStore()
    s.register_arm(ARM)
    s.register_arm(ARM)  # second call must not raise


async def test_unregistered_arm_raises(store: RuntimeStateStore):
    with pytest.raises(KeyError, match="not registered"):
        await store.update_skill("unknown_arm", None)


# ---------------------------------------------------------------------------
# Per-field updates
# ---------------------------------------------------------------------------


async def test_update_skill(store: RuntimeStateStore):
    skill = SkillInfo(name=SkillName.PICK, skill_run_id="run-1", phase=PhaseId.SELECT_GRASP)
    await store.update_skill(ARM, skill)
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.skill == skill


async def test_update_target(store: RuntimeStateStore):
    target = TargetInfo(
        handle="cube-1",
        hint_valid=True,
        confidence=0.85,
        obs_age_ms=15,
        time_skew_ms=2,
        delta_xyz_ee=(0.0, 0.0, 0.1),
        distance_m=0.1,
    )
    await store.update_target(ARM, target)
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.target == target


async def test_update_held_object_handle(store: RuntimeStateStore):
    await store.update_held_object_handle(ARM, "cube-1")
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.held_object_handle == "cube-1"


async def test_held_object_handle_defaults_to_none(store: RuntimeStateStore):
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.held_object_handle is None


async def test_update_perception(store: RuntimeStateStore):
    perc = PerceptionInfo(
        tracking_status=TrackingStatus.TRACKING,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )
    await store.update_perception(ARM, perc)
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.perception == perc


async def test_update_act(store: RuntimeStateStore):
    act = ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=300, buffer_low=False)
    await store.update_act(ARM, act)
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.act == act


async def test_update_safety(store: RuntimeStateStore):
    safety = SafetyInfo(state=SafetyState.FAULT, reflex_active=True, reason_codes=())
    await store.update_safety(ARM, safety)
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert snap.safety.state == SafetyState.FAULT
    assert snap.safety.reflex_active is True


# ---------------------------------------------------------------------------
# Snapshot identity and caching
# ---------------------------------------------------------------------------


async def test_snapshot_ids_increment(store: RuntimeStateStore):
    snap1 = await store.build_and_cache_snapshot(ARM, [])
    snap2 = await store.build_and_cache_snapshot(ARM, [])
    assert snap1.snapshot_id != snap2.snapshot_id
    assert snap1.arm_id == ARM


async def test_get_latest_snapshot_returns_last(store: RuntimeStateStore):
    assert await store.get_latest_snapshot(ARM) is None
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert await store.get_latest_snapshot(ARM) == snap

    snap2 = await store.build_and_cache_snapshot(ARM, [])
    assert await store.get_latest_snapshot(ARM) == snap2  # replaced, not appended


# ---------------------------------------------------------------------------
# Command ack ring
# ---------------------------------------------------------------------------


async def test_command_ack_ring_max_size(store: RuntimeStateStore):
    for i in range(RuntimeStateStore.COMMAND_ACK_RING_SIZE + 5):
        await store.add_command_ack(
            ARM,
            CommandAck(command_id=f"cmd-{i}", status=CommandAckStatus.ACCEPTED),
        )
    snap = await store.build_and_cache_snapshot(ARM, [])
    assert len(snap.command_acks) == RuntimeStateStore.COMMAND_ACK_RING_SIZE
    # Oldest dropped — last inserted command should be present
    last_id = f"cmd-{RuntimeStateStore.COMMAND_ACK_RING_SIZE + 4}"
    assert any(a.command_id == last_id for a in snap.command_acks)


# ---------------------------------------------------------------------------
# Multi-arm isolation
# ---------------------------------------------------------------------------


async def test_two_arms_are_independent(store: RuntimeStateStore):
    store.register_arm(ARM2)
    skill0 = SkillInfo(name=SkillName.PICK, skill_run_id="run-0", phase=PhaseId.LIFT)
    skill1 = SkillInfo(name=SkillName.PLACE, skill_run_id="run-1", phase=PhaseId.DESCEND_PLACE)

    await store.update_skill(ARM, skill0)
    await store.update_skill(ARM2, skill1)

    snap0 = await store.build_and_cache_snapshot(ARM, [])
    snap1 = await store.build_and_cache_snapshot(ARM2, [])

    assert snap0.skill == skill0
    assert snap1.skill == skill1
    assert snap0.arm_id == ARM
    assert snap1.arm_id == ARM2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_updates_do_not_corrupt(store: RuntimeStateStore):
    async def update_skill(i: int) -> None:
        skill = SkillInfo(
            name=SkillName.PICK,
            skill_run_id=f"run-{i}",
            phase=PhaseId.SELECT_GRASP,
        )
        await store.update_skill(ARM, skill)

    await asyncio.gather(*[update_skill(i) for i in range(50)])
    snap = await store.build_and_cache_snapshot(ARM, [])
    # skill must be one of the values we wrote (not corrupted)
    assert snap.skill is not None
    assert snap.skill.name == SkillName.PICK
