"""Tests for CommandRouter: idempotency, precondition, skill-run enforcement."""

import asyncio

import pytest

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandEnvelope,
    OverrideTargetPayload,
    StartSkillPayload,
)
from halo.contracts.enums import CommandAckStatus, CommandType, PhaseId, SkillName
from halo.contracts.snapshots import SkillInfo
from halo.runtime.runtime import HALORuntime

ARM = "arm0"


@pytest.fixture
def rt() -> HALORuntime:
    r = HALORuntime()
    r.register_arm(ARM)
    return r


def _start(command_id: str = "cmd-1", precondition: str | None = None) -> CommandEnvelope:
    return CommandEnvelope(
        command_id=command_id,
        arm_id=ARM,
        issued_at_ms=1000,
        type=CommandType.START_SKILL,
        payload=StartSkillPayload(skill_name=SkillName.PICK, target_handle="cube-1"),
        precondition_snapshot_id=precondition,
    )


def _abort(command_id: str, skill_run_id: str) -> CommandEnvelope:
    return CommandEnvelope(
        command_id=command_id,
        arm_id=ARM,
        issued_at_ms=1000,
        type=CommandType.ABORT_SKILL,
        payload=AbortSkillPayload(skill_run_id=skill_run_id, reason="test"),
    )


def _override(command_id: str, skill_run_id: str) -> CommandEnvelope:
    return CommandEnvelope(
        command_id=command_id,
        arm_id=ARM,
        issued_at_ms=1000,
        type=CommandType.OVERRIDE_TARGET,
        payload=OverrideTargetPayload(skill_run_id=skill_run_id, target_handle="cube-2"),
    )


# ---------------------------------------------------------------------------
# Accept cases
# ---------------------------------------------------------------------------

async def test_accept_no_precondition(rt: HALORuntime):
    ack = await rt.submit_command(_start())
    assert ack.status == CommandAckStatus.ACCEPTED


async def test_accept_matching_precondition(rt: HALORuntime):
    snap = await rt.get_latest_runtime_snapshot(ARM)
    ack = await rt.submit_command(_start(command_id="cmd-pre", precondition=snap.snapshot_id))
    assert ack.status == CommandAckStatus.ACCEPTED


async def test_accepted_ack_appears_in_next_snapshot(rt: HALORuntime):
    await rt.submit_command(_start(command_id="cmd-visible"))
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert any(a.command_id == "cmd-visible" for a in snap.command_acks)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

async def test_duplicate_command_id_returns_already_applied(rt: HALORuntime):
    await rt.submit_command(_start(command_id="cmd-dup"))
    ack2 = await rt.submit_command(_start(command_id="cmd-dup"))
    assert ack2.status == CommandAckStatus.ALREADY_APPLIED


async def test_already_applied_does_not_add_another_ack(rt: HALORuntime):
    await rt.submit_command(_start(command_id="cmd-once"))
    snap_before = await rt.get_latest_runtime_snapshot(ARM)
    count_before = len(snap_before.command_acks)

    await rt.submit_command(_start(command_id="cmd-once"))  # replay

    snap_after = await rt.get_latest_runtime_snapshot(ARM)
    assert len(snap_after.command_acks) == count_before


# ---------------------------------------------------------------------------
# Stale precondition
# ---------------------------------------------------------------------------

async def test_stale_precondition_rejected(rt: HALORuntime):
    ack = await rt.submit_command(_start(command_id="cmd-stale", precondition="snap-arm0-9999"))
    assert ack.status == CommandAckStatus.REJECTED_STALE
    assert ack.reason is not None


async def test_stale_precondition_when_no_snapshot_built_yet(rt: HALORuntime):
    rt.register_arm("arm2")
    cmd = CommandEnvelope(
        command_id="cmd-new",
        arm_id="arm2",
        issued_at_ms=1000,
        type=CommandType.START_SKILL,
        payload=StartSkillPayload(skill_name=SkillName.PICK, target_handle="cube-1"),
        precondition_snapshot_id="snap-arm2-1",
    )
    ack = await rt.submit_command(cmd)
    assert ack.status == CommandAckStatus.REJECTED_STALE


async def test_stale_ack_recorded_in_store(rt: HALORuntime):
    await rt.submit_command(_start(command_id="cmd-stale2", precondition="snap-arm0-9999"))
    snap = await rt.get_latest_runtime_snapshot(ARM)
    assert any(a.command_id == "cmd-stale2" for a in snap.command_acks)


# ---------------------------------------------------------------------------
# Wrong skill run
# ---------------------------------------------------------------------------

async def test_abort_wrong_skill_run_rejected(rt: HALORuntime):
    # No current skill set → any skill_run_id is wrong
    ack = await rt.submit_command(_abort("cmd-abort-bad", skill_run_id="run-999"))
    assert ack.status == CommandAckStatus.REJECTED_WRONG_SKILL_RUN
    assert ack.reason is not None


async def test_abort_matching_skill_run_accepted(rt: HALORuntime):
    skill = SkillInfo(name=SkillName.PICK, skill_run_id="run-1", phase=PhaseId.APPROACH_PREGRASP)
    await rt.store.update_skill(ARM, skill)
    await rt.get_latest_runtime_snapshot(ARM)  # cache snapshot with skill set

    ack = await rt.submit_command(_abort("cmd-abort-ok", skill_run_id="run-1"))
    assert ack.status == CommandAckStatus.ACCEPTED


async def test_override_wrong_skill_run_rejected(rt: HALORuntime):
    ack = await rt.submit_command(_override("cmd-override-bad", skill_run_id="run-wrong"))
    assert ack.status == CommandAckStatus.REJECTED_WRONG_SKILL_RUN


async def test_override_matching_skill_run_accepted(rt: HALORuntime):
    skill = SkillInfo(name=SkillName.PICK, skill_run_id="run-2", phase=PhaseId.APPROACH_PREGRASP)
    await rt.store.update_skill(ARM, skill)
    await rt.get_latest_runtime_snapshot(ARM)

    ack = await rt.submit_command(_override("cmd-override-ok", skill_run_id="run-2"))
    assert ack.status == CommandAckStatus.ACCEPTED


# ---------------------------------------------------------------------------
# Concurrent submission
# ---------------------------------------------------------------------------

async def test_concurrent_unique_commands_all_accepted(rt: HALORuntime):
    cmds = [_start(command_id=f"cmd-{i}") for i in range(20)]
    acks = await asyncio.gather(*[rt.submit_command(c) for c in cmds])
    assert all(a.status == CommandAckStatus.ACCEPTED for a in acks)
    assert len({a.command_id for a in acks}) == 20


async def test_concurrent_duplicate_commands_idempotent(rt: HALORuntime):
    # Send the same command_id 10 times concurrently
    cmds = [_start(command_id="cmd-concurrent-dup") for _ in range(10)]
    acks = await asyncio.gather(*[rt.submit_command(c) for c in cmds])
    accepted = sum(1 for a in acks if a.status == CommandAckStatus.ACCEPTED)
    already_applied = sum(1 for a in acks if a.status == CommandAckStatus.ALREADY_APPLIED)
    assert accepted == 1
    assert already_applied == 9
