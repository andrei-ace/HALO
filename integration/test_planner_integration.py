"""
Integration tests for PlannerAgent with a real LLM.

These tests require a running Ollama instance with the configured model.
They are kept outside the `tests/` directory on purpose — the standard
`uv run pytest` run (testpaths = ["tests"]) will never discover them.

Run explicitly:
    uv run pytest integration/ -v
    uv run pytest integration/ -v -s           # show LLM reasoning trace
    uv run pytest integration/ -v -k pick      # single scenario

Environment:
    HALO_OLLAMA_URL    Ollama base URL  (default: http://localhost:11434)
    HALO_MODEL_NAME    Model to use     (default: gpt-oss)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from halo.contracts.commands import CommandEnvelope
from halo.contracts.enums import (
    ActStatus,
    CommandType,
    PerceptionFailureCode,
    PhaseId,
    SafetyReflexReason,
    SafetyState,
    SkillFailureCode,
    SkillName,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    SkillInfo,
    TargetInfo,
)
from halo.services.planner_service.agent import PlannerAgent

pytestmark = [pytest.mark.integration]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("HALO_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("HALO_MODEL_NAME", "gpt-oss")

# Fixed timestamp — avoids nondeterminism if the agent ever reasons about
# time deltas and makes failure reproduction easier.
_TS_MS = 268_245_429


def _make_agent() -> PlannerAgent:
    prompts_dir = Path(__file__).parents[1] / "configs" / "planner"
    return PlannerAgent(model_name=MODEL_NAME, base_url=OLLAMA_URL, prompts_dir=prompts_dir)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_single_command(
    cmds: list[CommandEnvelope],
    expected_type: CommandType,
    context: str = "",
) -> CommandEnvelope:
    """Assert exactly one command of the expected type was returned."""
    suffix = f" ({context})" if context else ""
    types = [c.type for c in cmds]
    assert len(cmds) == 1, f"Expected exactly 1 command{suffix}, got {len(cmds)}: {types}"
    assert cmds[0].type == expected_type, f"Expected {expected_type}{suffix}, got {types[0]}"
    return cmds[0]


def _assert_no_commands(cmds: list[CommandEnvelope], context: str = "") -> None:
    """Assert the agent issued no commands."""
    suffix = f" ({context})" if context else ""
    types = [c.type for c in cmds]
    assert len(cmds) == 0, f"Expected 0 commands{suffix}, got {len(cmds)}: {types}"


def _assert_no_start_skill(cmds: list[CommandEnvelope], context: str = "") -> None:
    """Assert no START_SKILL command was issued (other commands are allowed)."""
    suffix = f" ({context})" if context else ""
    starts = [c for c in cmds if c.type == CommandType.START_SKILL]
    assert len(starts) == 0, f"Expected no START_SKILL{suffix}, got: {starts}"


# ---------------------------------------------------------------------------
# Snapshot factories — realistic but fully synthetic, fixed timestamps
# ---------------------------------------------------------------------------


def _idle_snap_target_tracked() -> PlannerSnapshot:
    """Arm idle, cube-1 confidently tracked, safety OK."""
    return PlannerSnapshot(
        snapshot_id="snap-idle-001",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.91,
            obs_age_ms=18,
            time_skew_ms=-3,
            delta_xyz_ee=(0.05, -0.02, 0.12),
            distance_m=0.14,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.IN_PROGRESS,
            reason_code=None,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(),
    )


def _pick_running_snap() -> PlannerSnapshot:
    """PICK skill in progress (APPROACH_PREGRASP), target tracked."""
    return PlannerSnapshot(
        snapshot_id="snap-pick-002",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=SkillInfo(
            name=SkillName.PICK,
            skill_run_id="run-42",
            phase=PhaseId.APPROACH_PREGRASP,
        ),
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.87,
            obs_age_ms=22,
            time_skew_ms=-2,
            delta_xyz_ee=(0.03, -0.01, 0.08),
            distance_m=0.09,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=240, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=3200, no_progress_ms=0, delta_distance=-0.008),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.IN_PROGRESS,
            reason_code=None,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(
            EventEnvelope(
                event_id="evt-10",
                type=EventType.SKILL_STARTED,
                ts_ms=_TS_MS - 3200,
                arm_id="arm0",
                data={"skill": "PICK"},
            ),
        ),
    )


def _pick_failed_snap(
    reason: SkillFailureCode,
    reacquire_fail_count: int = 0,
    extra_events: tuple[EventEnvelope, ...] = (),
) -> PlannerSnapshot:
    """PICK just failed — skill cleared, outcome carries the reason."""
    return PlannerSnapshot(
        snapshot_id="snap-fail-003",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.78,
            obs_age_ms=35,
            time_skew_ms=-5,
            delta_xyz_ee=(0.04, -0.01, 0.10),
            distance_m=0.11,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=reacquire_fail_count,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=7800, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.FAILURE,
            reason_code=reason,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(
            *extra_events,
            EventEnvelope(
                event_id="evt-21",
                type=EventType.SKILL_FAILED,
                ts_ms=_TS_MS - 200,
                arm_id="arm0",
                data={"skill": "PICK", "reason": reason.value},
            ),
        ),
    )


def _target_lost_snap(reacquire_fail_count: int = 1) -> PlannerSnapshot:
    """Arm idle, target perception is LOST."""
    return PlannerSnapshot(
        snapshot_id="snap-lost-004",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=None,
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.OUT_OF_VIEW,
            reacquire_fail_count=reacquire_fail_count,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.FAILURE,
            reason_code=SkillFailureCode.TIMEOUT,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(
            EventEnvelope(
                event_id="evt-30",
                type=EventType.PERCEPTION_FAILURE,
                ts_ms=_TS_MS - 500,
                arm_id="arm0",
                data={"failure_code": "OUT_OF_VIEW"},
            ),
        ),
    )


def _safety_reflex_snap() -> PlannerSnapshot:
    """Safety reflex is active (JOINT_LIMIT), arm idle."""
    return PlannerSnapshot(
        snapshot_id="snap-safe-005",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle="cube-1",
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=500,
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
        outcome=OutcomeInfo(
            state=SkillOutcomeState.FAILURE,
            reason_code=SkillFailureCode.UNSAFE_ABORT,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.FAULT,
            reflex_active=True,
            reason_codes=(SafetyReflexReason.JOINT_LIMIT,),
        ),
        command_acks=(),
        recent_events=(
            EventEnvelope(
                event_id="evt-50",
                type=EventType.SAFETY_REFLEX_TRIGGERED,
                ts_ms=_TS_MS - 100,
                arm_id="arm0",
                data={"reason": "JOINT_LIMIT"},
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests — existing scenarios (tightened assertions)
# ---------------------------------------------------------------------------


async def test_pick_command_starts_skill() -> None:
    """
    Scenario: arm idle, target tracked.
    Operator: "Pick the cube."
    Expected: exactly 1 command — START_SKILL(PICK, cube-1).
    """
    agent = _make_agent()
    snap = _idle_snap_target_tracked()

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    cmd = _assert_single_command(cmds, CommandType.START_SKILL)
    assert cmd.payload.skill_name == SkillName.PICK
    assert cmd.payload.target_handle == "cube-1"


async def test_abort_command_while_skill_running() -> None:
    """
    Scenario: PICK is running.
    Operator: "Abort, wrong target."
    Expected: exactly 1 command — ABORT_SKILL(run-42).
    """
    agent = _make_agent()
    snap = _pick_running_snap()

    cmds = await agent.decide(snap, operator_cmd="Abort, that's the wrong target.")

    cmd = _assert_single_command(cmds, CommandType.ABORT_SKILL)
    assert cmd.payload.skill_run_id == "run-42"


async def test_retry_after_no_grasp_failure() -> None:
    """
    Scenario: PICK just failed with NO_GRASP (first occurrence), target tracked.
    Operator: "Try picking again."
    Expected: exactly 1 command — START_SKILL(PICK).
    """
    agent = _make_agent()
    snap = _pick_failed_snap(SkillFailureCode.NO_GRASP)

    cmds = await agent.decide(snap, operator_cmd="Try picking again.")

    _assert_single_command(cmds, CommandType.START_SKILL)


async def test_perception_refresh_when_target_lost() -> None:
    """
    Scenario: target is LOST (OUT_OF_VIEW), reacquire count low, arm idle.
    Operator: "Find the cube again."
    Expected: exactly 1 command — DESCRIBE_SCENE.
    """
    agent = _make_agent()
    snap = _target_lost_snap(reacquire_fail_count=1)

    cmds = await agent.decide(snap, operator_cmd="Find the cube again.")

    _assert_single_command(cmds, CommandType.DESCRIBE_SCENE)


async def test_no_command_when_safety_reflex_active() -> None:
    """
    Scenario: safety reflex active (JOINT_LIMIT).
    Operator: "Pick the cube."
    Expected: 0 commands — safety rules block all skill starts.
    """
    agent = _make_agent()
    snap = _safety_reflex_snap()

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    _assert_no_start_skill(cmds, context="safety reflex active")


# ---------------------------------------------------------------------------
# Tests — important gaps
# ---------------------------------------------------------------------------


async def test_no_op_without_operator_instruction() -> None:
    """
    Scenario: arm idle, target tracked — but no operator instruction.
    Expected: 0 commands. The agent must not start skills autonomously.
    """
    agent = _make_agent()
    snap = _idle_snap_target_tracked()

    cmds = await agent.decide(snap, operator_cmd=None)

    _assert_no_commands(cmds, context="no operator instruction")


async def test_retry_limit_stops_after_3_no_grasp() -> None:
    """
    Scenario: same NO_GRASP failure has occurred 3 times (visible in recent_events).
    Operator: "Try again."
    Expected: 0 commands — retry limit reached, wait for operator intervention.
    """
    agent = _make_agent()

    prior_failures = (
        EventEnvelope(
            event_id="evt-f1",
            type=EventType.SKILL_FAILED,
            ts_ms=_TS_MS - 15000,
            arm_id="arm0",
            data={"skill": "PICK", "reason": "NO_GRASP"},
        ),
        EventEnvelope(
            event_id="evt-f2",
            type=EventType.SKILL_FAILED,
            ts_ms=_TS_MS - 10000,
            arm_id="arm0",
            data={"skill": "PICK", "reason": "NO_GRASP"},
        ),
        EventEnvelope(
            event_id="evt-f3",
            type=EventType.SKILL_FAILED,
            ts_ms=_TS_MS - 5000,
            arm_id="arm0",
            data={"skill": "PICK", "reason": "NO_GRASP"},
        ),
    )
    snap = _pick_failed_snap(SkillFailureCode.NO_GRASP, extra_events=prior_failures)

    cmds = await agent.decide(snap, operator_cmd="Try again.")

    _assert_no_commands(cmds, context="retry limit (3x NO_GRASP)")


async def test_no_refresh_when_reacquire_exhausted() -> None:
    """
    Scenario: reacquire_fail_count >= 3 — perception has given up.
    Operator: "Find the cube again."
    Expected: 0 commands — do not keep requesting refreshes.
    """
    agent = _make_agent()
    snap = _target_lost_snap(reacquire_fail_count=3)

    cmds = await agent.decide(snap, operator_cmd="Find the cube again.")

    refresh_cmds = [c for c in cmds if c.type == CommandType.DESCRIBE_SCENE]
    assert len(refresh_cmds) == 0, f"Agent must not request refresh when reacquire_fail_count >= 3, got: {cmds}"


async def test_no_op_on_safety_fault_without_reflex() -> None:
    """
    Scenario: safety.state=FAULT but reflex_active=False (arm stabilizing).
    Operator: "Pick the cube."
    Expected: 0 commands — system prompt says no-op and wait to stabilize.
    """
    agent = _make_agent()
    snap = PlannerSnapshot(
        snapshot_id="snap-fault-006",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.80,
            obs_age_ms=40,
            time_skew_ms=0,
            delta_xyz_ee=(0.05, -0.02, 0.12),
            distance_m=0.14,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.FAILURE,
            reason_code=SkillFailureCode.UNSAFE_ABORT,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.FAULT,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(
            EventEnvelope(
                event_id="evt-60",
                type=EventType.SAFETY_REFLEX_TRIGGERED,
                ts_ms=_TS_MS - 2000,
                arm_id="arm0",
                data={"reason": "JOINT_LIMIT"},
            ),
        ),
    )

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    _assert_no_commands(cmds, context="safety FAULT without reflex")


async def test_place_not_issued_in_v0() -> None:
    """
    Scenario: arm idle, PICK just succeeded. Operator says "place it".
    Expected: no START_SKILL(PLACE) — PLACE is not available in v0.
    """
    agent = _make_agent()
    snap = PlannerSnapshot(
        snapshot_id="snap-place-007",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.90,
            obs_age_ms=20,
            time_skew_ms=-2,
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=5000, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.SUCCESS,
            reason_code=None,
            needs_verify=False,
        ),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(),
        recent_events=(
            EventEnvelope(
                event_id="evt-70",
                type=EventType.SKILL_SUCCEEDED,
                ts_ms=_TS_MS - 100,
                arm_id="arm0",
                data={"skill": "PICK"},
            ),
        ),
    )

    cmds = await agent.decide(snap, operator_cmd="Great, now place it on the table.")

    place_cmds = [c for c in cmds if c.type == CommandType.START_SKILL and c.payload.skill_name == SkillName.PLACE]
    assert len(place_cmds) == 0, f"Agent must not issue START_SKILL(PLACE) in v0, got: {place_cmds}"


async def test_no_start_skill_while_already_running() -> None:
    """
    Scenario: PICK is already running. Operator says "pick the cube".
    Expected: no START_SKILL — must abort the current skill first (or no-op).
    The agent may issue ABORT_SKILL, but must not start a second skill in the
    same tick.
    """
    agent = _make_agent()
    snap = _pick_running_snap()

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    _assert_no_start_skill(cmds, context="skill already running")
