"""
Integration tests for PlannerService + PlannerAgent with a real LLM.

These tests require a running Ollama instance with the configured model.
They are kept outside the `tests/` directory on purpose — the standard
`uv run pytest` run (testpaths = ["tests"]) will never discover them.

Run explicitly:
    uv run pytest integration/ -v
    uv run pytest integration/ -v -s           # show LLM reasoning trace
    uv run pytest integration/ -v -k pick      # single scenario

Environment:
    HALO_OLLAMA_URL    Ollama base URL  (default: http://localhost:11434)
    HALO_MODEL_NAME    Model to use     (default: gpt-oss:20B)
"""
from __future__ import annotations

import os
import time

import pytest

from halo.contracts.commands import CommandAck
from halo.contracts.enums import (
    ActStatus,
    CommandAckStatus,
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("HALO_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("HALO_MODEL_NAME", "gpt-oss:20B")
PROMPTS_DIR = None  # defaults to configs/planner/ relative to package root


def _make_agent() -> PlannerAgent:
    from pathlib import Path

    prompts_dir = Path(__file__).parents[1] / "configs" / "planner"
    return PlannerAgent(model_name=MODEL_NAME, base_url=OLLAMA_URL, prompts_dir=prompts_dir)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


# ---------------------------------------------------------------------------
# Snapshot factories — realistic but fully synthetic
# ---------------------------------------------------------------------------

def _idle_snap_target_tracked() -> PlannerSnapshot:
    """Arm idle, cube-1 is confidently tracked, safety OK."""
    return PlannerSnapshot(
        snapshot_id="snap-idle-001",
        ts_ms=_now_ms(),
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
    """PICK skill in progress (APPROACH_PREGRASP phase), target tracked."""
    return PlannerSnapshot(
        snapshot_id="snap-pick-002",
        ts_ms=_now_ms(),
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
                ts_ms=_now_ms() - 3200,
                arm_id="arm0",
                data={"skill": "PICK"},
            ),
        ),
    )


def _pick_failed_snap(reason: SkillFailureCode) -> PlannerSnapshot:
    """PICK just failed — skill cleared, outcome carries the reason."""
    return PlannerSnapshot(
        snapshot_id="snap-fail-003",
        ts_ms=_now_ms(),
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
            reacquire_fail_count=0,
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
            EventEnvelope(
                event_id="evt-21",
                type=EventType.SKILL_FAILED,
                ts_ms=_now_ms() - 200,
                arm_id="arm0",
                data={"skill": "PICK", "reason": reason.value},
            ),
        ),
    )


def _target_lost_snap() -> PlannerSnapshot:
    """Arm idle, target perception is LOST."""
    return PlannerSnapshot(
        snapshot_id="snap-lost-004",
        ts_ms=_now_ms(),
        arm_id="arm0",
        skill=None,
        target=None,
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.OUT_OF_VIEW,
            reacquire_fail_count=1,
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
                ts_ms=_now_ms() - 500,
                arm_id="arm0",
                data={"failure_code": "OUT_OF_VIEW"},
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pick_command_starts_skill() -> None:
    """
    Scenario: arm idle, target tracked.
    Operator says "pick the cube".
    Expected: agent calls start_skill(PICK, cube-1).
    """
    agent = _make_agent()
    snap = _idle_snap_target_tracked()

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    assert len(cmds) >= 1, "Expected at least one command"
    types = [c.type for c in cmds]
    assert CommandType.START_SKILL in types, (
        f"Expected START_SKILL, got: {types}"
    )
    pick_cmd = next(c for c in cmds if c.type == CommandType.START_SKILL)
    assert pick_cmd.payload.skill_name == SkillName.PICK
    assert pick_cmd.payload.target_handle == "cube-1"


@pytest.mark.asyncio
async def test_abort_command_while_skill_running() -> None:
    """
    Scenario: PICK is running.
    Operator says "abort, wrong target".
    Expected: agent calls abort_skill for run-42.
    """
    agent = _make_agent()
    snap = _pick_running_snap()

    cmds = await agent.decide(snap, operator_cmd="Abort, that's the wrong target.")

    assert len(cmds) >= 1, "Expected at least one command"
    types = [c.type for c in cmds]
    assert CommandType.ABORT_SKILL in types, (
        f"Expected ABORT_SKILL, got: {types}"
    )
    abort_cmd = next(c for c in cmds if c.type == CommandType.ABORT_SKILL)
    assert abort_cmd.payload.skill_run_id == "run-42"


@pytest.mark.asyncio
async def test_retry_after_no_grasp_failure() -> None:
    """
    Scenario: PICK just failed with NO_GRASP, arm idle, target still tracked.
    Operator says "try again".
    Expected: agent calls start_skill(PICK) again.
    """
    agent = _make_agent()
    snap = _pick_failed_snap(SkillFailureCode.NO_GRASP)

    cmds = await agent.decide(snap, operator_cmd="Try picking again.")

    assert len(cmds) >= 1, "Expected at least one command"
    types = [c.type for c in cmds]
    assert CommandType.START_SKILL in types, (
        f"Expected START_SKILL retry, got: {types}"
    )


@pytest.mark.asyncio
async def test_perception_refresh_when_target_lost() -> None:
    """
    Scenario: target is LOST (OUT_OF_VIEW), arm idle.
    Operator says "find the cube again".
    Expected: agent calls request_perception_refresh.
    """
    agent = _make_agent()
    snap = _target_lost_snap()

    cmds = await agent.decide(snap, operator_cmd="Find the cube again.")

    assert len(cmds) >= 1, "Expected at least one command"
    types = [c.type for c in cmds]
    assert CommandType.REQUEST_PERCEPTION_REFRESH in types, (
        f"Expected REQUEST_PERCEPTION_REFRESH, got: {types}"
    )


@pytest.mark.asyncio
async def test_no_command_when_safety_reflex_active() -> None:
    """
    Scenario: safety reflex is active.
    Operator says "pick the cube".
    Expected: agent issues no skill commands (safety rules must block it).
    """
    agent = _make_agent()
    snap = PlannerSnapshot(
        snapshot_id="snap-safe-005",
        ts_ms=_now_ms(),
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
                ts_ms=_now_ms() - 100,
                arm_id="arm0",
                data={"reason": "JOINT_LIMIT"},
            ),
        ),
    )

    cmds = await agent.decide(snap, operator_cmd="Pick the cube.")

    skill_cmds = [c for c in cmds if c.type == CommandType.START_SKILL]
    assert len(skill_cmds) == 0, (
        f"Agent must not start a skill while safety reflex is active, got: {cmds}"
    )
