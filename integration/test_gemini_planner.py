"""
Integration tests for PlannerAgent with Gemini (cloud backend).

These tests require a valid GOOGLE_API_KEY environment variable.
They are kept outside the `tests/` directory on purpose — the standard
`uv run pytest` run (testpaths = ["tests"]) will never discover them.

Run explicitly:
    uv run pytest integration/test_gemini_planner.py -v
    uv run pytest integration/test_gemini_planner.py -v -s  # show reasoning

Environment:
    GOOGLE_API_KEY        Gemini API key (required)
    HALO_CLOUD_MODEL      Model to use (default: gemini-3.1-flash-lite-preview)
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
    SafetyState,
    SkillName,
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
from halo.services.planner_service.agent import PlannerAgent

pytestmark = [pytest.mark.cloud_integration]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLOUD_MODEL = os.getenv("HALO_CLOUD_MODEL", "gemini-3.1-flash-lite-preview")
_TS_MS = 268_245_429


def _make_agent() -> PlannerAgent:
    prompts_dir = Path(__file__).parents[1] / "configs" / "planner"
    return PlannerAgent(model_name=CLOUD_MODEL, base_url="", prompts_dir=prompts_dir, backend="cloud")


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_single_command(
    cmds: list[CommandEnvelope],
    expected_type: CommandType,
    context: str = "",
) -> CommandEnvelope:
    suffix = f" ({context})" if context else ""
    types = [c.type for c in cmds]
    assert len(cmds) == 1, f"Expected exactly 1 command{suffix}, got {len(cmds)}: {types}"
    assert cmds[0].type == expected_type, f"Expected {expected_type}{suffix}, got {types[0]}"
    return cmds[0]


# ---------------------------------------------------------------------------
# Snapshot factories
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
            delta_xyz_ee=(0.05, -0.02, 0.08),
            distance_m=0.10,
            center_px=(0.55, 0.60),
            bbox_xywh=(0.45, 0.50, 0.20, 0.20),
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
        ),
        act=ActInfo(
            status=ActStatus.IDLE,
            buffer_fill_ms=0,
            buffer_low=False,
            wrist_enabled=False,
        ),
        progress=ProgressInfo(elapsed_ms=0, attempts=0),
        outcome=OutcomeInfo(state=SkillOutcomeState.NOT_STARTED),
        safety=SafetyInfo(state=SafetyState.OK, reflex_reasons=()),
        command_acks=[],
        recent_events=[],
        held_object_handle=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_pick_command() -> None:
    """Gemini agent should issue a START_SKILL command when asked to pick."""
    agent = _make_agent()
    snap = _idle_snap_target_tracked()
    cmds = await agent.decide(snap, operator_cmd="Pick up cube-1.")

    cmd = _assert_single_command(cmds, CommandType.START_SKILL, context="pick cube-1")
    assert cmd.payload.skill_name == SkillName.PICK
    assert cmd.payload.target_handle == "cube-1"
    print(f"\nReasoning: {agent.last_reasoning}")


@pytest.mark.asyncio
async def test_gemini_describe_scene() -> None:
    """Gemini agent should issue DESCRIBE_SCENE when asked what it sees."""
    agent = _make_agent()
    snap = _idle_snap_target_tracked()
    snap = PlannerSnapshot(
        snapshot_id="snap-idle-002",
        ts_ms=_TS_MS,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(handle=None, hint_valid=False, confidence=0.0),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.OK,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False, wrist_enabled=False),
        progress=ProgressInfo(elapsed_ms=0, attempts=0),
        outcome=OutcomeInfo(state=SkillOutcomeState.NOT_STARTED),
        safety=SafetyInfo(state=SafetyState.OK, reflex_reasons=()),
        command_acks=[],
        recent_events=[],
        held_object_handle=None,
    )
    cmds = await agent.decide(snap, operator_cmd="What do you see?")

    _assert_single_command(cmds, CommandType.DESCRIBE_SCENE, context="describe scene")
    print(f"\nReasoning: {agent.last_reasoning}")
