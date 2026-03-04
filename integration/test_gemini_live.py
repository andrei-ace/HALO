"""
Integration tests for LivePlannerSession with Gemini Live API.

These tests require a valid GOOGLE_API_KEY environment variable.
They are kept in the integration/ directory — auto-skipped if no API key.

Run explicitly:
    uv run pytest integration/test_gemini_live.py -v
    uv run pytest integration/test_gemini_live.py -v -s  # show reasoning

Environment:
    GOOGLE_API_KEY        Gemini API key (required)
    HALO_LIVE_MODEL       Model to use (default: gemini-2.5-flash)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from halo.cognitive.config import LiveConfig
from halo.cognitive.live_session import LivePlannerSession
from halo.contracts.enums import (
    ActStatus,
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

pytestmark = [pytest.mark.cloud_integration]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIVE_MODEL = os.getenv("HALO_LIVE_MODEL", "gemini-2.5-flash")
_TS_MS = 268_245_429


def _make_session() -> LivePlannerSession:
    prompts_dir = Path(__file__).parents[1] / "configs" / "planner"
    config = LiveConfig(
        planner_model=LIVE_MODEL,
        audio_enabled=False,
        response_modalities=("TEXT",),
    )
    return LivePlannerSession(config=config, prompts_dir=prompts_dir)


# ---------------------------------------------------------------------------
# Snapshot factories
# ---------------------------------------------------------------------------


def _idle_snap() -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-idle-001",
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


def _idle_snap_target_tracked() -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-idle-002",
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
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False, wrist_enabled=False),
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
async def test_live_session_text_only_decide() -> None:
    """Text-only LivePlannerSession should produce reasoning on idle snapshot."""
    session = _make_session()
    try:
        snap = _idle_snap()
        cmds = await session.decide(snap)
        # Session should have produced some reasoning
        assert session.last_reasoning, "Expected non-empty reasoning"
        print(f"\nReasoning: {session.last_reasoning}")
        print(f"Commands: {[c.type.value for c in cmds]}")
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_live_session_describe_scene() -> None:
    """Text-only LivePlannerSession should issue DESCRIBE_SCENE when asked."""
    session = _make_session()
    try:
        snap = _idle_snap()
        cmds = await session.decide(snap, operator_cmd="What do you see on the table?")
        print(f"\nReasoning: {session.last_reasoning}")
        print(f"Commands: {[c.type.value for c in cmds]}")
        # Should have issued at least one command (likely DESCRIBE_SCENE)
        assert len(cmds) >= 1, f"Expected at least 1 command, got {len(cmds)}"
        types = [c.type for c in cmds]
        assert CommandType.DESCRIBE_SCENE in types, f"Expected DESCRIBE_SCENE in {types}"
    finally:
        await session.stop()
