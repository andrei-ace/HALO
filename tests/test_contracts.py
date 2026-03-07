"""Tests for contract types: enums, commands, snapshots, events."""

import dataclasses

import pytest

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandAck,
    CommandEnvelope,
    DescribeScenePayload,
    OverrideTargetPayload,
    StartSkillPayload,
)
from halo.contracts.enums import (
    ActStatus,
    CommandAckStatus,
    CommandType,
    PerceptionFailureCode,
    PhaseId,
    PlaceModifier,
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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_phase_id_values():
    assert PhaseId.IDLE == 0
    assert PhaseId.SELECT_GRASP == 1
    assert PhaseId.PLAN_APPROACH == 2
    assert PhaseId.MOVE_PREGRASP == 3
    assert PhaseId.VISUAL_ALIGN == 4
    assert PhaseId.EXECUTE_APPROACH == 5
    assert PhaseId.CLOSE_GRIPPER == 6
    assert PhaseId.LIFT == 7
    assert PhaseId.VERIFY_GRASP == 8
    assert PhaseId.DONE == 9
    assert PhaseId.TRANSIT_PREPLACE == 30
    assert PhaseId.SELECT_PLACE == 34
    assert PhaseId.RECOVER_RETRY_APPROACH == 50
    assert PhaseId.RECOVER_REGRASP == 51
    assert PhaseId.RECOVER_ABORT == 52


def test_perception_failure_codes_complete():
    expected = {
        "OK",
        "OCCLUDED",
        "OUT_OF_VIEW",
        "DEPTH_INVALID",
        "MULTIPLE_CANDIDATES",
        "CALIB_INVALID",
        "TRACK_JUMP_REJECTED",
        "REACQUIRE_FAILED",
    }
    assert {c.value for c in PerceptionFailureCode} == expected


def test_skill_failure_codes_complete():
    expected = {
        "TIMEOUT",
        "NO_PROGRESS",
        "NO_GRASP",
        "DROP_DETECTED",
        "PLACE_MISS",
        "PERCEPTION_LOST",
        "TARGET_MISMATCH",
        "UNSAFE_ABORT",
    }
    assert {c.value for c in SkillFailureCode} == expected


def test_safety_reflex_reasons_complete():
    expected = {"JOINT_LIMIT", "WORKSPACE_LIMIT", "COLLISION_RISK", "OVERCURRENT", "ESTOP"}
    assert {c.value for c in SafetyReflexReason} == expected


def test_str_enum_values_are_strings():
    assert str(PerceptionFailureCode.OK) == "OK"
    assert str(TrackingStatus.TRACKING) == "TRACKING"
    assert str(SkillName.PICK) == "PICK"


def test_place_modifier_values():
    assert PlaceModifier.PLACE_FLOOR == "PLACE_FLOOR"
    assert PlaceModifier.PLACE_NEXT_TO == "PLACE_NEXT_TO"
    assert PlaceModifier.PLACE_IN_TRAY == "PLACE_IN_TRAY"
    assert {m.value for m in PlaceModifier} == {"PLACE_FLOOR", "PLACE_NEXT_TO", "PLACE_IN_TRAY"}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _make_command_envelope() -> CommandEnvelope:
    return CommandEnvelope(
        command_id="cmd-1",
        arm_id="arm0",
        issued_at_ms=1000,
        type=CommandType.START_SKILL,
        payload=StartSkillPayload(
            skill_name=SkillName.PICK,
            target_handle="cube-1",
        ),
        precondition_snapshot_id="snap-arm0-1",
    )


def test_command_envelope_frozen():
    cmd = _make_command_envelope()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cmd.arm_id = "arm1"  # type: ignore[misc]


def test_command_ack_optional_reason():
    ack = CommandAck(command_id="cmd-1", status=CommandAckStatus.ACCEPTED)
    assert ack.reason is None

    ack_with_reason = CommandAck(
        command_id="cmd-2",
        status=CommandAckStatus.REJECTED_STALE,
        reason="snapshot too old",
    )
    assert ack_with_reason.reason == "snapshot too old"


def test_all_payload_types_instantiate():
    StartSkillPayload(skill_name=SkillName.PICK, target_handle="cube-1")
    AbortSkillPayload(skill_run_id="run-1", reason="operator abort")
    OverrideTargetPayload(skill_run_id="run-1", target_handle="cube-2")
    DescribeScenePayload(reason="lost target")


def test_command_asdict():
    cmd = _make_command_envelope()
    d = dataclasses.asdict(cmd)
    assert d["command_id"] == "cmd-1"
    assert d["payload"]["target_handle"] == "cube-1"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _make_event() -> EventEnvelope:
    return EventEnvelope(
        event_id="evt-1",
        type=EventType.PHASE_ENTER,
        ts_ms=2000,
        arm_id="arm0",
        data={"phase": "SELECT_GRASP"},
    )


def test_event_envelope_frozen():
    evt = _make_event()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        evt.arm_id = "arm1"  # type: ignore[misc]


def test_event_type_values():
    assert EventType.COMMAND_ACCEPTED == "COMMAND_ACCEPTED"
    assert EventType.SAFETY_REFLEX_TRIGGERED == "SAFETY_REFLEX_TRIGGERED"


def test_event_asdict():
    evt = _make_event()
    d = dataclasses.asdict(evt)
    assert d["type"] == "PHASE_ENTER"
    assert d["data"]["phase"] == "SELECT_GRASP"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _make_snapshot() -> PlannerSnapshot:
    return PlannerSnapshot(
        snapshot_id="snap-arm0-1",
        ts_ms=3000,
        arm_id="arm0",
        skill=SkillInfo(
            name=SkillName.PICK,
            skill_run_id="run-1",
            phase=PhaseId.SELECT_GRASP,
        ),
        target=TargetInfo(
            handle="cube-1",
            hint_valid=True,
            confidence=0.9,
            obs_age_ms=20,
            time_skew_ms=-3,
            delta_xyz_ee=(0.01, -0.02, 0.05),
            distance_m=0.12,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=220, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=500, no_progress_ms=0, delta_distance=-0.01),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.IN_PROGRESS,
            reason_code=None,
            needs_verify=False,
        ),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(_make_event(),),
    )


def test_snapshot_frozen():
    snap = _make_snapshot()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        snap.arm_id = "arm1"  # type: ignore[misc]


def test_snapshot_none_skill_and_target():
    snap = _make_snapshot()
    snap2 = dataclasses.replace(snap, skill=None, target=None)
    assert snap2.skill is None
    assert snap2.target is None


def test_snapshot_asdict():
    snap = _make_snapshot()
    d = dataclasses.asdict(snap)
    assert d["snapshot_id"] == "snap-arm0-1"
    assert d["skill"]["phase"] == PhaseId.SELECT_GRASP
    assert d["target"]["delta_xyz_ee"] == (0.01, -0.02, 0.05)
    assert d["safety"]["reason_codes"] == ()
    assert d["held_object_handle"] is None
