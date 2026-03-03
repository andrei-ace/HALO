"""Round-trip serialization tests for halo.contracts.serde."""

from __future__ import annotations

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandEnvelope,
    DescribeScenePayload,
    OverrideTargetPayload,
    StartSkillPayload,
    TrackObjectPayload,
)
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
from halo.contracts.serde import (
    command_envelope_from_dict,
    command_envelope_to_dict,
    snapshot_from_dict,
    snapshot_to_dict,
    vlm_scene_from_dict,
    vlm_scene_to_dict,
)
from halo.contracts.snapshots import (
    ActInfo,
    CommandAck,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    SkillInfo,
    TargetInfo,
)
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene

# ---------------------------------------------------------------------------
# Snapshot round-trip
# ---------------------------------------------------------------------------


def _full_snapshot() -> PlannerSnapshot:
    """Snapshot with all fields populated (including optional ones)."""
    return PlannerSnapshot(
        snapshot_id="snap-42",
        ts_ms=123456,
        arm_id="arm0",
        skill=SkillInfo(name=SkillName.PICK, skill_run_id="run-1", phase=PhaseId.EXECUTE_APPROACH),
        target=TargetInfo(
            handle="red_cube_01",
            hint_valid=True,
            confidence=0.95,
            obs_age_ms=20,
            time_skew_ms=5,
            delta_xyz_ee=(0.01, -0.02, 0.03),
            distance_m=0.15,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.TRACKING,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=True,
        ),
        act=ActInfo(status=ActStatus.RUNNING, buffer_fill_ms=200, buffer_low=False, wrist_enabled=True),
        progress=ProgressInfo(elapsed_ms=5000, no_progress_ms=100, delta_distance=-0.02),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(
            state=SafetyState.OK,
            reflex_active=False,
            reason_codes=(),
        ),
        command_acks=(CommandAck(command_id="cmd-1", status=CommandAckStatus.ACCEPTED),),
        recent_events=(
            EventEnvelope(
                event_id="ev-1",
                type=EventType.SKILL_STARTED,
                ts_ms=123000,
                arm_id="arm0",
                data={"skill_name": "PICK"},
            ),
        ),
        held_object_handle=None,
    )


def _idle_snapshot() -> PlannerSnapshot:
    """Snapshot with no skill active (outcome should be None in dict)."""
    return PlannerSnapshot(
        snapshot_id="snap-1",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=None,
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.IDLE,
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
    )


def _failed_snapshot() -> PlannerSnapshot:
    """Snapshot with a failure outcome and safety reflex."""
    return PlannerSnapshot(
        snapshot_id="snap-99",
        ts_ms=999999,
        arm_id="arm0",
        skill=SkillInfo(name=SkillName.PICK, skill_run_id="run-2", phase=PhaseId.RECOVER_ABORT),
        target=TargetInfo(
            handle="green_cube_01",
            hint_valid=False,
            confidence=0.1,
            obs_age_ms=500,
            time_skew_ms=200,
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.5,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
            failure_code=PerceptionFailureCode.REACQUIRE_FAILED,
            reacquire_fail_count=3,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.STALE, buffer_fill_ms=0, buffer_low=True),
        progress=ProgressInfo(elapsed_ms=30000, no_progress_ms=10000, delta_distance=0.0),
        outcome=OutcomeInfo(
            state=SkillOutcomeState.FAILURE, reason_code=SkillFailureCode.PERCEPTION_LOST, needs_verify=False
        ),
        safety=SafetyInfo(
            state=SafetyState.FAULT,
            reflex_active=True,
            reason_codes=(SafetyReflexReason.JOINT_LIMIT, SafetyReflexReason.WORKSPACE_LIMIT),
        ),
        command_acks=(),
        recent_events=(),
        held_object_handle="green_cube_01",
    )


def test_snapshot_roundtrip_full():
    snap = _full_snapshot()
    d = snapshot_to_dict(snap)
    restored = snapshot_from_dict(d)

    assert restored.snapshot_id == snap.snapshot_id
    assert restored.ts_ms == snap.ts_ms
    assert restored.arm_id == snap.arm_id
    assert restored.skill.name == snap.skill.name
    assert restored.skill.phase == snap.skill.phase
    assert restored.target.handle == snap.target.handle
    assert restored.target.confidence == snap.target.confidence
    assert restored.target.delta_xyz_ee == snap.target.delta_xyz_ee
    assert restored.perception.tracking_status == snap.perception.tracking_status
    assert restored.act.status == snap.act.status
    assert restored.act.wrist_enabled == snap.act.wrist_enabled
    assert restored.progress.elapsed_ms == snap.progress.elapsed_ms
    assert restored.outcome.state == snap.outcome.state
    assert restored.safety.state == snap.safety.state
    assert len(restored.command_acks) == 1
    assert restored.command_acks[0].status == CommandAckStatus.ACCEPTED
    assert len(restored.recent_events) == 1
    assert restored.recent_events[0].type == EventType.SKILL_STARTED


def test_snapshot_roundtrip_idle():
    snap = _idle_snapshot()
    d = snapshot_to_dict(snap)
    # outcome is None in dict when skill is None
    assert d["outcome"] is None
    restored = snapshot_from_dict(d)

    assert restored.skill is None
    assert restored.target is None
    # Outcome defaults to IN_PROGRESS when None in dict
    assert restored.outcome.state == SkillOutcomeState.IN_PROGRESS


def test_snapshot_roundtrip_failure():
    snap = _failed_snapshot()
    d = snapshot_to_dict(snap)
    restored = snapshot_from_dict(d)

    assert restored.outcome.reason_code == SkillFailureCode.PERCEPTION_LOST
    assert restored.safety.reflex_active is True
    assert len(restored.safety.reason_codes) == 2
    assert restored.held_object_handle == "green_cube_01"


# ---------------------------------------------------------------------------
# CommandEnvelope round-trip
# ---------------------------------------------------------------------------


def test_command_start_skill_roundtrip():
    cmd = CommandEnvelope(
        command_id="cmd-1",
        arm_id="arm0",
        issued_at_ms=1000,
        type=CommandType.START_SKILL,
        payload=StartSkillPayload(skill_name=SkillName.PICK, target_handle="red_cube_01", options={"timeout": 30}),
        precondition_snapshot_id="snap-1",
    )
    d = command_envelope_to_dict(cmd)
    restored = command_envelope_from_dict(d)

    assert restored.command_id == cmd.command_id
    assert restored.type == CommandType.START_SKILL
    assert restored.payload.skill_name == SkillName.PICK
    assert restored.payload.target_handle == "red_cube_01"
    assert restored.payload.options == {"timeout": 30}
    assert restored.precondition_snapshot_id == "snap-1"


def test_command_abort_skill_roundtrip():
    cmd = CommandEnvelope(
        command_id="cmd-2",
        arm_id="arm0",
        issued_at_ms=2000,
        type=CommandType.ABORT_SKILL,
        payload=AbortSkillPayload(skill_run_id="run-1", reason="operator abort"),
        precondition_snapshot_id="snap-2",
    )
    d = command_envelope_to_dict(cmd)
    restored = command_envelope_from_dict(d)

    assert restored.type == CommandType.ABORT_SKILL
    assert restored.payload.skill_run_id == "run-1"
    assert restored.payload.reason == "operator abort"


def test_command_override_target_roundtrip():
    cmd = CommandEnvelope(
        command_id="cmd-3",
        arm_id="arm0",
        issued_at_ms=3000,
        type=CommandType.OVERRIDE_TARGET,
        payload=OverrideTargetPayload(skill_run_id="run-1", target_handle="green_cube_01"),
    )
    d = command_envelope_to_dict(cmd)
    restored = command_envelope_from_dict(d)

    assert restored.type == CommandType.OVERRIDE_TARGET
    assert restored.payload.target_handle == "green_cube_01"


def test_command_describe_scene_roundtrip():
    cmd = CommandEnvelope(
        command_id="cmd-4",
        arm_id="arm0",
        issued_at_ms=4000,
        type=CommandType.DESCRIBE_SCENE,
        payload=DescribeScenePayload(reason="initial scan"),
        precondition_snapshot_id=None,
    )
    d = command_envelope_to_dict(cmd)
    restored = command_envelope_from_dict(d)

    assert restored.type == CommandType.DESCRIBE_SCENE
    assert restored.payload.reason == "initial scan"
    assert restored.precondition_snapshot_id is None


def test_command_track_object_roundtrip():
    cmd = CommandEnvelope(
        command_id="cmd-5",
        arm_id="arm0",
        issued_at_ms=5000,
        type=CommandType.TRACK_OBJECT,
        payload=TrackObjectPayload(target_handle="red_cube_01"),
    )
    d = command_envelope_to_dict(cmd)
    restored = command_envelope_from_dict(d)

    assert restored.type == CommandType.TRACK_OBJECT
    assert restored.payload.target_handle == "red_cube_01"


# ---------------------------------------------------------------------------
# VlmScene round-trip
# ---------------------------------------------------------------------------


def test_vlm_scene_roundtrip():
    scene = VlmScene(
        scene="A table with two cubes",
        detections=[
            VlmDetection(
                handle="red_cube_01",
                label="red cube",
                bbox=(0.1, 0.2, 0.3, 0.4),
                centroid=(0.2, 0.3),
                is_graspable=True,
            ),
            VlmDetection(
                handle="robot_hand_01",
                label="robot hand",
                bbox=(0.5, 0.5, 0.8, 0.9),
                centroid=(0.65, 0.7),
                is_graspable=False,
            ),
        ],
    )
    d = vlm_scene_to_dict(scene)
    restored = vlm_scene_from_dict(d)

    assert restored.scene == scene.scene
    assert len(restored.detections) == 2
    assert restored.detections[0].handle == "red_cube_01"
    assert restored.detections[0].is_graspable is True
    assert restored.detections[1].handle == "robot_hand_01"
    assert restored.detections[1].is_graspable is False
    assert restored.detections[0].bbox == (0.1, 0.2, 0.3, 0.4)
    assert restored.detections[0].centroid == (0.2, 0.3)


def test_vlm_scene_empty():
    scene = VlmScene(scene="", detections=[])
    d = vlm_scene_to_dict(scene)
    restored = vlm_scene_from_dict(d)
    assert restored.scene == ""
    assert restored.detections == []
