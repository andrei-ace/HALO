"""Shared round-trip serializers for contract types.

Both the local codebase and the remote cloud_service import from here.
"""

from __future__ import annotations

from halo.cognitive.compactor import MessageRecord
from halo.cognitive.context_store import CognitiveState, ContextEntry
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
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

_NON_SERIALIZABLE_KEYS = {"vlm_image"}


def snapshot_to_dict(snap: PlannerSnapshot) -> dict:
    """Convert PlannerSnapshot to a plain Python dict (no dataclasses, no enum instances)."""
    skill_dict = None
    if snap.skill is not None:
        skill_dict = {
            "name": snap.skill.name.value,
            "skill_run_id": snap.skill.skill_run_id,
            "phase": snap.skill.phase.name,
        }

    target_dict = None
    if snap.target is not None:
        t = snap.target
        target_dict = {
            "handle": t.handle,
            "hint_valid": t.hint_valid,
            "confidence": t.confidence,
            "obs_age_ms": t.obs_age_ms,
            "time_skew_ms": t.time_skew_ms,
            "delta_xyz_ee": list(t.delta_xyz_ee),
            "distance_m": t.distance_m,
        }

    p = snap.perception
    perception_dict = {
        "tracking_status": p.tracking_status.value,
        "failure_code": p.failure_code.value,
        "reacquire_fail_count": p.reacquire_fail_count,
        "vlm_job_pending": p.vlm_job_pending,
    }

    a = snap.act
    act_dict = {
        "status": a.status.value,
        "buffer_fill_ms": a.buffer_fill_ms,
        "buffer_low": a.buffer_low,
        "wrist_enabled": a.wrist_enabled,
    }

    pr = snap.progress
    progress_dict = {
        "elapsed_ms": pr.elapsed_ms,
        "no_progress_ms": pr.no_progress_ms,
        "delta_distance": pr.delta_distance,
    }

    if snap.skill is not None:
        o = snap.outcome
        outcome_dict = {
            "state": o.state.value,
            "reason_code": o.reason_code.value if o.reason_code is not None else None,
            "needs_verify": o.needs_verify,
        }
    else:
        outcome_dict = None

    s = snap.safety
    safety_dict = {
        "state": s.state.value,
        "reflex_active": s.reflex_active,
        "reason_codes": [r.value for r in s.reason_codes],
    }

    command_acks_list = [{"command_id": ack.command_id, "status": ack.status.value} for ack in snap.command_acks]

    recent_events_list = [
        {
            "event_id": ev.event_id,
            "type": ev.type.value,
            "data": {k: v for k, v in ev.data.items() if k not in _NON_SERIALIZABLE_KEYS},
        }
        for ev in snap.recent_events
    ]

    return {
        "snapshot_id": snap.snapshot_id,
        "ts_ms": snap.ts_ms,
        "arm_id": snap.arm_id,
        "skill": skill_dict,
        "target": target_dict,
        "held_object_handle": snap.held_object_handle,
        "perception": perception_dict,
        "act": act_dict,
        "progress": progress_dict,
        "outcome": outcome_dict,
        "safety": safety_dict,
        "command_acks": command_acks_list,
        "recent_events": recent_events_list,
    }


def snapshot_to_text(d: dict) -> str:
    """Format a snapshot dict as operator-friendly text summary.

    Accepts the output of ``snapshot_to_dict()`` and returns a compact
    multi-line description suitable for conversational agents.
    Omits planner-internal details (run IDs, hint validity, confidence, distances).
    """
    lines: list[str] = []

    # Skill status
    skill = d.get("skill")
    if skill:
        skill_name = skill.get("name") or skill.get("skill_name", "unknown")
        phase = skill.get("phase", "unknown")
        lines.append(f"Skill: {skill_name}, phase: {phase}")
    else:
        lines.append("Idle.")

    # Target
    target = d.get("target") or {}
    handle = target.get("handle")
    if handle:
        lines.append(f"Target: {handle}")

    # Held object
    held = d.get("held_object_handle")
    if held:
        lines.append(f"Holding: {held}")

    # Outcome
    outcome = d.get("outcome") or {}
    state = outcome.get("state")
    reason = outcome.get("reason_code")
    if state and state != "IN_PROGRESS":
        line = f"Outcome: {state}"
        if reason:
            line += f" ({reason})"
        lines.append(line)

    # Safety — only surface problems
    safety = d.get("safety") or {}
    safety_state = safety.get("state", "OK")
    if safety_state != "OK":
        reasons = safety.get("reason_codes", [])
        detail = f": {', '.join(reasons)}" if reasons else ""
        lines.append(f"Safety: {safety_state}{detail}")

    return "\n".join(lines)


def snapshot_from_dict(d: dict) -> PlannerSnapshot:
    """Reconstruct a PlannerSnapshot from a plain dict (inverse of snapshot_to_dict)."""
    skill = None
    if d.get("skill") is not None:
        sk = d["skill"]
        skill = SkillInfo(
            name=SkillName(sk["name"]),
            skill_run_id=sk["skill_run_id"],
            phase=PhaseId[sk["phase"]],
        )

    target = None
    if d.get("target") is not None:
        t = d["target"]
        target = TargetInfo(
            handle=t["handle"],
            hint_valid=t["hint_valid"],
            confidence=t["confidence"],
            obs_age_ms=t["obs_age_ms"],
            time_skew_ms=t["time_skew_ms"],
            delta_xyz_ee=tuple(t["delta_xyz_ee"]),
            distance_m=t["distance_m"],
        )

    p = d["perception"]
    perception = PerceptionInfo(
        tracking_status=TrackingStatus(p["tracking_status"]),
        failure_code=PerceptionFailureCode(p["failure_code"]),
        reacquire_fail_count=p["reacquire_fail_count"],
        vlm_job_pending=p["vlm_job_pending"],
    )

    a = d["act"]
    act = ActInfo(
        status=ActStatus(a["status"]),
        buffer_fill_ms=a["buffer_fill_ms"],
        buffer_low=a["buffer_low"],
        wrist_enabled=a.get("wrist_enabled", False),
    )

    pr = d["progress"]
    progress = ProgressInfo(
        elapsed_ms=pr["elapsed_ms"],
        no_progress_ms=pr["no_progress_ms"],
        delta_distance=pr["delta_distance"],
    )

    outcome_dict = d.get("outcome")
    if outcome_dict is not None:
        outcome = OutcomeInfo(
            state=SkillOutcomeState(outcome_dict["state"]),
            reason_code=SkillFailureCode(outcome_dict["reason_code"]) if outcome_dict["reason_code"] else None,
            needs_verify=outcome_dict["needs_verify"],
        )
    else:
        outcome = OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False)

    s = d["safety"]
    safety = SafetyInfo(
        state=SafetyState(s["state"]),
        reflex_active=s["reflex_active"],
        reason_codes=tuple(SafetyReflexReason(r) for r in s["reason_codes"]),
    )

    command_acks = tuple(
        CommandAck(
            command_id=ack["command_id"],
            status=CommandAckStatus(ack["status"]),
        )
        for ack in d.get("command_acks", ())
    )

    recent_events = tuple(
        EventEnvelope(
            event_id=ev["event_id"],
            type=EventType(ev["type"]),
            ts_ms=ev.get("ts_ms", 0),
            arm_id=ev.get("arm_id", ""),
            data=ev.get("data", {}),
        )
        for ev in d.get("recent_events", ())
    )

    return PlannerSnapshot(
        snapshot_id=d["snapshot_id"],
        ts_ms=d["ts_ms"],
        arm_id=d["arm_id"],
        skill=skill,
        target=target,
        perception=perception,
        act=act,
        progress=progress,
        outcome=outcome,
        safety=safety,
        command_acks=command_acks,
        recent_events=recent_events,
        held_object_handle=d.get("held_object_handle"),
    )


# ---------------------------------------------------------------------------
# CommandEnvelope
# ---------------------------------------------------------------------------

_PAYLOAD_BUILDERS: dict[str, type] = {
    CommandType.START_SKILL: StartSkillPayload,
    CommandType.ABORT_SKILL: AbortSkillPayload,
    CommandType.OVERRIDE_TARGET: OverrideTargetPayload,
    CommandType.DESCRIBE_SCENE: DescribeScenePayload,
}


def command_envelope_to_dict(cmd: CommandEnvelope) -> dict:
    """Serialize a CommandEnvelope to a plain dict."""
    payload_dict: dict = {}
    p = cmd.payload
    if isinstance(p, StartSkillPayload):
        payload_dict = {"skill_name": p.skill_name.value, "target_handle": p.target_handle, "options": p.options}
    elif isinstance(p, AbortSkillPayload):
        payload_dict = {"skill_run_id": p.skill_run_id, "reason": p.reason}
    elif isinstance(p, OverrideTargetPayload):
        payload_dict = {"skill_run_id": p.skill_run_id, "target_handle": p.target_handle}
    elif isinstance(p, DescribeScenePayload):
        payload_dict = {"reason": p.reason}

    d = {
        "command_id": cmd.command_id,
        "arm_id": cmd.arm_id,
        "issued_at_ms": cmd.issued_at_ms,
        "type": cmd.type.value,
        "payload": payload_dict,
        "precondition_snapshot_id": cmd.precondition_snapshot_id,
    }
    if cmd.epoch is not None:
        d["epoch"] = cmd.epoch
    d["lease_token"] = cmd.lease_token
    return d


def command_envelope_from_dict(d: dict) -> CommandEnvelope:
    """Reconstruct a CommandEnvelope from a plain dict."""
    cmd_type = CommandType(d["type"])
    raw = d["payload"]

    if cmd_type == CommandType.START_SKILL:
        payload = StartSkillPayload(
            skill_name=SkillName(raw["skill_name"]),
            target_handle=raw["target_handle"],
            options=raw.get("options", {}),
        )
    elif cmd_type == CommandType.ABORT_SKILL:
        payload = AbortSkillPayload(skill_run_id=raw["skill_run_id"], reason=raw["reason"])
    elif cmd_type == CommandType.OVERRIDE_TARGET:
        payload = OverrideTargetPayload(skill_run_id=raw["skill_run_id"], target_handle=raw["target_handle"])
    elif cmd_type == CommandType.DESCRIBE_SCENE:
        payload = DescribeScenePayload(reason=raw["reason"])
    else:
        msg = f"Unknown command type: {cmd_type}"
        raise ValueError(msg)

    return CommandEnvelope(
        command_id=d["command_id"],
        arm_id=d["arm_id"],
        issued_at_ms=d["issued_at_ms"],
        type=cmd_type,
        payload=payload,
        precondition_snapshot_id=d.get("precondition_snapshot_id"),
        epoch=d.get("epoch"),
        lease_token=d.get("lease_token"),
    )


# ---------------------------------------------------------------------------
# VlmScene
# ---------------------------------------------------------------------------


def vlm_scene_to_dict(scene: VlmScene) -> dict:
    """Serialize a VlmScene to a plain dict."""
    return {
        "scene": scene.scene,
        "detections": [
            {
                "handle": det.handle,
                "label": det.label,
                "bbox": list(det.bbox),
                "centroid": list(det.centroid),
                "is_graspable": det.is_graspable,
            }
            for det in scene.detections
        ],
    }


def vlm_scene_from_dict(d: dict) -> VlmScene:
    """Reconstruct a VlmScene from a plain dict."""
    detections = [
        VlmDetection(
            handle=det["handle"],
            label=det["label"],
            bbox=tuple(det["bbox"]),
            centroid=tuple(det["centroid"]),
            is_graspable=det.get("is_graspable", True),
        )
        for det in d.get("detections", [])
    ]
    return VlmScene(scene=d.get("scene", ""), detections=detections)


# ---------------------------------------------------------------------------
# ContextEntry
# ---------------------------------------------------------------------------


def context_entry_to_dict(entry: ContextEntry) -> dict:
    """Serialize a ContextEntry to a plain dict."""
    return {
        "cursor": entry.cursor,
        "ts_ms": entry.ts_ms,
        "epoch": entry.epoch,
        "backend": entry.backend,
        "entry_type": entry.entry_type,
        "summary": entry.summary,
        "data": entry.data,
    }


def context_entry_from_dict(d: dict) -> ContextEntry:
    """Reconstruct a ContextEntry from a plain dict."""
    return ContextEntry(
        cursor=d["cursor"],
        ts_ms=d["ts_ms"],
        epoch=d["epoch"],
        backend=d["backend"],
        entry_type=d["entry_type"],
        summary=d["summary"],
        data=d.get("data", {}),
    )


# ---------------------------------------------------------------------------
# CognitiveState
# ---------------------------------------------------------------------------


def cognitive_state_to_dict(state: CognitiveState) -> dict:
    """Serialize a CognitiveState to a plain dict."""
    return {
        "ts_ms": state.ts_ms,
        "epoch": state.epoch,
        "cursor": state.cursor,
        "active_target_handle": state.active_target_handle,
        "held_object_handle": state.held_object_handle,
        "known_scene_handles": list(state.known_scene_handles),
        "last_scene_description": state.last_scene_description,
        "pending_operator_instruction": state.pending_operator_instruction,
        "recent_decisions": list(state.recent_decisions),
        "last_snapshot_id": state.last_snapshot_id,
        "last_arm_id": state.last_arm_id,
        "last_skill_phase": state.last_skill_phase,
        "last_skill_name": state.last_skill_name,
        "last_outcome_state": state.last_outcome_state,
        "recent_event_summaries": list(state.recent_event_summaries),
        "goal_summary": state.goal_summary,
    }


def cognitive_state_from_dict(d: dict) -> CognitiveState:
    """Reconstruct a CognitiveState from a plain dict."""
    return CognitiveState(
        ts_ms=d["ts_ms"],
        epoch=d["epoch"],
        cursor=d["cursor"],
        active_target_handle=d.get("active_target_handle"),
        held_object_handle=d.get("held_object_handle"),
        known_scene_handles=d.get("known_scene_handles", []),
        last_scene_description=d.get("last_scene_description", ""),
        pending_operator_instruction=d.get("pending_operator_instruction"),
        recent_decisions=d.get("recent_decisions", []),
        last_snapshot_id=d.get("last_snapshot_id"),
        last_arm_id=d.get("last_arm_id"),
        last_skill_phase=d.get("last_skill_phase"),
        last_skill_name=d.get("last_skill_name"),
        last_outcome_state=d.get("last_outcome_state"),
        recent_event_summaries=d.get("recent_event_summaries", []),
        goal_summary=d.get("goal_summary"),
    )


# ---------------------------------------------------------------------------
# MessageRecord
# ---------------------------------------------------------------------------


def message_record_to_dict(rec: MessageRecord) -> dict:
    """Serialize a MessageRecord to a plain dict."""
    return {
        "msg_id": rec.msg_id,
        "role": rec.role,
        "text": rec.text,
        "ts_ms": rec.ts_ms,
        "is_summary": rec.is_summary,
    }


def message_record_from_dict(d: dict) -> MessageRecord:
    """Reconstruct a MessageRecord from a plain dict."""
    return MessageRecord(
        msg_id=d["msg_id"],
        role=d["role"],
        text=d["text"],
        ts_ms=d["ts_ms"],
        is_summary=d.get("is_summary", False),
    )
