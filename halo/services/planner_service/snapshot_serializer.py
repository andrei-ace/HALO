from __future__ import annotations

from halo.contracts.snapshots import PlannerSnapshot


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

    # Outcome is only meaningful when a skill is active.
    # When skill is null, suppress outcome to avoid confusing the LLM.
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
        {"event_id": ev.event_id, "type": ev.type.value, "data": ev.data} for ev in snap.recent_events
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
