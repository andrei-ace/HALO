# Contracts

Canonical type definitions, enums, and serialization for the HALO runtime. Defines the contracts between all services and ensures type safety across the system.

## Files

| File | Purpose |
|------|---------|
| `enums.py` | Stable enums: `PhaseId`, `SkillName`, `TrackingStatus`, failure codes, safety reasons, command/event types |
| `snapshots.py` | Frozen dataclasses: `PlannerSnapshot` and its components (`SkillInfo`, `TargetInfo`, `PerceptionInfo`, `ActInfo`, `ProgressInfo`, `OutcomeInfo`, `SafetyInfo`, `QueuedSkillInfo`) |
| `commands.py` | `CommandEnvelope`, payload dataclasses (`StartSkillPayload`, `AbortSkillPayload`, `OverrideTargetPayload`, `DescribeScenePayload`), `CommandAck` |
| `events.py` | `EventType` enum, `EventEnvelope` dataclass |
| `actions.py` | `Action` (7D EE-frame deltas), `ActionChunk`, `JointPositionAction` (6D SO-101), `JointPositionChunk`, `ZERO_ACTION`, `ZERO_JOINT_ACTION` |
| `serde.py` | Round-trip serialization: `snapshot_to_dict/from_dict`, `snapshot_to_text`, `command_envelope_to/from_dict`, `vlm_scene_to/from_dict`, `context_entry_to/from_dict`, `cognitive_state_to/from_dict`, `message_record_to/from_dict` |
| `enums.json` | JSON Schema for all enum values |
| `commands.json` | JSON Schema for command contracts |
| `events.json` | JSON Schema for event contracts |
| `snapshot.json` | JSON Schema for PlannerSnapshot |

## PhaseId Values

| Phase | ID | Skill |
|-------|----|-------|
| IDLE | 0 | — |
| SELECT_GRASP | 1 | PICK |
| PLAN_APPROACH | 2 | PICK |
| MOVE_PREGRASP | 3 | PICK |
| VISUAL_ALIGN | 4 | PICK |
| EXECUTE_APPROACH | 5 | PICK |
| CLOSE_GRIPPER | 6 | PICK |
| LIFT | 7 | PICK |
| VERIFY_GRASP | 8 | PICK |
| DONE | 9 | all |
| ACQUIRING | 20 | TRACK |
| TRANSIT_PREPLACE | 30 | PLACE |
| DESCEND_PLACE | 31 | PLACE |
| OPEN | 32 | PLACE |
| RETREAT | 33 | PLACE |
| SELECT_PLACE | 34 | PLACE |
| RECOVER_RETRY_APPROACH | 50 | PICK/PLACE |
| RECOVER_REGRASP | 51 | PICK |
| RECOVER_ABORT | 52 | PICK |
| RETURNING | 60 | PICK/PLACE |

## PlannerSnapshot Composition

```
PlannerSnapshot
├── snapshot_id, ts_ms, arm_id
├── skill: SkillInfo | None (name, skill_run_id, phase)
├── target: TargetInfo | None (handle, hint_valid, confidence, distance_m, delta_xyz_ee, center_px, bbox_xywh, obs_age_ms, time_skew_ms)
├── perception: PerceptionInfo (tracking_status, failure_code, reacquire_fail_count, vlm_job_pending, buffer counts, has_pending_tracker)
├── act: ActInfo (status, buffer_fill_ms, buffer_low, wrist_enabled)
├── progress: ProgressInfo (elapsed_ms, no_progress_ms, delta_distance)
├── outcome: OutcomeInfo (state, reason_code, needs_verify)
├── safety: SafetyInfo (state, reflex_active, reason_codes)
├── command_acks: tuple[CommandAck, ...] (last 10)
├── recent_events: tuple[EventEnvelope, ...] (last 8)
├── held_object_handle: str | None
└── queued_skills: tuple[QueuedSkillInfo, ...] (skills waiting in queue)
```

All coordinates (center_px, bbox_xywh) are normalised 0..1. Snapshot is **replaced** (never appended) in planner context.

## Action Spaces

- **HALO core (runtime/bridge):** `Action` — `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]` — 7D EE-frame deltas per timestep
- **MuJoCo sim:** `JointPositionAction` — `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` — 6D joint-position targets

## Serde Round-Trip Pattern

All serde functions are lossless: `snapshot_from_dict(snapshot_to_dict(snap))` reproduces the original. Enums serialize to string values, dataclasses to plain dicts. `_NON_SERIALIZABLE_KEYS` excludes transient fields (e.g. `vlm_image`).

## Key Invariants

1. All dataclasses are **frozen** (`frozen=True`)
2. Commands use UUID `command_id` for idempotency
3. Commands carry optional `precondition_snapshot_id` for stale-state rejection
4. Commands carry optional `epoch` + `lease_token` for split-brain prevention
5. `WRIST_ACTIVE_PHASES` frozenset defines which phases get real wrist camera frames
