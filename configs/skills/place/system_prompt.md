---
name: PLACE
description: Place a held object at a target location. Start after a successful PICK.
version: 2.0.0
---

## Goal

Carry the grasped object to a place target and release it. The SkillRunner
handles all motion phases automatically.

## Modifiers

| Modifier | Description | target_handle |
|---|---|---|
| `PLACE_FLOOR` | Place beside the arm in a free spot | Held object handle |
| `PLACE_NEXT_TO` | Place next to a tracked reference object | Reference object handle |

Pass the modifier via options: `start_skill(PLACE, target, options='{"modifier": "PLACE_FLOOR"}')`.

## FSM phases

```
SELECT_PLACE → TRANSIT_PREPLACE → DESCEND_PLACE → OPEN → RETREAT → DONE
```

Recovery: `RECOVER_RETRY_APPROACH` loops back to `TRANSIT_PREPLACE`.

## When to start

- PICK just succeeded (`outcome.state` == "SUCCESS").
- `held_object_handle` is set (arm is holding an object).
- A place target has been identified and is reachable.
- No safety fault is active.

```
start_skill(skill_name="PLACE", target_handle=<target>, options='{"modifier": "..."}')
```

## When to abort

- The object was dropped in transit (`outcome.reason_code` == "DROP_DETECTED").
- The place target is unreachable (`outcome.reason_code` == "PLACE_MISS") after one retry.
- A safety fault is active.

## Recovery after failure

| reason_code | What happened | Recommended action |
|---|---|---|
| `DROP_DETECTED` | Object lost during transit | Request perception refresh, then restart PICK |
| `PLACE_MISS` | Could not reach place target | Retry PLACE once; if it fails again, wait for operator |
| `PERCEPTION_LOST` | Lost tracking of reference | Retry TRACK then PLACE |
| `TIMEOUT` | Recovery retries exhausted | Wait for operator guidance |
