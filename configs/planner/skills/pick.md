---
name: PICK
description: Pick up a tracked target object. Start when the arm is idle and the target is visible; abort on repeated perception failure or safety fault.
version: 1.0.0
---

## Goal

Move the arm to the target object and grasp it. The SkillRunner handles all
motion phases (select grasp, plan approach, move to pregrasp, visual align,
execute approach, close gripper, verify, lift) automatically — you only decide
when to start, when to give up, and how to recover.

## When to start

- No skill is currently running (`skill` is null).
- The target is being tracked (`perception.tracking_status` == "TRACKING").
- No safety fault is active.

```
start_skill(skill_name="PICK", target_handle=<target.handle>)
```

## When to abort

Stop PICK early if the situation is clearly unrecoverable this attempt:

- Perception has given up reacquiring the target (`reacquire_fail_count >= 3`).
- A safety fault has triggered and the arm must hold still.

For transient perception hiccups (RELOCALIZING, single REACQUIRE_FAILED), let
the skill continue — the SkillRunner and perception subsystem will resolve them.

## Recovery after failure

Choose the recovery action based on `outcome.reason_code`:

| reason_code | What happened | Recommended action |
|---|---|---|
| `NO_GRASP` | Arm reached the object but failed to grip it | Retry PICK |
| `NO_PROGRESS` | Arm stopped moving toward the target | Retry PICK |
| `TIMEOUT` | Skill ran too long without completing | Request perception refresh, then retry on next tick |
| `UNSAFE_ABORT` | Safety system intervened | Wait for safety to clear (no-op) |

If the same failure repeats 3 times, stop retrying and wait for operator.
