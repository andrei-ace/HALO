---
name: PICK
description: Pick up a tracked object.
---

## Prerequisites
- No skill running, no safety fault.
- **The pick target must be tracked first.** Run `start_skill(TRACK, target_handle)` and wait for SKILL_SUCCEEDED before starting PICK.

## Start
```
start_skill(skill_name="PICK", target_handle=<handle>)
```

## Recovery

| reason_code | Action |
|---|---|
| `NO_GRASP` / `NO_PROGRESS` | Retry PICK |
| `TIMEOUT` | describe_scene, then retry |
| `UNSAFE_ABORT` | Wait for safety clear |

3 same failures → stop, wait for operator.
