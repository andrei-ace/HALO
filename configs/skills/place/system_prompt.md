---
name: PLACE
description: Place a held object at a target location. Start after a successful PICK. Not yet available in v0 — do not issue this command.
version: 1.0.0
---

> **Not available in v0.** Do not call `start_skill(skill_name="PLACE", ...)`.
> If PICK succeeded, no-op this tick and wait for operator guidance.

## Goal

Carry the grasped object to a place target and release it. As with PICK, the
SkillRunner handles all motion phases automatically.

## When to start

- PICK just succeeded (`outcome.state` == "SUCCESS").
- A place target has been identified and is reachable.
- No safety fault is active.

```
start_skill(skill_name="PLACE", target_handle=<place_target_handle>)
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
