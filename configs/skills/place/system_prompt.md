---
name: PLACE
description: Place a held object at a target location.
---

## Prerequisites
- PICK just succeeded, `held_object_handle` is set, no safety fault.
- **The place target must be tracked first.** Run `start_skill(TRACK, target_handle)` and wait for SKILL_SUCCEEDED before starting PLACE. The target to track is the destination (the reference object, tray, etc.), NOT the held object.

## Start
```
start_skill(skill_name="PLACE", target_handle=<handle>, options='{"modifier": "<MODIFIER>"}')
```

## Modifiers

| Modifier | target_handle | Use case |
|---|---|---|
| `PLACE_NEXT_TO` | Reference object handle | Place next to another object |
| `PLACE_IN_TRAY` | Tray handle | Place into a container |
| `PLACE_FLOOR` | Held object handle | Place on table (no tracking needed) |

## Recovery

| reason_code | Action |
|---|---|
| `DROP_DETECTED` | describe_scene, re-PICK, then PLACE again |
| `PLACE_MISS` | Retry PLACE once, then wait for operator |
| `PERCEPTION_LOST` | TRACK target, then retry PLACE |
| `TIMEOUT` | Wait for operator |
