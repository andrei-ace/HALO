## PLACE Skill

### When to start

- The previous PICK succeeded: `outcome.state` == "SUCCESS"
- A place target has been identified (conveyed via `target.handle` or operator command)
- `safety.state` == "OK" and `safety.reflex_active` == false

Call `start_skill(skill_name="PLACE", target_handle=<place_target_handle>)`.

### When to abort

Abort the PLACE skill (call `abort_skill`) if:

- `outcome.reason_code` == "DROP_DETECTED" — object was dropped before placing
- `outcome.reason_code` == "PLACE_MISS" — end-effector could not reach place target
- `safety.reflex_active` == true

### Recovery hints

| reason_code | Recommended action |
|---|---|
| `DROP_DETECTED` | The object is lost; request perception refresh and restart PICK |
| `PLACE_MISS` | Retry PLACE once; if it fails again, wait for operator |

### Important note

**PLACE skill is not yet implemented in SkillRunnerService (v0).**
Do not issue `start_skill(skill_name="PLACE", ...)` in the current version.
If a PICK succeeded and a PLACE would normally follow, log a no-op and wait
for operator guidance.
