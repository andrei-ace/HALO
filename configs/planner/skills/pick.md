## PICK Skill

### When to start

- `skill` is null (arm is idle)
- `perception.tracking_status` == "TRACKING"
- `target.hint_valid` == true
- `safety.state` == "OK" and `safety.reflex_active` == false

Call `start_skill(skill_name="PICK", target_handle=<target.handle>)`.

### When to abort

Abort the PICK skill (call `abort_skill`) if any of the following are true:

- `perception.failure_code` == "REACQUIRE_FAILED" — target permanently lost
- `progress.no_progress_ms` > 8000 — arm is stuck (timeout)
- `safety.reflex_active` == true — safety override in progress

### Recovery hints

After aborting, decide based on `outcome.reason_code`:

| reason_code | Recommended action |
|---|---|
| `NO_GRASP` | Start PICK again with `options={"recovery": "RECOVER_REGRASP"}` |
| `NO_PROGRESS` | Start PICK again with `options={"recovery": "RECOVER_RETRY_DESCEND"}` |
| `TIMEOUT` | Request perception refresh, then retry PICK on next tick |
| `UNSAFE_ABORT` | Wait for safety state to clear (no-op this tick) |

### Phase progression (for context only — not controlled by planner)

The SkillRunnerService drives FSM transitions automatically:

`RESET → APPROACH_PREGRASP → ALIGN → DESCEND_GRASP → CLOSE → LIFT → SUCCESS_PICK`

Recovery states: `RECOVER_RETRY_APPROACH (20)`, `RECOVER_RETRY_DESCEND (21)`, `RECOVER_REGRASP (22)`

`GRASP_CLOSE` is triggered deterministically by distance + persistence threshold,
never by the planner.
