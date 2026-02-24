# SkillRunnerService

10–20 Hz service that owns the Pick skill FSM, schedules ACT action chunks, and publishes phase/skill lifecycle events.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `SkillRunnerConfig` — FSM thresholds, timeouts, buffer targets |
| `fsm.py` | `PickFSM` — pure synchronous state machine (no asyncio, no I/O) |
| `service.py` | `SkillRunnerService` — async orchestrator, event publisher, chunk scheduler |

## Key Types

```python
ChunkFn = Callable[[str, PhaseId, PlannerSnapshot], Awaitable[ActionChunk | None]]
PushFn  = Callable[[ActionChunk], Awaitable[None]]

SkillRunnerService(arm_id, runtime, chunk_fn, push_fn, config=SkillRunnerConfig())
```

## PickFSM (pure, deterministic)

Zero side effects — all transitions are synchronous. Designed for isolated unit testing.

### Phase Flow

```
APPROACH_PREGRASP → ALIGN → DESCEND_GRASP → CLOSE → [VERIFY_GRASP] → LIFT → DONE
                                                                        ↑
Recovery: RECOVER_RETRY_APPROACH / RECOVER_RETRY_DESCEND ───────────────┘
```

### advance() Check Order (per tick)

1. **Timeout** → fail (NO_PROGRESS / NO_GRASP)
2. **Target loss** (no_target_tolerance_ms exceeded) → recovery phase
3. **Forward progress** (distance thresholds) → next phase

Returns old phase on transition, None if stayed. **One transition per tick maximum.**

### GRASP_CLOSE Trigger

**Deterministic, never planner-driven.** Requires `distance < grasp_distance_threshold_m` held for `grasp_persistence_ms`. Distance bounce resets the timer.

### Target-Loss Recovery

- Any approach/align/descend phase can enter RECOVER_RETRY_*
- Recovery waits `recover_wait_ms`, increments reacquire counter
- After `max_reacquire_attempts` → fail with TIMEOUT

## Service tick() Sequence

1. Fetch snapshot (cached via store, fallback to full build)
2. `fsm.advance()` with current target/perception/act
3. On transition: publish `PHASE_EXIT` → update store → publish `PHASE_ENTER`
4. On DONE: publish `SKILL_SUCCEEDED` or `SKILL_FAILED`
5. Update progress (elapsed_ms, delta_distance)
6. If active and `needs_chunk()`: call `chunk_fn` → `push_fn`

## Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `runner_rate_hz` | 10.0 | FSM tick frequency |
| `approach_align_threshold_m` | 0.15 | APPROACH → ALIGN |
| `descend_threshold_m` | 0.05 | ALIGN → DESCEND |
| `grasp_distance_threshold_m` | 0.01 | Grasp trigger distance |
| `grasp_persistence_ms` | 300 | Grasp stability requirement |
| `approach_timeout_ms` | 10000 | 10 s max approach |
| `align_timeout_ms` | 5000 | |
| `descend_timeout_ms` | 5000 | |
| `close_duration_ms` | 1000 | Timer-based |
| `verify_duration_ms` | 500 | Timer-based |
| `lift_duration_ms` | 2000 | Timer-based |
| `no_target_tolerance_ms` | 2000 | Target loss patience |
| `max_reacquire_attempts` | 3 | Before TIMEOUT |
| `buffer_target_ms` | 200 | Chunk request threshold |
| `chunk_horizon_steps` | 10 | v0: 10 steps @ 10 Hz = 1 s |
| `skip_verify_grasp` | False | Skip VERIFY_GRASP phase |

## Events Published

| Event | Trigger |
|-------|---------|
| `SKILL_STARTED` | `start_skill()` |
| `SKILL_SUCCEEDED` | FSM → DONE with SUCCESS |
| `SKILL_FAILED` | FSM → DONE with FAILURE |
| `PHASE_ENTER` | Every phase transition (incl. start) |
| `PHASE_EXIT` | Every phase transition |

## Integration

- **Reads**: target, perception, act from RuntimeStateStore (via snapshot)
- **Writes**: skill info, outcome, progress to store
- **Publishes**: SKILL_STARTED/SUCCEEDED/FAILED, PHASE_ENTER/EXIT
- **Consumed by**: ControlService (PHASE_ENTER → buffer trim), PlannerService (SKILL_SUCCEEDED/FAILED → replan)

## Testing

PickFSM is pure — test all transitions with synthetic `TargetInfo`/`ActInfo`. Service tests use `HALORuntime` fixture + `_seed_store()` helper.
