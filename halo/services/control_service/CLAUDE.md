# ControlService

Real-time (50–100 Hz) action-streaming service. Never waits on LLM or VLM.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `ControlServiceConfig` dataclass — rate, limits, thresholds |
| `te_buffer.py` | `TemporalEnsemblingBuffer` — blends overlapping ACT chunks with exponential decay |
| `action_buffer.py` | `ActionBuffer` — legacy FIFO deque (kept, not used by default) |
| `safety_guard.py` | `SafetyGuard` — joint-limit checks + clamping |
| `service.py` | `ControlService` — asyncio loop, event drain, applies actions to robot |

## Key Types

```python
ApplyFn = Callable[[str, JointPositionAction], Awaitable[None]]   # (arm_id, action) → robot/sim

ControlService(arm_id, runtime, apply_fn, initial_state, config=ControlServiceConfig())
```

## tick() Order (critical path)

1. Get latest snapshot from runtime; derive `wrist_enabled` from phase
2. **Hint freshness check** — stale target (obs_age > max or hint_valid=False) → re-apply `_last_applied` (hold position), status=STALE, **no reflex**
3. Pop blended action from `TemporalEnsemblingBuffer` (under lock)
4. **Safety check** — absolute joint-limit violation → trigger reflex (once), re-apply `_last_applied` (hold position). Per-tick velocity violations are **not** faulted here — they are handled by clamp.
5. **Reflex recovery** — first clean tick after reflex → emit SAFETY_RECOVERED
6. **Clamp** via `SafetyGuard.clamp()` — enforces absolute limits + per-tick velocity limit (`max_joint_delta_rad` for arm, `max_gripper_delta` for gripper)
7. `apply_fn(arm_id, clamped_action)` — if `BridgeTransportError` raised, write STALE and return None
8. Update store with `ActInfo` (status, buffer_fill_ms, buffer_low, wrist_enabled)

### Bridge transport failure handling

All three `apply_fn` call sites (stale-hint hold, reflex hold, normal apply) catch `BridgeTransportError` from `halo.bridge`. On the normal apply path, a transport failure writes `ActStatus.STALE` to the store and returns `None` — the rest of the system sees that actuation is not happening. On the stale-hint and reflex paths, transport failures are logged but the existing status (STALE / reflex) already signals the problem.

## Temporal Ensembling

Blends overlapping chunks per-timestep: `w = exp(-temp * (current_tick - push_tick))`. Newer chunks weighted higher. `temp=0.0` gives uniform blending.

## Reflex Lifecycle

- **Trigger**: first tick with safety violation → publish `SAFETY_REFLEX_TRIGGERED`, set FAULT
- **Hold**: subsequent violation ticks re-apply `_last_applied` (hold position), no re-trigger
- **Recover**: first clean tick → publish `SAFETY_RECOVERED`, set OK

## Phase Transitions

Subscribes to `PHASE_ENTER` events from EventBus. On phase switch: trims TE buffer to `buffer_trim_ms` (~75 ms default) to discard stale tail actions from previous phase.

## Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `control_rate_hz` | 50.0 | Loop frequency |
| `buffer_low_threshold_ms` | 100 | Below this → `BUFFER_LOW` status |
| `buffer_trim_ms` | 75 | Trim on phase switch |
| `joint_limits_lower` | SO-101 defaults | Per-joint lower bounds (rad) |
| `joint_limits_upper` | SO-101 defaults | Per-joint upper bounds (rad) |
| `max_joint_delta_rad` | 0.5 | Per-timestep velocity limit |
| `max_gripper_delta` | 2.0 | Full open/close span (1.92 rad) in one tick |
| `max_obs_age_ms` | 200 | Observation freshness limit |
| `ensembling_temp` | 0.01 | Decay constant (0 = uniform) |

## Construction

`initial_state` is clamped to joint limits at construction via `SafetyGuard.clamp()`, so out-of-range hardware telemetry cannot leak into hold commands. `ControlServiceConfig.__post_init__` validates that `joint_limits_lower`/`upper` tuples have exactly `SO101_DOF` elements.

## Integration

- **Reads**: target hint freshness from RuntimeStateStore; `WRIST_ACTIVE_PHASES` from `contracts.enums`
- **Writes**: `ActInfo` (status, buffer_fill_ms, buffer_low, wrist_enabled), `SafetyInfo` (state, reflex_active)
- **Catches**: `BridgeTransportError` from `halo.bridge` (apply_fn failures → STALE status)
- **Publishes**: `SAFETY_REFLEX_TRIGGERED`, `SAFETY_RECOVERED`
- **Subscribes to**: `PHASE_ENTER` (for buffer trim)
- **Called by**: SkillRunnerService via `push_chunk()`

## Testing

`tick()` is directly callable in tests — no need to `start()` the loop. Tests mock `apply_fn` and seed the store with synthetic snapshots.
