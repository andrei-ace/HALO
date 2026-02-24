# TargetPerceptionService

Fast-loop (10 Hz) perception service for target tracking and async VLM scene analysis. Publishes target hints and perception status to RuntimeStateStore. The fast loop never blocks on VLM.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `TargetPerceptionServiceConfig` — loop rate, plausibility thresholds |
| `service.py` | `TargetPerceptionService` — tick loop, VLM orchestration, state transitions, command listener |
| `vlm_parser.py` | `VlmDetection`, `VlmScene`, `parse_vlm_response()` — structured VLM output |
| `ollama_vlm_fn.py` | `make_ollama_vlm_fn()` — factory for async Ollama VLM callable (`qwen2.5vl`) |
| `mock_fns.py` | `make_mock_observe_fn()`, `make_mock_vlm_fn()` — test factories (no Ollama needed) |

## Key Types

```python
ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]  # (arm_id, target_handle)
VlmFn    = Callable[[str], Awaitable[VlmScene]]                 # (arm_id)

TargetPerceptionService(arm_id, runtime, observe_fn=None, vlm_fn=None, config=...)
```

## tick() Logic

1. No target handle → publish LOST, return
2. Call `observe_fn(arm_id, target_handle)`
3. If observe=None and VLM seed pending → use seed as observation
4. If observe=None → increment reacquire counter; if limit hit → REACQUIRE_FAILED + spawn VLM
5. **Plausibility gates** on valid observation:
   - `obs_age_ms > obs_age_limit_ms` → hint_valid=False, DEPTH_INVALID
   - `|time_skew_ms| > time_skew_limit_ms` → hint_valid=False, CALIB_INVALID
6. Update store (target + perception info)
7. Emit events on state transitions (once per transition, not every tick)

## State Transitions & Events

| Transition | Event |
|------------|-------|
| OK → non-OK failure code | `PERCEPTION_FAILURE` (once) |
| Non-OK → OK | `PERCEPTION_RECOVERED` (once) |
| First TRACKING after `set_tracking_target()` | `TARGET_ACQUIRED` (once) |
| VLM completes (via `request_refresh`) | `SCENE_DESCRIBED` |

## VLM Async Pipeline

- At most one VLM task at a time (duplicates dropped)
- Result stored as `_vlm_seed`; consumed by `tick()` when `observe_fn` returns None
- Triggered by: `set_tracking_target()`, `request_refresh()`, reacquire fail limit
- `ollama_vlm_fn`: image resized to 1024px width; prompt from `configs/perception/scene_analysis.md`
- Robust JSON extraction handles bare JSON, fenced blocks, or embedded JSON in prose

## Command Listener

Listens on EventBus for `COMMAND_ACCEPTED` events:

| Command | Action |
|---------|--------|
| `DESCRIBE_SCENE` | `request_refresh()` → VLM → `SCENE_DESCRIBED` |
| `TRACK_OBJECT` | `set_tracking_target(target_handle)` |

## Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `fast_loop_hz` | 10.0 | Hint publish rate |
| `obs_age_limit_ms` | 150 | Gate: invalidate if obs too old |
| `time_skew_limit_ms` | 50 | Gate: invalidate on clock skew |
| `reacquire_fail_limit` | 3 | Consecutive observe=None before REACQUIRE_FAILED |

## Integration

- **Writes**: `TargetInfo` + `PerceptionInfo` to RuntimeStateStore every tick
- **Publishes**: `PERCEPTION_FAILURE`, `PERCEPTION_RECOVERED`, `TARGET_ACQUIRED`, `SCENE_DESCRIBED`
- **Subscribes to**: `COMMAND_ACCEPTED` events (DESCRIBE_SCENE, TRACK_OBJECT)
- **Consumed by**: SkillRunnerService reads target hints directly from store; PlannerService sees perception status in snapshot

## Testing

`tick()` is directly callable. Use `make_mock_observe_fn()` and `make_mock_vlm_fn()` for tests without Ollama. Tests verify: state transitions, plausibility gates, VLM async (never blocks tick), event emission (once per transition), command handling, lifecycle cleanup.
