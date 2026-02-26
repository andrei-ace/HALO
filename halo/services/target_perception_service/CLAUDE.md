# TargetPerceptionService

Fast-loop (10 Hz) perception service for target tracking and async VLM scene analysis. Publishes target hints and perception status to RuntimeStateStore. The fast loop never blocks on VLM.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `TargetPerceptionServiceConfig` — loop rate, plausibility thresholds, buffer cap |
| `service.py` | `TargetPerceptionService` — tick loop, VLM orchestration, frame-buffer replay, tracker switchover, state transitions, command listener |
| `frame_buffer.py` | `CapturedFrame`, `FrameRingBuffer` — append-only buffer with read-cursor for replay |
| `vlm_parser.py` | `VlmDetection`, `VlmScene`, `parse_vlm_response()` — structured VLM output |
| `ollama_vlm_fn.py` | `make_ollama_vlm_fn()` — factory for async Ollama VLM callable (`qwen2.5vl`); accepts live camera frames |
| `video_capture_fn.py` | `make_video_capture_fn()` — factory for `CaptureFn` backed by a looping video file (OpenCV) |
| `mock_fns.py` | `make_mock_capture_fn()`, `make_mock_tracker_factory_fn()` — test factories |

## Key Types

```python
ObserveFn          = Callable[[str, str], Awaitable[TargetInfo | None]]          # (arm_id, handle)
VlmFn              = Callable[[str, object], Awaitable[VlmScene]]               # (arm_id, image)
CaptureFn          = Callable[[str], Awaitable[CapturedFrame]]                   # (arm_id) — fast frame grab
TrackerUpdateFn    = Callable[[CapturedFrame], Awaitable[TargetInfo | None]]      # feed frame to tracker
TrackerFactoryFn   = Callable[[CapturedFrame, VlmDetection], Awaitable[tuple[TargetInfo, TrackerUpdateFn]]]

TargetPerceptionService(arm_id, runtime, observe_fn=None, vlm_fn=None,
                        capture_fn=None, tracker_factory_fn=None, config=...)
```

## tick() Logic

1. No target handle → publish LOST, return
2. **Frame capture**: if `capture_fn` is set and replay buffer is active, capture a frame and push it
3. **Obtain observation** from the active tracking source:
   - If `_active_tracker_fn` (post-switchover): feed captured frame to it
   - Elif `observe_fn` (pre-switchover): call it; consume VLM seed if observe=None
   - Else: VLM-only mode — re-publish cached observation
4. If observe=None → increment reacquire counter; if limit hit → REACQUIRE_FAILED + spawn VLM
5. **Plausibility gates** on valid observation:
   - `obs_age_ms > obs_age_limit_ms` → hint_valid=False, DEPTH_INVALID
   - `|time_skew_ms| > time_skew_limit_ms` → hint_valid=False, CALIB_INVALID
6. Update store (target + perception info)
7. Emit events on state transitions (once per transition, not every tick)

## Frame-Buffer Replay & Tracker Switchover

When VLM runs for a new target (`for_new_target=True`), frames are buffered:

**During VLM inference:**
- The existing tracker (`observe_fn` or `_active_tracker_fn`) keeps running uninterrupted
- Every tick also captures a frame into the replay buffer via `capture_fn`

**When VLM completes:**
- `_replay_and_init_tracker(detection)` runs in the background task
- Initialises a new tracker on the first buffered frame using `tracker_factory_fn`
- Replays remaining frames sequentially; new frames keep arriving during replay
- Once caught up, sets `_active_tracker_fn` — tick() switches to the new tracker

**Switchover:**
- Post-switchover: tick() captures a fresh frame and feeds it to `_active_tracker_fn` each tick
- `set_tracking_target()` resets `_active_tracker_fn = None`, reverting to `observe_fn`

**Fallback:** If `capture_fn` or `tracker_factory_fn` is not provided, the original synthetic-seed behaviour is used (full backward compatibility).

## VLM Image Pipeline

The VLM receives a live camera frame on each invocation:
1. `_run_vlm()` calls `capture_fn(arm_id)` to grab the latest frame
2. The frame's `image` field (numpy BGR HWC, PIL, or bytes) is passed to `vlm_fn(arm_id, image)`
3. `ollama_vlm_fn` converts to base64 PNG, resized to 1024px width, then sends to Ollama
4. `video_capture_fn` provides frames from a looping video file (`data/video.mp4`) for dev/testing

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
| `frame_buffer_max_size` | 300 | Safety cap: max frames buffered during VLM inference |

## Integration

- **Writes**: `TargetInfo` + `PerceptionInfo` to RuntimeStateStore every tick
- **Publishes**: `PERCEPTION_FAILURE`, `PERCEPTION_RECOVERED`, `TARGET_ACQUIRED`, `SCENE_DESCRIBED`
- **Subscribes to**: `COMMAND_ACCEPTED` events (DESCRIBE_SCENE, TRACK_OBJECT)
- **Consumed by**: SkillRunnerService reads target hints directly from store; PlannerService sees perception status in snapshot

## Testing

`tick()` is directly callable. Use `make_mock_capture_fn()` and `make_mock_tracker_factory_fn()` for tests without Ollama. Tests verify: state transitions, plausibility gates, VLM async (never blocks tick), frame-buffer replay and tracker switchover, event emission (once per transition), command handling, lifecycle cleanup.
