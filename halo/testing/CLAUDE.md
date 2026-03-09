# Testing

Modular testing infrastructure for HALO. Provides service orchestration, event capture, state factories, mock functions, and post-run metrics.

## Files

| File | Purpose |
|------|---------|
| `runner.py` | `HeadlessRunner` + `RunnerConfig` — service orchestrator replacing the TUI for tests; auto-wires services, manages lifecycle, routes commands |
| `event_recorder.py` | `EventRecorder` + `RecordedEvent` — subscribes to EventBus, captures all events for post-hoc assertions; `wait_for_event()` async |
| `state_seeder.py` | `make_target()`, `make_perception()`, `make_act()`, `make_skill()`, `seed_store()` — factory functions for test state (all coordinates normalised 0..1) |
| `mock_fns.py` | `LatencyProfile` + mock factories: `make_mock_decide_fn()`, `make_mock_vlm_fn()`, `make_mock_capture_fn_with_latency()`, `make_mock_chunk_fn()`, `make_mock_apply_fn()`, `make_scripted_decide_fn()`, `make_mock_start_pick_fn()`, `make_mock_sim_phase_fn()`, `make_video_capture_fn()`, `make_mock_tracker_factory_fn_with_latency()`, `make_command()` |
| `metrics.py` | `RunReport`, `SkillMetrics`, `PhaseTimingEntry`, `SafetyMetrics`, `PerceptionMetrics`, `ControlMetrics`, `compute_run_report()` — post-run analysis from `EventRecorder` data |

## 4-Tier Testing Framework

| Tier | Scope | Speed | Example |
|------|-------|-------|---------|
| **Unit** | Single class/function, mocks only | Instant | FSM handler `evaluate()` with synthetic `StateContext` |
| **Component** | One service + `HALORuntime` + mocks | Fast | `SkillRunnerService` with mock `chunk_fn`/`push_fn` |
| **System** | Multiple services + `HeadlessRunner` | Medium | Planner → SkillRunner → ControlService with mock decide/apply |
| **E2E** | Full stack, optional real LLM | Slow | Integration tests with Ollama (`integration/`) |

## HeadlessRunner

Wires and manages service lifecycle. Constructor accepts enable flags and callable overrides per service.

**Auto-wiring:** if both `enable_skill_runner` and `enable_control` are true with no explicit `push_fn`, wires `push_fn → control_svc.push_chunk`.

**Sim detection:** if `start_pick_fn` is provided, switches to sim mode (no control service needed).

**Lifecycle:** `start()` boots services in order (recorder → control → skill_runner → perception → command routing → planner). `stop()` reverses. `run(until=predicate)` combines both with timeout.

## LatencyProfile

```python
LatencyProfile.instant()          # zero delays (unit tests)
LatencyProfile.realistic()        # LLM 2-5s, VLM 1-3s, hardware ~ms
LatencyProfile.fast_integration() # CI-friendly (~50-150ms decide)
```

## Common Test Patterns

```python
# Component test setup
rt = HALORuntime()
rt.register_arm("arm0")
await seed_store(rt, "arm0", target=make_target(distance_m=0.2), perception=make_perception(tracking_status=TrackingStatus.TRACKING))

# Event assertions
recorder = EventRecorder(rt.bus, "arm0")
await recorder.start()
# ... run test ...
assert recorder.events_of_type(EventType.SKILL_SUCCEEDED)
await recorder.stop()

# HeadlessRunner system test
runner = HeadlessRunner(RunnerConfig(arm_id="arm0", enable_planner=True, enable_skill_runner=True))
report = await runner.run(until=lambda: runner.recorder.events_of_type(EventType.SKILL_SUCCEEDED))
assert report.skills[0].outcome == "succeeded"
```
