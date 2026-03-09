# Runtime

Core state management, event publication, and command validation. Top-level entry point for the HALO system.

## Files

| File | Purpose |
|------|---------|
| `runtime.py` | `HALORuntime` — top-level facade wiring store, bus, and router; exposes two planner-facing APIs |
| `state_store.py` | `RuntimeStateStore` — per-arm state, per-field update methods, snapshot assembly and caching |
| `event_bus.py` | `EventBus` — asyncio pub/sub, per-arm subscriber queues, ring buffer of 8 recent events |
| `command_router.py` | `CommandRouter` — four-tier command validation (idempotency → epoch → precondition → skill-run match) |

## Two Planner-Facing APIs

```python
runtime.get_latest_runtime_snapshot(arm_id) → PlannerSnapshot   # read
runtime.submit_command(cmd: CommandEnvelope) → CommandAck        # write
```

Services update state via `runtime.store` and `runtime.bus`; the planner only sees the snapshot.

## HALORuntime

- `__init__(lease_manager=None)` — creates store, bus, router; optional `LeaseManager` for split-brain prevention
- `register_arm(arm_id)` — initializes per-arm state
- `get_latest_runtime_snapshot()` — builds fresh snapshot from store + bus events, filters out internal events (PHASE_ENTER/EXIT, BACKEND_SWITCHED, SESSION_COMPACTED), caches result (replaces prior)
- `submit_command()` — routes to `CommandRouter.submit()`

## RuntimeStateStore

Per-arm state partitioned by `arm_id`. All updates async with `asyncio.Lock`.

**Per-field update methods:** `update_skill()`, `update_target()`, `update_held_object_handle()`, `update_perception()`, `update_target_and_perception()` (atomic), `update_act()`, `update_progress()`, `update_outcome()`, `update_safety()`, `add_command_ack()`

**Snapshot assembly:** `build_and_cache_snapshot(arm_id, recent_events)` — assembles from all fields, assigns monotonic `snapshot_id`, **replaces** (never appends) cached snapshot. Ring of 10 command acks, 8 recent events.

## EventBus

- `subscribe(arm_id, maxsize=100)` → `asyncio.Queue` — per-arm subscriber
- `publish(event)` — non-blocking; drops to full queues (publisher never stalled)
- `get_recent_events(arm_id, n=8)` — last n from ring buffer (oldest first)
- Event IDs generated monotonically: `evt-0`, `evt-1`, ...

## CommandRouter Validation Order

1. **Idempotency** — `command_id` in accepted set → `ALREADY_APPLIED`
2. **Epoch + Token** — if `LeaseManager` active, must match current epoch/token → `REJECTED_WRONG_EPOCH`
3. **Precondition** — `precondition_snapshot_id` must match latest → `REJECTED_STALE`
4. **Skill-run Match** — for ABORT_SKILL/OVERRIDE_TARGET, `skill_run_id` must match current → `REJECTED_WRONG_SKILL_RUN`

Publishes `COMMAND_ACCEPTED` / `COMMAND_REJECTED` events via EventBus.

## Key Invariants

1. All state partitioned by `arm_id` from day one
2. Snapshot replaces (never appends) — planner sees exactly one
3. Non-blocking publish — publisher never blocks on slow subscribers
4. Ring buffering — bounded recent history (8 events, 10 acks)
5. Idempotency — accepted command IDs remembered; duplicates return `ALREADY_APPLIED`

## Testing

All components are directly testable. Tests use `HALORuntime()` + `register_arm("arm0")` fixture. `EventBus.subscribe()` returns an `asyncio.Queue` for assertions.
