# SkillRunnerService

10–20 Hz service that owns skill execution via a Mermaid-driven FSM engine, schedules ACT action chunks, and publishes phase/skill lifecycle events.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `SkillRunnerConfig` — FSM thresholds, timeouts, buffer targets, PLACE params, queue size |
| `graph.py` | `FsmGraph`, `FsmNode`, `FsmEdge` — immutable graph topology with validation |
| `mermaid_parser.py` | `parse_mermaid_fsm()` — parses Mermaid `stateDiagram-v2` files → `FsmGraph` |
| `handlers.py` | `StateHandler` / `GlobalGuard` protocols, `HandlerResult` API, built-in handlers for PICK/TRACK/PLACE, handler factory functions |
| `engine.py` | `FsmEngine` — core FSM: `create_run()`, `advance()`, `sync_phase()`, `abort()`, `fail()`, `needs_chunk()` |
| `skill_run.py` | `SkillRun` (mutable runtime state), `QueuedSkill`, `NodeStatus`, `TransitionRecord` |
| `queue.py` | `SkillQueue` — FIFO deque (max 16), dedup by `(skill_name, target_handle)` |
| `definitions.py` | `SkillDefinition`, `SkillRegistry`, `build_default_registry()` — loads PICK/TRACK/PLACE from Mermaid files |
| `view_model.py` | `FsmViewModel`, `NodeViewModel`, `QueuedSkillViewModel`, `build_fsm_view_model()` — UI rendering |
| `service.py` | `SkillRunnerService` — async orchestrator, 3 modes (ACT/sim/track), event publisher, chunk scheduler |
| `fsm.py` | `PickFSM` — legacy (kept for backward compat, not used by service) |
| `track_fsm.py` | `TrackFSM` — legacy (kept for backward compat, not used by service) |

## FSM Engine Architecture

```
Mermaid .mmd files → parse_mermaid_fsm() → FsmGraph (immutable topology)
                                              ↓
FsmEngine(graph, handlers, config, global_guards)
    ↓
create_run() → SkillRun (mutable runtime state)
    ↓
advance(run, now_ms, target, perception, act) → PhaseId | None
```

Skills are authored as **Mermaid stateDiagram-v2** in `configs/skills/{pick,track,place}/default.mmd`. The parser extracts nodes (with PhaseId annotations) and edges. `FsmEngine` evaluates handlers per tick — one transition max.

## Handler & Guard Protocol

```python
class StateHandler(Protocol):
    def evaluate(self, ctx: StateContext) -> HandlerResult: ...

class GlobalGuard(Protocol):
    def check(self, ctx: StateContext) -> HandlerResult | None: ...
```

`HandlerResult` API:
- `HandlerResult.stay()` — no transition
- `HandlerResult.go(node, trigger)` — transition to named node
- `HandlerResult.fail(code, trigger)` — fail with `SkillFailureCode`
- `HandlerResult.done(trigger)` — succeed (→ DONE)

`StateContext` provides: `now_ms`, `elapsed_ms`, `target`, `perception`, `act`, `config`, `state_bag` (mutable dict), `target_handle`, `successors`, `held_object_handle`.

**Evaluation order per tick:** global guards first (any non-None result applies), then node handler.

## Phase Flows

**PICK:**
```
SELECT_GRASP → PLAN_APPROACH → MOVE_PREGRASP → VISUAL_ALIGN → EXECUTE_APPROACH
  → CLOSE_GRIPPER → LIFT → VERIFY_GRASP → DONE
Recovery: MOVE_PREGRASP/VISUAL_ALIGN/EXECUTE_APPROACH → RECOVER_RETRY_APPROACH → MOVE_PREGRASP
```

**TRACK:**
```
ACQUIRING → DONE
```

**PLACE:**
```
SELECT_PLACE → TRANSIT_PREPLACE → DESCEND_PLACE → OPEN → RETREAT → RETURNING → DONE
Recovery: TRANSIT_PREPLACE/DESCEND_PLACE → RECOVER_RETRY_APPROACH → TRANSIT_PREPLACE
```

## Built-in Handlers

| Handler | Phase | Logic |
|---------|-------|-------|
| `SelectGraspHandler` | SELECT_GRASP | Waits for `TRACKING` + correct handle; timeout → `PERCEPTION_LOST` |
| `PlanApproachHandler` | PLAN_APPROACH | Pass-through (v0); timeout → `NO_PROGRESS` |
| `MovePregraspHandler` | MOVE_PREGRASP | Distance < `approach_align_threshold_m` → VISUAL_ALIGN; target loss → recovery |
| `VisualAlignHandler` | VISUAL_ALIGN | Distance < `execute_approach_threshold_m` → EXECUTE_APPROACH; target loss → recovery |
| `ExecuteApproachHandler` | EXECUTE_APPROACH | Distance < `grasp_distance_threshold_m` held for `grasp_persistence_ms` → CLOSE_GRIPPER; target loss → recovery |
| `CloseGripperHandler` | CLOSE_GRIPPER | Timer-based → LIFT |
| `LiftHandler` | LIFT | Timer-based → VERIFY_GRASP (or DONE if `skip_verify_grasp`) |
| `VerifyGraspHandler` | VERIFY_GRASP | Timer-based → DONE |
| `RecoverRetryApproachHandler` | RECOVER_RETRY | Wait `recover_wait_ms`, increment counter; > `max_reacquire_attempts` → `TIMEOUT` |
| `AcquiringHandler` | ACQUIRING | Waits for `TRACKING` + correct handle; timeout → `PERCEPTION_LOST` or `TARGET_MISMATCH` |
| `SelectPlaceHandler` | SELECT_PLACE | Same as SelectGrasp but for PLACE |
| `TransitPreplaceHandler` | TRANSIT_PREPLACE | Distance < `place_align_threshold_m` → DESCEND_PLACE; target loss → recovery |
| `DescendPlaceHandler` | DESCEND_PLACE | Distance < `place_distance_threshold_m` → OPEN; target loss → recovery |
| `OpenHandler` | OPEN | Timer-based → RETREAT |
| `RetreatHandler` | RETREAT | Timer-based → RETURNING |
| `ReturningHandler` | RETURNING | Timer-based → DONE |

## Global Guards

| Guard | Skills | Logic |
|-------|--------|-------|
| `ReacquireFailedGuard` | PICK, PLACE | `REACQUIRE_FAILED` → fail `PERCEPTION_LOST` |
| `PlaceHeldObjectGuard` | PLACE | No `held_object_handle` → fail `PLACE_MISS` |

## Service Modes

**ACT mode:** `chunk_fn` + `push_fn` — requests action chunks when `needs_chunk()`, pushes to ControlService.

**Sim mode:** `start_pick_fn`/`start_place_fn` + `sim_phase_fn` — triggers autonomous sim skill, syncs FSM from telemetry via `engine.sync_phase()`.

**Track mode:** Neither ACT nor sim — waits for perception to establish tracking.

## Service tick() Sequence

1. Fetch snapshot (cached via store, fallback to full build)
2. `engine.advance()` with current target/perception/act
3. On transition: publish `PHASE_EXIT` → update store → publish `PHASE_ENTER`
4. On DONE: publish `SKILL_SUCCEEDED` or `SKILL_FAILED`
5. Update progress (elapsed_ms, delta_distance)
6. If active and `needs_chunk()`: call `chunk_fn` → `push_fn`

In sim mode, `_tick_sim()` defers `start_pick_fn`/`start_place_fn` until FSM exits entry phase. After trigger: syncs FSM from `sim_phase_fn` telemetry. Stale guard (`sim_stale_guard_timeout_ms`) handles immediate IK failure.

## SkillQueue

FIFO deque (max 16 by default). Deduplicates by `(skill_name, target_handle)`. Service drains queue when current skill completes successfully; clears queue on failure. Queue state is synced to `RuntimeStateStore` (via `_sync_queue_to_store()`) after every mutation (enqueue, dequeue, clear), making it visible to the planner as `queued_skills` in the `PlannerSnapshot`.

## Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `runner_rate_hz` | 10.0 | FSM tick frequency |
| `approach_align_threshold_m` | 0.15 | MOVE_PREGRASP → VISUAL_ALIGN |
| `execute_approach_threshold_m` | 0.05 | VISUAL_ALIGN → EXECUTE_APPROACH |
| `grasp_distance_threshold_m` | 0.01 | Grasp trigger distance |
| `grasp_persistence_ms` | 300 | Grasp stability requirement |
| `select_grasp_timeout_ms` | 10,000 | Tracking gate timeout |
| `acquiring_timeout_ms` | 10,000 | TRACK: per-attempt acquisition wait |
| `acquiring_retry_budget` | 3 | TRACK: attempts before failing (total 30s) |
| `plan_approach_timeout_ms` | 30,000 | |
| `move_pregrasp_timeout_ms` | 30,000 | |
| `visual_align_timeout_ms` | 30,000 | |
| `execute_approach_timeout_ms` | 30,000 | |
| `close_gripper_duration_ms` | 30,000 | Timer-based |
| `verify_duration_ms` | 30,000 | Timer-based |
| `lift_duration_ms` | 30,000 | Timer-based |
| `no_target_tolerance_ms` | 30,000 | Target absent this long → recovery |
| `recover_wait_ms` | 500 | Recovery pause before retry |
| `max_reacquire_attempts` | 3 | Before TIMEOUT |
| `buffer_target_ms` | 200 | Chunk request threshold |
| `chunk_horizon_steps` | 10 | v0: 10 steps @ 10 Hz = 1 s |
| `skip_verify_grasp` | False | Skip VERIFY_GRASP phase |
| `select_place_timeout_ms` | 30,000 | PLACE tracking gate |
| `place_align_threshold_m` | 0.10 | TRANSIT_PREPLACE → DESCEND_PLACE |
| `place_distance_threshold_m` | 0.02 | DESCEND_PLACE → OPEN |
| `transit_preplace_timeout_ms` | 30,000 | |
| `descend_place_timeout_ms` | 30,000 | |
| `open_gripper_duration_ms` | 30,000 | Timer-based |
| `retreat_duration_ms` | 30,000 | Timer-based |
| `returning_timeout_ms` | 30,000 | Max wait for return-to-home |
| `sim_stale_guard_timeout_ms` | 2,000 | Wait for first done=False before accepting done=True |
| `max_queue_size` | 16 | SkillQueue capacity |

## Events Published

| Event | Trigger |
|-------|---------|
| `SKILL_STARTED` | `start_skill()` — data includes `skill_name`, `target_handle` |
| `SKILL_SUCCEEDED` | FSM → DONE with SUCCESS — data includes `skill_name` |
| `SKILL_FAILED` | FSM → DONE with FAILURE — data includes `skill_name`, `failure_code` |
| `PHASE_ENTER` | Every phase transition (incl. start) |
| `PHASE_EXIT` | Every phase transition |

## Integration

- **Reads**: target, perception, act, held_object_handle from RuntimeStateStore (via snapshot)
- **Writes**: skill info, outcome, progress, queued_skills to store
- **Publishes**: SKILL_STARTED/SUCCEEDED/FAILED, PHASE_ENTER/EXIT
- **Consumed by**: ControlService (PHASE_ENTER → buffer trim), PlannerService (SKILL_SUCCEEDED/FAILED → replan)
- **Registry**: `build_default_registry()` loads Mermaid files from `configs/skills/`

## Testing

Handlers are pure — test with synthetic `StateContext`. `FsmEngine` tests use `create_run()` + `advance()` loops. Service tests use `HALORuntime` fixture + `seed_store()` helper. Custom `SkillRegistry` can be injected via `registry` param for test-specific skill definitions.
