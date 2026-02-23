# HALO Architecture Specification (Planner + Perception + ACT Skill Runner)

Date: 2026-02-23
Scope: HALO — v1: Isaac Sim/Lab (single-arm pick/place), local models via Ollama, ACT for continuous control. Real SO-ARM101 hardware is a later phase.

This document is the **architecture reference** that complements the plan summary. It focuses on **module boundaries, runtime contracts, dataflows, timing**, and **code-facing interfaces**.

---

## 0) Design principles (non-negotiables)

1) **No stop-and-go motion**
- Continuous control runs independently of LLM reasoning.
- Planning is low-frequency and **never blocks** control.

2) **Deterministic safety**
- Safety-critical decisions live outside the LLM loop (hard guards + reflex layer).

3) **Keep numeric control hints out of LLM context**
- Target vectors, transforms, and controller tuning are passed machine-to-machine.
- Planner consumes **compact snapshots**, not raw telemetry.

4) **Fresh perception over long memory**
- State is summarized into a *latest snapshot* plus a small event ring.
- Old snapshots are deprecated/overwritten in planner context.

---

## 1) System overview

### 1.1 Major services

**PlannerService (LLM agent; stateful, low-rate)**
- Chooses next high-level action: start skill, abort, retry, retarget, request refresh.
- Reads the latest runtime snapshot; writes commands.
- Does *not* time micro-actions like “close gripper now”.

**TargetPerceptionService (unified perception; medium-rate)**
- Maintains a track on the active target and publishes fused **target hints**.
- Internals (not exposed to planner): VLM grounding, segmentation, tracking, depth fusion, plausibility gates.

**SkillRunnerService (deterministic executor; medium-rate)**
- Runs a **fixed FSM** per skill (Pick, Place, etc.).
- Calls ACT to generate action chunks.
- Uses fast success checks to transition phases (approach → align → grasp → lift).

**ControlService (real-time loop; high-rate)**
- Streams actions at 50–100Hz (or as required).
- Applies smoothing + clamps + safety interlocks.
- Never waits on LLM, VLM

**Safety/Reflex (hard real-time-ish)**
- Immediate stop/retract/open-gripper on unsafe conditions.
- Publishes reflex events; returns to safe-hold mode.

**RuntimeStateStore + EventBus**
- Single source of truth for:
  - current skill/phase
  - target hint + validity
  - ACT buffer status
  - safety state
  - command acknowledgements
  - recent events
- Transport can be ROS2 topics, ZeroMQ, shared memory, Redis, etc. (choose later; contracts stay stable).

---

## 2) Deployment topology (processes/threads)

### 2.1 Suggested processes
- `planner_service` (LLM + tool adapter)
- `target_perception_service` (VLM + SAM/Tracker + depth fusion)
- `skill_runner_service` (FSM + ACT inference + chunk planner)
- `control_service` (realtime executor + safety + reflex)
- (optional) `logger_service` (episode capture + dataset writer)

### 2.2 Threading expectations
- Control loop thread has the highest priority; avoid allocations.
- Perception runs separate capture + inference threads (ZED capture + tracking + VLM reacquire jobs).
- SkillRunner runs FSM tick + ACT inference thread(s).
- Planner is single-threaded from an orchestration standpoint.

---

## 3) Dataflows

### 3.1 Control path (machine-to-machine, low latency)
```
Cameras + RobotState
   -> TargetPerceptionService
       -> target_hint_vec (robot frame + EE-relative deltas, validity, confidence)
           -> RuntimeStateStore
               -> SkillRunnerService
                   -> ACT (chunk inference)
                       -> action_chunks
                           -> ControlService (50–100Hz streaming + clamps)
                               -> Robot
```

**Rule:** SkillRunner reads `target_hint_vec` directly from runtime state — it does not go through the Planner.

### 3.2 Decision path (LLM; low frequency)
```
RuntimeStateStore -> get_latest_runtime_snapshot() -> PlannerService -> async commands -> RuntimeStateStore
```

Planner issues commands **asynchronously**. Results appear in:
- command ack fields in snapshots
- event stream (accepted/rejected/stale/already_applied)
- skill outcome events

---

## 4) Timing, rates, and budgets

### 4.1 Typical rates (v0 defaults)
- ControlService: **50–100 Hz**
- ACT inference: **10–20 Hz** (producing short action chunks)
- Wrist camera for ACT: as needed, ideally aligned to ACT rate
- TargetPerceptionService fusion publish: **10–30 Hz** (tracking; fast loop budget ≤80–120ms, excludes VLM reacquire)
- VLM reacquire: event-driven, low duty cycle
- PlannerService: **event-driven** (tick on SKILL_SUCCEEDED/FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE) + **30 s watchdog** fallback. No fixed polling rate — decide_fn (LLM) is awaited before the next event is processed, so ticks are always serialized.

### 4.2 Action chunk buffering (receding horizon)
- Predict horizon: ~200–500ms for moving targets
- Maintain buffer fill: ~150–300ms
- On phase transition: trim buffer to ~50–100ms to avoid “old-phase tail actions”.

### 4.3 Latency budgets (fast loop vs. semantic fallback)

- **Fast perception loop (steady-state):** tracker + depth fusion + plausibility gates + hint publish should target **≤80–120ms** end-to-end (camera frame timestamp → `target_hint_vec` published). This is the budget that supports moving targets.
- **Semantic reacquire loop (VLM):** runs **asynchronously** and is allowed to take **hundreds of ms to several seconds**. It must **never** sit on the critical path of the steady-state 10–30Hz hint publish loop.
  - Triggered only on startup acquisition, repeated validity failures (`hint_valid=false`), `OUT_OF_VIEW`, repeated `TRACK_JUMP_REJECTED`, or explicit refresh requests.
  - Output is a coarse **seed** (bbox/point/ROI + optional label) used to initialize SAM/Tracker, which then resumes the fast loop.

---

## 5) Frames, calibration, and timestamp discipline

### 5.1 Required transforms (calibrated)
- `T_base<-scene_cam` (ZED)
- `T_ee<-wrist_cam` (UVC)
- `T_base<-ee` (kinematics)
- (optional) object/world frames as needed

### 5.2 Timestamp fields (must be present in every hint)
- `hint_ts`
- `robot_state_ts` (EE pose source)
- `time_skew_ms = hint_ts - robot_state_ts`
- `obs_age_ms = now - hint_ts`

### 5.3 Freshness gating (safety interlock)
If `obs_age_ms` or `time_skew_ms` exceeds thresholds:
- `hint_valid = false`
- SkillRunner switches to REACQUIRE / HOLD state
- ControlService may hold position or execute a safe retreat depending on policy

---

## 6) TargetPerceptionService internals (black-box to planner)

### 6.1 Pipeline stages
1) **Target acquisition / reacquisition** (rare, async)
   - VLM runs as an **asynchronous job** to find candidate objects/regions in the scene camera.
   - Output is a coarse seed (bbox/point/ROI + optional label) to initialize segmentation/tracking; it is **not** part of the steady-state tracking loop.
2) **Segmentation** (init/refine)
   - SAM/SAM2 produces a mask for the selected candidate.
3) **Tracking** (continuous)
   - Fast tracker updates mask / keypoints at frame rate.
4) **Depth fusion**
   - Mask + ZED depth -> 3D estimate.
5) **Plausibility gates**
   - Reject impossible motion, low depth-valid ratio, track jumps, etc.
6) **Hint publication**
   - Publish both base-frame pose and EE-relative deltas + confidence.

### 6.2 Implemented modules

**`vlm_parser.py`** — Parse VLM JSON responses into typed dataclasses:
- `VlmDetection` (frozen): `handle`, `label`, `bbox` (x1, y1, x2, y2), `centroid` (computed midpoint), `is_graspable`
- `VlmScene` (frozen): `scene` (description string), `detections` (list)
- `parse_vlm_response(response: dict) -> VlmScene`

**`ollama_vlm_fn.py`** — Ollama VLM integration (async, scene camera only):
- `make_ollama_vlm_fn(base_url, model, prompt_path, ...) -> VlmFn` — factory returning async fn
- Input images resized to `_VLM_INPUT_WIDTH = 1024` for stable bbox coords
- Prompt loaded from `configs/perception/scene_analysis.md`
- `VlmFn = Callable[[str], Awaitable[VlmScene]]` — arm_id → scene analysis result

**`mock_fns.py`** — Mock perception functions backed by `docs/data/mock/` JSON fixtures:
- `make_mock_observe_fn(mock_dir) -> ObserveFn` — returns async fn for tracker simulation
- `make_mock_vlm_fn(mock_dir) -> VlmFn` — returns async fn for VLM simulation (with simulated latency)

**`service.py`** — TargetPerceptionService orchestrates the fast loop + async VLM:
- Type aliases: `ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]`, `VlmFn = Callable[[str], Awaitable[VlmScene]]`
- VLM is optional (`vlm_fn=None` disables async reacquisition)
- At most one VLM task at a time; result stored as `_vlm_seed`, consumed by `tick()` when `observe_fn` returns `None`
- VLM result publishes `VLM_RESULT` event and is logged via `RunLogger.log_vlm_result()`

### 6.3 VLM prompt (`configs/perception/scene_analysis.md`)
Structured prompt for Qwen2.5-VL:
- Detect cubes, boxes, containers, balls, bottles, cups on the table surface
- JSON output: `{"scene":"...","detections":[{"handle":"...","label":"...","bounding_box":[x1,y1,x2,y2],"is_graspable":bool}]}`
- `handle` format: `<type>-<N>` (e.g., `cube-1`, `box-2`)

### 6.4 Health/failure codes (stable enums)
- `OCCLUDED`
- `OUT_OF_VIEW`
- `DEPTH_INVALID`
- `MULTIPLE_CANDIDATES`
- `CALIB_INVALID`
- `TRACK_JUMP_REJECTED`
- `REACQUIRE_FAILED`
- `OK`

### 6.5 What the planner can request
- `set_tracking_target(...)`
- `request_refresh(mode, reason)` (rare)

---

## 7) SkillRunner architecture (FSM + ACT)

### 7.1 What SkillRunner owns
- Skill FSM and phase transitions
- Phase-conditioned policy inputs (`phase_id`)
- Fast success checks and retry logic
- Buffer trimming and chunk scheduling
- Optional verification hooks (e.g., VLM verify after grasp)

### 7.2 Pick skill FSM (canonical states)
- `IDLE`
- `ACQUIRE_TARGET`
- `REACQUIRE_TARGET`
- `APPROACH`
- `PREGRASP_ALIGN`
- `GRASP_CLOSE`
- `VERIFY_GRASP` (optional)
- `WAIT_VERIFY` (optional, perception verify)
- `LIFT`
- `GRASP_RECOVER`
- `SUCCESS_PICK`
- `FAIL(<reason>)`

**Critical:** `GRASP_CLOSE` is triggered deterministically when in the grasp window (distance + persistence), not by the planner.

### 7.3 Outcome monitoring
A `SkillOutcomeMonitor` computes:
- `in_progress | success | failure | uncertain`
- `reason_code`
- `needs_verify`

Signals:
- target following EE, height change, in-bin containment
- gripper width/effort/current (if available)
- progress watchdog (`delta_distance`, `no_progress_ms`)
- safety/reflex events

---

## 8) ACT integration contract

### 8.1 Inputs to ACT
- Wrist RGB (primary)
- Robot proprio (joints, gripper)
- Low-dimensional target hints (prefer EE-relative)
- `phase_id` token

### 8.2 Outputs from ACT
- Action chunks in a fixed action space (define explicitly; v0 example):
  - `Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd`
- Chunk timestamps + ids for traceability.

**Delta semantics (v0 default, to avoid drift):**
- Each `Δ*` is a **per-timestep incremental command in the EE frame**, applied relative to the **current measured** EE pose (receding-horizon servo), with the low-level controller closing the loop.
- Do **not** integrate and “play back” a full chunk open-loop; always treat chunks as a short rolling buffer that is continuously refreshed.
- If using temporal ensembling, ensemble **in action space per timestep** (i.e., overlapping predicted deltas are blended into a *single* commanded delta sequence) before mapping to IK/OSC.

**Alternative encodings (if drift shows up):**
- Predict **absolute EE poses relative to the chunk start** (or joint targets) and treat ACT output as setpoints; this often transfers better to real hardware when controllers have latency.

### 8.3 Execution mapping
- SkillRunner generates desired deltas.
- ControlService maps deltas to your low-level controller (IK / resolved-rate / operational space) with:
  - clamps (vel/acc/jerk)
  - interpolation / sample-and-hold
  - collision/workspace limits

---

## 9) SafetyGuard + Reflex layer

### 9.1 Hard guards (pre-execution)
- joint limits
- workspace AABB limits
- velocity/acceleration/jerk limits
- coarse collision checks
- stale-hint / invalid-calibration interlocks

### 9.2 Reflexes (immediate overrides)
- stop
- retract-to-safe pose
- open gripper
- disable torque / estop integration as appropriate

### 9.3 Contract with planner
- Planner may request recovery actions only after robot is stabilized.
- Reflex events are surfaced in snapshot + event stream.

---

## 10) Runtime contracts (commands, snapshots, events)

### 10.1 Command protocol (async + idempotent)

**Command envelope**
```json
{
  "command_id": "uuid",
  "arm_id": "arm0",
  "issued_at_ms": 0,
  "type": "start_skill | abort_skill | override_target | request_perception_refresh",
  "precondition_snapshot_id": "snap-123",
  "payload": {}
}
```

**Idempotency rules**
- duplicate `command_id` => `already_applied`
- stale precondition => `rejected:stale`
- wrong skill_run => `rejected:wrong_skill_run`

### 10.2 Planner snapshot (compact, planner-grade)

```json
{
  "snapshot_id": "snap-123",
  "ts_ms": 0,
  "arm_id": "arm0",

  "skill": {"name": "pick", "skill_run_id": "run-9", "phase": "PREGRASP_ALIGN"},
  "target": {
    "handle": "cube-1",
    "hint_valid": true,
    "confidence": 0.84,
    "obs_age_ms": 23,
    "time_skew_ms": -5,
    "delta_xyz_ee": [0.03, -0.01, 0.08],
    "distance_m": 0.09
  },

  "perception": {"tracking_status": "tracking", "failure_code": "OK", "reacquire_fail_count": 0},
  "act": {"status": "running", "buffer_fill_ms": 220, "buffer_low": false},

  "progress": {"elapsed_ms": 4300, "no_progress_ms": 0, "delta_distance": -0.01},
  "outcome": {"state": "in_progress", "reason_code": null, "needs_verify": false},

  "safety": {"state": "OK", "reflex_active": false, "reason_codes": []},

  "command_ack": [{"command_id": "uuid", "status": "accepted"}],
  "recent_events": [{"event_id": "evt-77", "type": "PHASE_ENTER", "data": {"phase": "PREGRASP_ALIGN"}}]
}
```

**Planner context rule (implementation-critical):**
- The planner must see **exactly one** `get_latest_runtime_snapshot()` payload: the *latest* one.
- When a new snapshot arrives, middleware must **replace** the prior snapshot tool output in the LLM context (do not append multiple snapshots).
- Keep only a small `recent_events` ring; never stream raw telemetry into the planner prompt.
- Every mutating command must include `precondition_snapshot_id`; the command router must reject stale preconditions even if the LLM repeats calls.


### 10.3 Event stream (small, canonical)
Event types (examples):
- `COMMAND_ACCEPTED / COMMAND_REJECTED`
- `SKILL_STARTED / SKILL_SUCCEEDED / SKILL_FAILED`
- `PHASE_ENTER / PHASE_EXIT`
- `PERCEPTION_FAILURE / PERCEPTION_RECOVERED`
- `SAFETY_REFLEX_TRIGGERED / SAFETY_RECOVERED`

---

## 11) Logging, tracing, and dataset capture

### 11.1 What to log (minimum)
- `snapshot_id`, `skill_run_id`, `phase`, `chunk_id`
- `target_hint_version` (monotonic counter)
- `command_id` + ack status
- state transitions with reason codes
- safety events + reflex reasons
- per-episode success/failure labels

### 11.2 Dataset episode schema (teleop-aligned)
Record at ACT timestep:
- wrist RGB
- robot state
- executed action
- `phase_id`
- target hints (EE-relative)
- success/failure per episode

Add QA filters:
- dropped frames / timestamp gaps
- saturation/clipping frequency
- replay mismatch > threshold
- corrupted images

---

## 12) Extensibility notes

### 12.1 Multi-arm
Namespace everything by `arm_id` from day one:
- runtime state partitions
- command routing
- calibration sets
- logs and dataset episodes

### 12.2 Adding skills
Each skill should provide:
- FSM definition (states, transitions, predicates)
- ACT phase ids (stable enums)
- per-phase thresholds + timing profile
- outcome monitor rules
- failure taxonomy mapping

### 12.3 Debug / inspection mode
Expose a separate debug snapshot tool that can include:
- full joint arrays
- raw transforms
- detailed tracker stats
- last N images metadata
Keep it **off** the steady-state planner loop.

---

## 13) Repo structure (actual)

```
halo/
  contracts/
    enums.py            # all enums (PhaseId, SafetyReflexReason, ActStatus, etc.)
    snapshots.py        # PlannerSnapshot, TargetInfo, ActInfo, SafetyInfo, etc.
    commands.py         # CommandEnvelope, CommandAck, payload types
    events.py           # EventEnvelope, EventType
    actions.py          # Action, ActionChunk, ZERO_ACTION
    enums.json          # JSON schema
    commands.json
    events.json
    snapshot.json
  runtime/
    runtime.py          # HALORuntime (register_arm, get_latest_runtime_snapshot, submit_command)
    state_store.py      # RuntimeStateStore (per-arm state, snapshot caching)
    event_bus.py        # EventBus (subscribe/unsubscribe/publish/get_recent_events)
    command_router.py   # CommandRouter (idempotency + precondition + skill-run validation)
  services/
    planner_service/
      config.py               # PlannerServiceConfig (watchdog_interval_s, max_commands_per_tick)
      snapshot_serializer.py  # snapshot_to_dict() — PlannerSnapshot → plain dict for LLM
      tools.py                # AgentContext, build_tools() — 4 LangChain @tool functions
      agent.py                # PlannerAgent, make_decide_fn() — LangGraph ReAct agent
      service.py              # PlannerService (event-driven loop, 30 s watchdog)
    target_perception_service/
      config.py           # TargetPerceptionServiceConfig
      service.py          # TargetPerceptionService (fast loop + async VLM)
      vlm_parser.py       # VlmDetection, VlmScene, parse_vlm_response()
      ollama_vlm_fn.py    # make_ollama_vlm_fn() — Ollama VLM client factory
      mock_fns.py         # make_mock_observe_fn(), make_mock_vlm_fn() — mock perception
    skill_runner_service/
      config.py           # SkillRunnerConfig
      fsm.py              # PickFSM (pure synchronous state machine)
      service.py          # SkillRunnerService
    control_service/
      config.py           # ControlServiceConfig
      action_buffer.py    # ActionBuffer (legacy, kept for compatibility)
      te_buffer.py        # TemporalEnsemblingBuffer (production blending)
      safety_guard.py     # SafetyGuard (check, clamp, hint_freshness)
      service.py          # ControlService (50–100 Hz loop)
  tui/
    app.py              # Textual TUI — mock + live modes
    run_logger.py       # RunLogger: writes JSONL session logs to runs/
  models/               # (planned) act/, vlm/
  configs/
    planner/
      system_prompt.md        # core agent instructions
      skills/pick.md          # PICK skill reference for planner
      skills/place.md         # PLACE skill reference for planner
    perception/
      scene_analysis.md       # VLM prompt for qwen2.5vl
    calib/                    # (planned)
    skills/                   # (planned)
    safety/                   # (planned)
  tools/                      # (planned) ollama_clients/, zed_capture/, uvc_capture/
  eval/                       # (planned) sim/, real/
docs/
  halo_architecture.md        # this file
  halo_plan_summary.md        # project plan + Isaac Lab strategy
  data/mock/                  # mock data: observe_fn_result.json, vlm_response.json, perception_info.json, mock.png
tests/                        # 207 unit tests (14 test modules)
integration/                  # LLM integration tests (require Ollama)
  conftest.py                 # Ollama health-check; auto-skip if model unavailable
  runs/                       # timestamped result folders
runs/                         # live TUI session logs (JSONL, git-ignored)
```

---

## 14) Quick “who owns what” checklist

- **Planner** owns: task orchestration, retries, retargeting, high-level recovery decisions.
- **Perception** owns: target discovery/track, fused hints, validity/confidence, failure codes.
- **SkillRunner** owns: phase timing, success predicates, ACT chunking, micro-retries.
- **ControlService** owns: real-time streaming, smoothing, clamps, enforcing safety gates.
- **Safety/Reflex** owns: immediate overrides; LLM cannot bypass.

---

## Appendix A — Naming conventions (stable)

- `PlannerService`
- `TargetPerceptionService`
- `SkillRunnerService`
- `ControlService`
- `SafetyGuard`, `ReflexLayer`
- `SkillOutcomeMonitor`
- `RuntimeStateStore`, `EventBus`

