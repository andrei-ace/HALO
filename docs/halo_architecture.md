# HALO Architecture Specification (Planner + Perception + ACT Skill Runner)

Date: 2026-03-02
Scope: HALO — three-phase sim strategy: (1) MuJoCo + SO-101 (current), (2) Isaac Lab (future), (3) real SO-ARM101 hardware (later). Single-arm pick/place, local models via Ollama, ACT for continuous control.

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
- `mujoco_sim.server` (sim-only: ZMQ server owning SO101Env + PickTeacher; single-threaded for macOS OpenGL)
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
- ControlService: **50–100 Hz** (target); **20 Hz** in v0 MuJoCo sim
- ACT inference: **10–20 Hz** (producing short action chunks)
- Wrist camera for ACT: as needed, ideally aligned to ACT rate
- TargetPerceptionService fusion publish: **10–30 Hz** (tracking; fast loop budget ≤80–120ms, excludes VLM reacquire)
- VLM reacquire: event-driven, low duty cycle
- PlannerService: **event-driven** (tick on SKILL_SUCCEEDED/FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE, SCENE_DESCRIBED, TARGET_ACQUIRED, COMMAND_REJECTED) + **30 s watchdog** fallback. No fixed polling rate — decide_fn (LLM) is awaited before the next event is processed, so ticks are always serialized.

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
- `VlmFn = Callable[[str, object], Awaitable[VlmScene]]` — (arm_id, image) → scene analysis result
- Image provided per-call by the service (captured from camera via `capture_fn`)

**`video_capture_fn.py`** — Video file camera simulation:
- `make_video_capture_fn(video_path) -> CaptureFn` — factory returning async fn backed by a looping video file (OpenCV)

**`mock_fns.py`** — Test factories:
- `make_mock_capture_fn() -> CaptureFn` — synthetic frames with counter
- `make_mock_tracker_factory_fn(init_hint, update_hint) -> TrackerFactoryFn` — predictable tracker

**`service.py`** — TargetPerceptionService orchestrates the fast loop + async VLM:
- Type aliases: `ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]`, `VlmFn = Callable[[str, object], Awaitable[VlmScene]]`
- VLM is optional (`vlm_fn=None` disables async reacquisition)
- At most one VLM task at a time; result stored as `_vlm_seed`, consumed by `tick()` when `observe_fn` returns `None`
- VLM result publishes `SCENE_DESCRIBED` event and is logged via `RunLogger.log_vlm_result()`

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
- `set_tracking_target(...)` — also available as `track_object(target_handle)` planner tool
- `describe_scene(reason)` — triggers async VLM scene analysis; result delivered via `SCENE_DESCRIBED` event

---

## 7) SkillRunner architecture (FSM + ACT)

### 7.1 What SkillRunner owns
- Skill FSM and phase transitions
- Phase-conditioned policy inputs (`phase_id`)
- Fast success checks and retry logic
- Buffer trimming and chunk scheduling
- Optional verification hooks (e.g., VLM verify after grasp)

### 7.2 Pick skill FSM (implemented states)
- `IDLE` (0) — initial state
- `SELECT_GRASP` (1) — v0 pass-through (immediate transition)
- `PLAN_APPROACH` (2) — v0 pass-through (immediate transition)
- `MOVE_PREGRASP` (3) — move to pregrasp pose
- `VISUAL_ALIGN` (4) — fine alignment (wrist camera active)
- `EXECUTE_APPROACH` (5) — descend to grasp pose; grasp persistence timer starts when distance < threshold (wrist camera active)
- `CLOSE_GRIPPER` (6) — gripper close + dwell (wrist camera active)
- `VERIFY_GRASP` (7) — optional (configurable via `skip_verify_grasp`) (wrist camera active)
- `LIFT` (8) — lift after grasp (wrist camera active)
- `DONE` (9) — terminal state (outcome: SUCCESS or FAILURE with reason code)
- Place reserved: `PLACE_*` (30–33)
- Recovery: `RECOVER_RETRY_APPROACH` (50), `RECOVER_REGRASP` (51), `RECOVER_ABORT` (52)

**Critical:** `CLOSE_GRIPPER` is triggered deterministically when distance < `grasp_distance_threshold_m` held for `grasp_persistence_ms`, not by the planner.

Wrist camera active phases: `VISUAL_ALIGN`, `EXECUTE_APPROACH`, `CLOSE_GRIPPER`, `VERIFY_GRASP`, `LIFT` (defined as `WRIST_ACTIVE_PHASES` in `contracts/enums.py`).

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
  - **HALO core (runtime/bridge):** `Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd` — 7D EE-frame deltas
  - **MuJoCo sim (`mujoco_sim/`):** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` — 6D joint-position targets written to `data.ctrl[:]`
- Chunk timestamps + ids for traceability.
- Action space is **intentionally different** between core and sim (EE-delta vs joint-position); conversion is the responsibility of the `apply_fn` factory.

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
v0 implements:
- per-timestep linear/angular delta magnitude limits (`max_linear_delta_m`, `max_angular_delta_rad`)
- stale-hint / invalid-calibration interlocks (hint freshness gating in ControlService)

Planned (not yet implemented):
- absolute workspace AABB limits
- velocity/acceleration/jerk rate limits
- coarse collision checks

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
  "type": "start_skill | abort_skill | override_target | describe_scene | track_object",
  "precondition_snapshot_id": "snap-123 | null",
  "payload": {}
}
```

**Idempotency rules**
- duplicate `command_id` => `ALREADY_APPLIED`
- stale precondition => `REJECTED_STALE`
- wrong skill_run => `REJECTED_WRONG_SKILL_RUN`
- `precondition_snapshot_id = null` (used by `describe_scene`, `track_object`) => accepted without precondition check

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

  "command_acks": [{"command_id": "uuid", "status": "ACCEPTED"}],
  "recent_events": [{"event_id": "evt-77", "type": "PHASE_ENTER", "data": {"phase": "PREGRASP_ALIGN"}}]
}
```

**Planner context rule (implementation-critical):**
- The planner must see **exactly one** `get_latest_runtime_snapshot()` payload: the *latest* one.
- When a new snapshot arrives, middleware must **replace** the prior snapshot tool output in the LLM context (do not append multiple snapshots).
- Keep only a small `recent_events` ring; never stream raw telemetry into the planner prompt.
- Every mutating command must include `precondition_snapshot_id`; the command router must reject stale preconditions even if the LLM repeats calls. Stateless commands (`describe_scene`, `track_object`) set `precondition_snapshot_id = null` to avoid premature rejection.


### 10.3 Event stream (small, canonical)
Event types:
- `COMMAND_ACCEPTED / COMMAND_REJECTED`
- `SKILL_STARTED / SKILL_SUCCEEDED / SKILL_FAILED`
- `PHASE_ENTER / PHASE_EXIT`
- `PERCEPTION_FAILURE / PERCEPTION_RECOVERED`
- `SCENE_DESCRIBED`
- `TARGET_ACQUIRED`
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
    snapshots.py        # PlannerSnapshot, TargetInfo, ActInfo(wrist_enabled), SafetyInfo, etc.
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
  bridge/                          # ZMQ bridge to MuJoCo sim server
    __init__.py                    # BridgeTransportError exception
    config.py                      # SimBridgeConfig (2-channel URLs, managed flag, timeouts)
    sim_client.py                  # SimClient (ZMQ client: TelemetryStream SUB + CommandRPC REQ)
    sim_source.py                  # SimSource (drop-in MuJocoVideoSource replacement via ZMQ)
    transforms.py                  # world_to_ee_frame() quaternion math
  services/                        # each service has its own CLAUDE.md with detailed docs
    planner_service/
      CLAUDE.md                 # tools, one-tool-per-tick, loop detection, snapshot middleware
      config.py                 # PlannerServiceConfig (watchdog_interval_s, max_commands_per_tick)
      snapshot_serializer.py    # snapshot_to_dict() — PlannerSnapshot → plain dict for LLM
      tools.py                  # AgentContext, build_tools() — 5 plain functions (ADK introspects signature + docstring)
      agent.py                  # PlannerAgent, make_decide_fn() — ADK Agent with LiteLlm (Ollama)
      service.py                # PlannerService (event-driven loop, 30 s watchdog)
    target_perception_service/
      CLAUDE.md             # tick logic, plausibility gates, VLM async pipeline, state transitions
      config.py             # TargetPerceptionServiceConfig
      service.py            # TargetPerceptionService (fast loop + async VLM)
      vlm_parser.py         # VlmDetection, VlmScene, parse_vlm_response()
      ollama_vlm_fn.py      # make_ollama_vlm_fn() — Ollama VLM client factory
      video_capture_fn.py   # make_video_capture_fn() — looping video file CaptureFn
      frame_buffer.py       # CapturedFrame, FrameRingBuffer — replay buffer
      mock_fns.py           # make_mock_capture_fn(), make_mock_tracker_factory_fn()
    skill_runner_service/
      CLAUDE.md             # FSM phase flow, advance() check order, grasp persistence, recovery
      config.py             # SkillRunnerConfig
      fsm.py                # PickFSM (pure synchronous state machine)
      service.py            # SkillRunnerService
    control_service/
      CLAUDE.md             # tick order, TE buffer, reflex lifecycle, safety guard
      config.py             # ControlServiceConfig
      action_buffer.py      # ActionBuffer (legacy, kept for compatibility)
      te_buffer.py          # TemporalEnsemblingBuffer (production blending)
      safety_guard.py       # SafetyGuard (check, clamp, hint_freshness)
      service.py            # ControlService (50–100 Hz loop)
  testing/                         # multi-tier test framework
    event_recorder.py              # EventRecorder (subscribe, wait_for, query by type)
    state_seeder.py                # make_target(), make_perception(), seed_store()
    mock_fns.py                    # LatencyProfile, mock factories for all callables
    runner.py                      # HeadlessRunner (orchestrate services without TUI)
    metrics.py                     # RunReport, compute_run_report()
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
mujoco_sim/                     # MuJoCo + SO-101 sim (see section 15)
  mujoco_sim/
    constants.py                # phase IDs, action fields, gripper semantics (synced from halo.contracts)
    scene_info.py               # SceneInfo: runtime geometry extraction + all grasp/teacher constants
    config/
      env_config.py             # EnvConfig (SO-101, dual cameras, 20 Hz)
    env/
      so101_env.py              # SO101Env: raw MuJoCo wrapper, dual cameras, joint-position control
    dataset/
      raw_episode.py            # Timestep (with phase_id, joint_pos), RawEpisode, EpisodeMetadata
      writer_hdf5.py            # write_episode() — HDF5 serialization with gzip images
      reader_hdf5.py            # read_episode() — HDF5 → RawEpisode
    teacher/
      pick_teacher.py           # PickTeacher: trajectory-planned policy (pre-computed on first step)
      ik_helper.py              # Damped least-squares IK (position-only + coupled 6D)
      keyframe_planner.py       # SE(3) keyframes from GraspPose
      waypoint_generator.py     # Keyframes → joint waypoints via IK (yaw-retry fallbacks)
      trajectory.py             # Jerk-limited ruckig trajectory planning
      grasp_planner.py          # 64-candidate grasp enumeration, filtering, scoring
    runner/
      run_teacher.py            # run_teacher(): episode generation loop + verification
    server/
      __init__.py               # SimServer: ZMQ server owning SO101Env + PickTeacher
      __main__.py               # CLI: python -m mujoco_sim.server
      config.py                 # SimServerConfig (host, ports, render FPS, JPEG quality)
      handlers.py               # dispatch_command(): route ZMQ messages
      protocol.py               # Message types, msgpack/JPEG serialization
    scripts/                    # test_env, generate_episodes, inspect_episode, visualize_ik_pose
    assets/
      so101/                    # MJCF (so101_new_calib.xml) + pick_scene.xml + 13 STL meshes
    tests/                      # 112 tests (constants, dataset, teacher, grasp, trajectory, server)
docs/
  halo_architecture.md          # this file
  halo_plan_summary.md          # project plan + Isaac Lab strategy
  data/                         # gitignored; video.mp4 for video capture simulation
sim/                            # Isaac Lab extension (planned — see sim/README.md)
tests/                          # unit + component + system tests
  e2e/                          # end-to-end tests (MuJoCo source + VLM, auto-skip if unavailable)
integration/                    # LLM integration tests (require Ollama)
  conftest.py                   # Ollama health-check; auto-skip if model unavailable
  runs/                         # timestamped result folders
runs/                           # live TUI session logs (JSONL, git-ignored)
```

---

## 14) Quick “who owns what” checklist

- **Planner** owns: task orchestration, retries, retargeting, high-level recovery decisions.
- **Perception** owns: target discovery/track, fused hints, validity/confidence, failure codes.
- **SkillRunner** owns: phase timing, success predicates, ACT chunking, micro-retries.
- **ControlService** owns: real-time streaming, smoothing, clamps, enforcing safety gates.
- **Safety/Reflex** owns: immediate overrides; LLM cannot bypass.

---

## 15) MuJoCo simulation module (`mujoco_sim/`)

Phase 1 of the sim strategy. Raw MuJoCo + SO-101 arm (no robosuite). Generates teacher pick demos, records episodes to HDF5, and bridges to HALO runtime via ZMQ. Separate workspace member with its own `pyproject.toml`.

### 15.1 SO-101 robot

5-DOF arm + 1-DOF gripper = 6 actuated joints (position-controlled STS3215 servos). EE site: `gripperframe`. Position actuators (`kp=998.22`) track targets written to `data.ctrl[:]`. Physics: `dt=0.005`, 20 Hz control, 10 substeps per step.

MJCF: `so101_new_calib.xml` + `pick_scene.xml` (robot + floor + cube + dual cameras). Contact solver tuned for zero slip: `impratio=10`, `cone="elliptic"`, `noslip_iterations=3`, gripper friction=2.0, actuator force=±6.0 N.

### 15.2 SO101Env

Raw MuJoCo wrapper. Action: 6D joint-position targets. Observations: `rgb_scene` (480,640,3), `rgb_wrist` (240,320,3), `qpos` (13,), `qvel` (12,), `gripper`, `ee_pose` (7,), `object_pose` (7,), `joint_pos` (6,). Seeded resets with cube position randomization.

### 15.3 Trajectory planning pipeline

PickTeacher pre-computes a full trajectory on first `step()`, then samples in real time:

```
grasp_planner (64 candidates, geometric filter, IK scoring)
  → keyframe_planner (5 SE(3) keyframes: home → pregrasp → grasp → close → lift)
    → waypoint_generator (IK with yaw-retry fallbacks)
      → trajectory (jerk-limited ruckig segments, start/end at rest)
        → pick_teacher.step() samples at elapsed time → (action, phase_id, done)
```

Teacher phase sequence: `IDLE → MOVE_PREGRASP → EXECUTE_APPROACH → CLOSE_GRIPPER → LIFT → DONE`. Planning-only phases (SELECT_GRASP, PLAN_APPROACH, VISUAL_ALIGN) are folded into the initial computation.

### 15.4 Scene constants (`scene_info.py`)

Single source of truth for all grasp/teacher defaults. `SceneInfo.from_model()` extracts runtime geometry (cube sizes, table height) from MuJoCo model. Key constants: `TCP_PINCH_OFFSET_LOCAL=[0,0,0]`, face standoff 3 mm, 64 grasp candidates (16/face, 5° cone), pregrasp standoff 0.08 m, lift height 0.08 m, IK ori tolerance 55°.

### 15.5 Dataset format (HDF5)

One file per episode. Observations (rgb gzipped), actions (6D), phase_id, metadata as attrs. In-memory: `Timestep` → `RawEpisode` buffer. Episode generation: reset → 5 s stabilization → teacher loop → write HDF5 → verify lift. **100% success rate** with current tuning.

### 15.6 Constants sync

Phase IDs, gripper semantics, wrist-active phases synced between `halo/contracts/enums.py` and `mujoco_sim/constants.py`, verified by cross-module tests. Action space intentionally different (sim: 6D joint-position, core: 7D EE-delta).

### 15.7 Status

112 tests (constants, dataset, teacher, grasp planner, trajectory pipeline, server). PR1-3 done (env, dataset, teacher). PR4-6 pending (phase FSM, VCR replay, annotation).

---

## 16) ZMQ bridge (`halo/bridge/`)

Connects HALO runtime to MuJoCo sim server via 2-channel ZMQ.

| Channel | ZMQ Pattern | Port | Direction | Purpose |
|---------|-------------|------|-----------|---------|
| TelemetryStream | PUB/SUB | 5560 | Sim → HALO | Frames + state @ 10 Hz |
| CommandRPC | REQ/REP | 5561 | HALO → Sim | step, reset, teacher_step, configure, shutdown |

**SimServer** (`mujoco_sim.server`): single-threaded main loop (macOS OpenGL), owns SO101Env + PickTeacher. Protocol: msgpack + JPEG.

**SimClient** (`halo/bridge/sim_client.py`): background telemetry thread (SUB), main thread for commands (REQ, thread-safe). Managed mode spawns server subprocess. `BridgeTransportError` on timeout (ControlService catches → `ActStatus.STALE`).

**SimSource** (`halo/bridge/sim_source.py`): drop-in video source replacement, wraps SimClient. Provides `capture_fn` returning `CapturedFrame` for TargetPerceptionService.

**Transforms** (`halo/bridge/transforms.py`): `world_to_ee_frame()` quaternion rotation for ACT command conversion.

---

## 17) Cognitive backend switching (`halo/cognitive/`)

Transparent proxy layer that routes planner (LLM) and perception (VLM) calls through a **Switchboard** to one of two backends — **LOCAL** (Ollama) or **CLOUD** (Gemini Live API / remote HTTP). Split-brain prevention via **LeaseManager**, context continuity via **ContextStore**, and ADK-native event compaction with cross-backend sync.

PlannerService and TargetPerceptionService call `switchboard.decide()` / `switchboard.vlm_scene()` as drop-in replacements — they are unaware of which backend is active.

### 17.1 Component overview

| File | Role |
|---|---|
| `config.py` | `BackendType`, `BackendReadiness`, `CognitiveConfig`, `LocalConfig`, `CloudConfig`, `RemoteCloudConfig`, `CompactionConfig` |
| `backend.py` | `CognitiveBackend` protocol (decide + vlm_scene + health_check), `WarmableBackend` extension (warm_up + readiness + caught_up_cursor) |
| `switchboard.py` | Transparent proxy, retry logic, failure counting, failover/failback, health loop, event journal loop, compaction sync |
| `lease.py` | `LeaseManager` + `Lease` — epoch-monotonic grants with UUID token + TTL; `CommandRouter` rejects stale epoch/token |
| `context_store.py` | `ContextStore` (append-only journal), `ContextEntry`, `ContextSnapshot`, `CognitiveState` for handoff |
| `compactor.py` | `MessageHistory` (UUID-tracked parallel message list), `CompactionResult` for cross-backend sync |
| `local_backend.py` | `LocalCognitiveBackend` — wraps PlannerAgent (ADK + LiteLLM/Ollama) + Ollama VLM |
| `cloud_backend.py` | `CloudCognitiveBackend` — Gemini Live API, ADK compaction, audio I/O |
| `remote_backend.py` | `RemoteCognitiveBackend` — HTTP client to Cloud Run cognitive service |
| `live_session.py` | `LivePlannerSession` — Gemini Live API session management |
| `audio_io.py` | Audio capture/playback for voice interaction with cloud backend |

### 17.2 Failover flow

When the active backend fails 3 consecutive times (retries exhausted or non-retryable 429/quota errors), the Switchboard automatically fails over to the alternate backend.

```mermaid
sequenceDiagram
    participant PS as PlannerService
    participant SB as Switchboard
    participant Active as Active Backend (cloud)
    participant Standby as Standby Backend (local)
    participant LM as LeaseManager
    participant CS as ContextStore
    participant Bus as EventBus

    PS->>SB: decide(snapshot)
    SB->>Active: decide() — attempt 1
    Active-->>SB: error
    SB->>Active: decide() — attempt 2 (after 0.5s)
    Active-->>SB: error
    SB->>Active: decide() — attempt 3 (after 1.0s)
    Active-->>SB: error (3 retries exhausted)

    Note over SB: consecutive_failures reaches 3

    SB->>CS: take_snapshot(epoch)
    SB->>CS: build_cognitive_state(epoch, snapshot)
    SB->>CS: get_entries_after(-1)
    SB->>Standby: warm_up(state, journal_entries)
    SB->>LM: revoke(old_epoch)
    SB->>LM: grant("local") → new epoch + token
    SB->>Active: reset_loop_state()
    SB->>Standby: reset_loop_state()
    SB->>Bus: publish(BACKEND_SWITCHED)

    Note over SB: Replay original call on new backend

    SB->>Standby: decide(snapshot)
    Standby-->>SB: commands
    SB->>SB: stamp epoch + lease_token on commands
    SB-->>PS: commands
```

### 17.3 Failback flow (warm-up handoff)

A background health loop (every 5 s) checks whether the preferred backend has recovered. Failback uses a graduated warm-up protocol to ensure seamless context transfer before switching.

```mermaid
sequenceDiagram
    participant HL as Health Loop
    participant SB as Switchboard
    participant Preferred as Preferred Backend (cloud)
    participant CS as ContextStore

    Note over SB: Currently on fallback (local)

    HL->>Preferred: health_check()
    Preferred-->>HL: healthy ✓

    alt readiness = COLD / FAILED
        HL->>CS: build_cognitive_state(epoch, snapshot)
        HL->>CS: get_entries_after(-1)
        HL->>Preferred: warm_up(full state + all journal entries)
        Note over Preferred: readiness → WARMING
    else readiness = WARMING
        HL->>CS: get_entries_after(caught_up_cursor)
        HL->>Preferred: warm_up(null state, incremental batch ≤20)
        Note over Preferred: readiness → READY (when caught up)
    else readiness = READY
        HL->>HL: check cursor parity
        alt cursor behind
            HL->>Preferred: warm_up(catchup batch)
        else cursor caught up
            HL->>SB: switch_to(preferred, "recovered and warmed up")
            Note over SB: Full switch_to() protocol (see §17.2)
        end
    end
```

### 17.4 ADK compaction and cross-backend sync

The cloud backend uses ADK-native event compaction to keep session context bounded. When compaction occurs, the summary is propagated to the inactive local backend so failback starts with concise context.

```mermaid
sequenceDiagram
    participant CB as CloudBackend
    participant ADK as ADK Session
    participant MH as MessageHistory
    participant SB as Switchboard
    participant LB as LocalBackend

    Note over CB: After ~20 events on cloud session

    CB->>ADK: decide() returns response
    CB->>CB: _detect_compaction(session_events)
    Note over CB: Finds compaction boundary in ADK events

    CB->>MH: apply_compaction(up_to_msg_id, summary)
    Note over MH: Replace old records with single summary record

    CB->>SB: on_compaction(CompactionResult)

    SB->>LB: agent.reset_session()
    SB->>LB: agent.inject_handoff_context(summary)
    SB->>LB: msg_history.clear()

    SB->>CS: append(entry_type="compaction", summary)
    Note over CS: Journal records compaction event
```

### 17.5 Lease protocol

The `LeaseManager` prevents split-brain — only one backend may issue commands at a time.

- **Epoch**: monotonically increasing integer, incremented on every `grant()`
- **Token**: UUID string, unique per grant — prevents replayed commands from a prior epoch that happen to share the same epoch number
- **TTL**: 30 s default, renewed on every successful `decide()` / `vlm_scene()` call
- **Validation**: `CommandRouter` checks both `epoch` and `lease_token` on every command when a `LeaseManager` is active; commands with missing or stale values are rejected

Lifecycle: `grant(holder) → renew(epoch) → revoke(epoch) → grant(new_holder)`

### 17.6 ContextStore journal

Append-only journal (bounded to 200 entries) that captures what the planner knows and has decided, enabling context transfer across backend switches.

**Entry types:**

| Type | When recorded | Effect on tracked state |
|---|---|---|
| `decision` | After successful `decide()` with reasoning | Clears `pending_operator_instruction` |
| `scene` | After successful `vlm_scene()` | Updates `known_scene_handles`, `last_scene_description` |
| `event` | Runtime events (SKILL_STARTED/SUCCEEDED/FAILED, SAFETY_REFLEX, TARGET_ACQUIRED, PERCEPTION_FAILURE) | — |
| `operator` | Operator instruction received | Sets `pending_operator_instruction` |
| `compaction` | ADK compaction detected | — |

**Handoff**: `build_cognitive_state()` produces a `CognitiveState` with journal-derived context (recent decisions, events, goal summary) plus snapshot-derived runtime state (skill phase, outcome, held object) — everything a new backend needs to resume without calling back to the edge runtime.

**Cursor-based sync**: `get_entries_after(cursor)` enables incremental catch-up. `WarmableBackend.caught_up_cursor` tracks how far the standby backend has consumed, enabling bounded batch warm-up during failback.

---

## Appendix A — Naming conventions (stable)

- `PlannerService`
- `TargetPerceptionService`
- `SkillRunnerService`
- `ControlService`
- `SafetyGuard`, `ReflexLayer`
- `SkillOutcomeMonitor`
- `RuntimeStateStore`, `EventBus`

