# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Always use Context7 MCP when needing library/API documentation, code generation, setup or configuration steps — no need to ask explicitly.
- Always run `make ruff` before committing to ensure code is linted and formatted.

## Commands

```bash
uv sync --extra dev --extra sim       # install deps + MuJoCo (first time or after pyproject.toml changes)
uv run python -m pytest               # run all unit tests
uv run python -m pytest tests/test_contracts.py   # run a single test file
uv run python -m pytest -k test_snapshot_ids_increment  # run a single test by name

make install           # install deps + MuJoCo (uv sync --extra dev --extra sim)
make ruff              # lint + format (ruff check --fix + ruff format); run before every commit
make test-sim          # run mujoco_sim tests (requires make install-sim)
make tui-mock          # launch TUI in mock mode (no Ollama needed)
make tui-live-videoloop # launch TUI with video loop source (requires Ollama)
make tui-live-mujoco   # launch TUI with MuJoCo scene camera (requires Ollama + MuJoCo; run make install-sim first)
make generate-episodes # generate teacher episodes (EPISODES=10 EPISODE_DIR=episodes SEED_BASE=0; requires make install-sim)
make test-integration  # run LLM integration tests (requires Ollama); saves results to integration/runs/YYYYMMDD_HHMMSS/
```

**Note:** `uv run pytest` fails if `uv sync --extra dev` hasn't been run yet; use `uv run python -m pytest` to be safe.

**Integration tests** require `uv sync --extra planner` and a running Ollama instance with `gpt-oss:20b` loaded. They are auto-skipped if Ollama is unreachable. Configure via env vars: `HALO_OLLAMA_URL` (default `http://localhost:11434`) and `HALO_MODEL_NAME` (default `gpt-oss:20b`).

## Implementation Status

All v0 backbone services are implemented and tested:

| Layer | Status |
|---|---|
| contracts (enums, snapshots, commands, events, actions) | ✅ done |
| runtime (RuntimeStateStore, EventBus, CommandRouter, HALORuntime) | ✅ done |
| ControlService + TemporalEnsemblingBuffer + SafetyGuard | ✅ done |
| SkillRunnerService + PickFSM | ✅ done |
| PlannerService | ✅ done |
| TargetPerceptionService (mock + VLM pipeline) | ✅ done |
| PlannerAgent (ADK ReAct + tools) | ✅ done |
| TUI (`halo/tui/app.py`) | ✅ done |
| RunLogger + observability | ✅ done |
| Integration tests (`integration/`) | ✅ done (requires Ollama) |
| Bridge adapters (`halo/bridge/`) | ✅ done (ZMQ 2-channel: SimClient, SimSource, transforms) |
| Cognitive backend switching (`halo/cognitive/`) | ✅ done — Switchboard, LeaseManager, ContextStore, failover/failback, epoch+token gating, warm-up handoff |
| MuJoCo sim (`mujoco_sim/`) | ✅ done — SO-101 + raw MuJoCo (env, dataset, teacher, IK, autonomous SimServer); PR4-6 pending (phase FSM, VCR, annotation) |
| Isaac Lab extension (`sim/`) | 📋 planned (after MuJoCo pipeline validated) |

The TUI supports multiple modes:
- **Mock mode** (`make tui-mock`): static fixture data, no services needed.
- **Live local** (`make tui-live-videoloop` or `make tui-live-mujoco`): Ollama planner + VLM.
- **Live cloud** (`make tui-live-cloud`): Gemini Live API (bidirectional audio + text) via Switchboard. Requires `GOOGLE_API_KEY`.
- **Live remote cloud** (`make tui-live-remote-cloud`): HTTP client to Cloud Run via Switchboard. Requires `HALO_CLOUD_URL`.

All cloud modes use the Switchboard with LeaseManager for split-brain prevention, automatic failover (3 consecutive failures → switch), and warm-up handoff on failback. Each session writes a JSONL log to `runs/YYYYMMDD_HHMMSS_<arm_id>.jsonl` + `events.jsonl` (via `halo/tui/run_logger.py`). No env resets between skills in live-mujoco mode.

To regenerate the screenshot: `uv run python -m halo.tui.app --screenshot runs/halo_tui.svg`

## Project Overview

HALO is a robotic manipulation system with a **three-phase sim strategy**: (1) **MuJoCo + SO-101** (current) for teacher demos, ACT training, and closed-loop eval; (2) **Isaac Lab** (future) for GPU-accelerated parallel envs and domain randomization at scale; (3) **Real SO-ARM101 hardware** (later). The core design principle is **continuous control decoupled from LLM reasoning**: the robot never pauses motion waiting for the planner. Perception and control are machine-to-machine; numeric control hints never flow through LLM context.

## Repository Structure

```
halo/
  contracts/        # enums.py, snapshots.py, commands.py, events.py, actions.py
                    # + JSON schemas: enums.json, commands.json, events.json, snapshot.json
  runtime/          # state_store.py, event_bus.py, command_router.py, runtime.py
  cognitive/        # backend switching: config.py, backend.py, switchboard.py, lease.py,
                    # context_store.py, local_backend.py, cloud_backend.py, remote_backend.py,
                    # live_session.py, audio_io.py
  bridge/            # ZMQ 2-channel bridge to MuJoCo sim server
                      # __init__.py (BridgeTransportError), config.py (SimBridgeConfig),
                      # sim_client.py (SimClient), sim_source.py (SimSource), transforms.py,
                      # sim_tracker_service.py
  services/                    # each service has its own CLAUDE.md with detailed docs
    control_service/           # config.py, action_buffer.py, te_buffer.py, safety_guard.py, service.py
    skill_runner_service/      # config.py, fsm.py, service.py
    planner_service/           # config.py, snapshot_serializer.py, tools.py, agent.py, service.py
    target_perception_service/ # config.py, service.py, vlm_parser.py, ollama_vlm_fn.py, mock_fns.py, handle_match.py, tracker_fn.py, frame_buffer.py
  tui/
    app.py          # Textual TUI — mock + live modes
    run_logger.py   # RunLogger: writes JSONL session logs to runs/
  models/           # (planned) act/, vlm/
  configs/
    planner/        # system_prompt.md, skills/pick.md, skills/place.md
    perception/     # scene_analysis.md (VLM prompt for qwen2.5vl)
    calib/          # (planned)
    skills/         # (planned)
    safety/         # (planned)
  tools/            # (planned) ollama_clients/, zed_capture/, uvc_capture/
  eval/             # (planned) sim/, real/
mujoco_sim/         # MuJoCo + SO-101 sim (env, dataset, teacher, autonomous SimServer; see mujoco_sim/CLAUDE.md)
sim/                # Isaac Lab extension (planned — see sim/README.md)
docs/
  halo_architecture.md   # module boundaries, runtime contracts, dataflows, timing
  halo_plan_summary.md   # project plan including Isaac Lab sim-to-real strategy
  data/                  # gitignored; video.mp4 for video capture simulation
runs/               # live TUI session logs (JSONL, one file per session; git-ignored except .gitkeep)
tests/
integration/        # LLM integration tests (require Ollama)
  conftest.py       # Ollama health-check; auto-skips if model unavailable
  runs/             # timestamped result folders from make test-integration
```

## Architecture

### Five services, strict role separation

| Service | Rate | Owns |
|---|---|---|
| **PlannerService** | event-driven (30 s watchdog) | Task orchestration, skill selection, retries, high-level recovery. LLM: `gpt-oss:20b` via Ollama. Tick fires on urgent events (SKILL_SUCCEEDED/FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE, SCENE_DESCRIBED, TARGET_ACQUIRED, COMMAND_REJECTED); watchdog ensures a tick every 30 s even if no events arrive. Ticks are serialized — decide_fn is awaited before the next event is processed. |
| **TargetPerceptionService** | 10–30 Hz (fast loop), async (VLM) | Target discovery/tracking, fused target hints, validity/confidence, failure codes. VLM: `qwen2.5vl` via Ollama (scene camera only, async reacquire). Includes VLM parser (`vlm_parser.py`), Ollama VLM client (`ollama_vlm_fn.py`), mock fns (`mock_fns.py`). SAM/SAM2 for segmentation, fast tracker for steady-state, ZED X depth fusion (planned). |
| **SkillRunnerService** | 10–20 Hz (ACT inference) | Pick FSM, phase transitions, ACT chunk buffering, buffer trimming on phase switch, fast success/failure checks. Dual-mode: ACT (chunk_fn/push_fn) or Sim (start_pick_fn/sim_phase_fn). |
| **ControlService** | 50–100 Hz | Real-time action streaming, temporal ensembling, per-timestep delta clamping, safety interlocks. Never waits on LLM or VLM. |
| **SafetyGuard / ReflexLayer** | Hard real-time | v0: per-timestep linear/angular delta limits + hint freshness gating. Planned: workspace AABB, vel/accel/jerk, collision checks. LLM cannot bypass. |

### Dataflows

**Control path (machine-to-machine, no LLM):**
```
Cameras + RobotState → TargetPerceptionService → target_hint_vec → RuntimeStateStore
  → SkillRunnerService → ACT → action_chunks → ControlService (50–100Hz) → Robot
```

**Decision path (LLM, low frequency):**
```
RuntimeStateStore → get_latest_runtime_snapshot() → PlannerService → async commands → RuntimeStateStore
```

### Key invariants to maintain
1. SkillRunner reads `target_hint_vec` **directly from runtime state**, not through the Planner.
2. Planner sees **exactly one** snapshot: the latest. Middleware must **replace** (not append) the prior snapshot in LLM context.
3. Every mutating planner command carries a `command_id` (UUID) and `precondition_snapshot_id`; the router must enforce idempotency and reject stale preconditions. Stateless commands (`describe_scene`, `track_object`) set `precondition_snapshot_id = None` to avoid premature rejection.
4. VLM reacquire runs **asynchronously** — it is never on the critical path of the 10–30 Hz hint-publish loop.
5. On phase transition, **trim the ACT buffer** to ~50–100 ms to avoid executing old-phase tail actions.
6. When a `LeaseManager` is active, every command must carry both `epoch` and `lease_token`. Commands without them (or with stale values) are rejected by the `CommandRouter`.

### HALORuntime (`halo/runtime/runtime.py`)
Top-level entry point. Owns `RuntimeStateStore`, `EventBus`, and `CommandRouter`. Exposes the two planner-facing APIs: `get_latest_runtime_snapshot(arm_id)` and `submit_command(cmd)`.

### RuntimeStateStore / EventBus
Single source of truth (transport TBD: ROS2 topics, ZeroMQ, Redis, shared memory). Partitioned by `arm_id` from day one.

### Pick Skill FSM states
`IDLE → SELECT_GRASP → PLAN_APPROACH → MOVE_PREGRASP → VISUAL_ALIGN → EXECUTE_APPROACH → CLOSE_GRIPPER → VERIFY_GRASP → LIFT → DONE`
Recovery: `RECOVER_RETRY_APPROACH`, `RECOVER_REGRASP`, `RECOVER_ABORT`

`CLOSE_GRIPPER` is triggered **deterministically** (distance < threshold held for `grasp_persistence_ms`), never by the planner.

Wrist camera active phases: `VISUAL_ALIGN`, `EXECUTE_APPROACH`, `CLOSE_GRIPPER`, `VERIFY_GRASP`, `LIFT` (defined as `WRIST_ACTIVE_PHASES` in `contracts/enums.py`).

### ACT action space
**HALO core (runtime/bridge):** `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]` — 7D EE-frame deltas, per-timestep servo increments. Temporal ensembling blends overlapping deltas per-timestep before IK/OSC mapping.

**MuJoCo sim (`mujoco_sim/`):** `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` — 6D joint-position targets for SO-101 (5 arm DOF + 1 gripper DOF). Written directly to `data.ctrl[:]`.

### Planner snapshot fields (compact; no raw telemetry)
`snapshot_id`, `arm_id`, `skill/phase`, `target` (hint_valid, confidence, obs_age_ms, delta_xyz_ee, distance_m, center_px, bbox_xywh — all normalised 0..1), `perception` (tracking_status, failure_code), `act` (buffer_fill_ms, buffer_low, wrist_enabled), `progress`, `outcome`, `safety`, `command_acks`, `recent_events` (ring of 8), `held_object_handle` (str|None — which object is in the gripper after a successful pick).

### Stable enums (define in `contracts/enums`)
- Perception failure codes: `OK`, `OCCLUDED`, `OUT_OF_VIEW`, `DEPTH_INVALID`, `MULTIPLE_CANDIDATES`, `CALIB_INVALID`, `TRACK_JUMP_REJECTED`, `REACQUIRE_FAILED`
- Skill failure codes: `TIMEOUT`, `NO_PROGRESS`, `NO_GRASP`, `DROP_DETECTED`, `PLACE_MISS`, `PERCEPTION_LOST`, `UNSAFE_ABORT`
- Safety reflex reasons: `JOINT_LIMIT`, `WORKSPACE_LIMIT`, `COLLISION_RISK`, `OVERCURRENT`, `ESTOP`
- Phase IDs: `0 IDLE`, `1 SELECT_GRASP`, `2 PLAN_APPROACH`, `3 MOVE_PREGRASP`, `4 VISUAL_ALIGN`, `5 EXECUTE_APPROACH`, `6 CLOSE_GRIPPER`, `7 VERIFY_GRASP`, `8 LIFT`, `9 DONE`, `30-33 PLACE_*`, `50-52 RECOVER_*`

## Hardware (real SO-ARM101 phase — phase 3)

Current sim work uses MuJoCo + SO-101 (phase 1), then Isaac Lab (phase 2). The real hardware target is SO-ARM101 with:
- **Scene camera**: ZED X (VLM grounding, global target discovery, depth-based 3D, `T_base<-scene_cam`)
- **Wrist camera**: 1080p USB2 UVC (ACT observation, local visual servoing, `T_ee<-wrist_cam`)
- **LLM/VLM**: local via Ollama — planner uses `gpt-oss:20b`, perception uses `qwen2.5vl:3b`

## Timing Budgets

- Fast perception loop (camera frame → `target_hint_vec` published): **≤80–120 ms**
- VLM reacquire: async, **hundreds of ms to several seconds** — never on critical path
- ACT chunk horizon: **200–500 ms** for moving targets; buffer fill target **150–300 ms**
- v0 MuJoCo: 10 Hz control, 10-step chunks (1 s horizon) for debugging simplicity

## Sim strategy (three phases)

**Phase 1 — MuJoCo + SO-101 (current):** Single-env teacher demos with raw MuJoCo, 6D joint-position actions, damped least-squares IK. Autonomous SimServer runs physics and trajectories; HALO runtime triggers via `start_pick` and monitors progress via telemetry. ACT training pipeline, closed-loop eval. SO-101 env, HDF5 episodes, trajectory-planned teacher with 5 s stabilization + phase tracking. See `mujoco_sim/CLAUDE.md`.

**Phase 2 — Isaac Lab (future):** GPU-accelerated parallel envs (64 envs on A6000), domain randomization at scale, sim-to-real transfer. See `sim/README.md`.

**Phase 3 — Real SO-ARM101 hardware (later):** Same dataset schema and action space, swap sensor sources and controller.

Key invariants across all phases:
- Dataset schema must stay **identical** between sim and real (same action space, chunking, observation keys)
- Train/val/test splits by **episode seed / object placement**, not by timestep
- Verify replay parity before training: recorded actions re-executed in sim should reproduce the episode

## Calibration Transforms (real hardware phase)

- `T_base<-scene_cam` (ZED X)
- `T_ee<-wrist_cam` (wrist UVC)
- `T_base<-ee` (kinematics)

Every published hint must carry: `hint_ts`, `robot_state_ts`, `time_skew_ms`, `obs_age_ms`. If either exceeds threshold → `hint_valid = false`, force REACQUIRE/HOLD. In sim these are available from ground-truth state but the same timestamp discipline applies.
