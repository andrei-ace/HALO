# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Always use Context7 MCP when needing library/API documentation, code generation, setup or configuration steps — no need to ask explicitly.
- Always run `make ruff` before committing to ensure code is linted and formatted.

## Commands

```bash
uv sync --extra dev                   # install + dev deps (first time or after pyproject.toml changes)
uv run python -m pytest               # run all unit tests
uv run python -m pytest tests/test_contracts.py   # run a single test file
uv run python -m pytest -k test_snapshot_ids_increment  # run a single test by name

make ruff              # lint + format (ruff check --fix + ruff format); run before every commit
make tui-mock          # launch TUI in mock mode (no Ollama needed)
make tui-live          # launch TUI wired to HALORuntime + PlannerAgent (requires Ollama)
make test-integration  # run LLM integration tests (requires Ollama); saves results to integration/runs/YYYYMMDD_HHMMSS/
```

**Note:** `uv run pytest` fails if `uv sync --extra dev` hasn't been run yet; use `uv run python -m pytest` to be safe.

**Integration tests** require `uv sync --extra planner` and a running Ollama instance with `gpt-oss` loaded. They are auto-skipped if Ollama is unreachable. Configure via env vars: `HALO_OLLAMA_URL` (default `http://localhost:11434`) and `HALO_MODEL_NAME` (default `gpt-oss`).

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
| PlannerAgent (LangGraph ReAct + tools) | ✅ done |
| TUI (`halo/tui/app.py`) | ✅ done |
| RunLogger + observability | ✅ done |
| Integration tests (`integration/`) | ✅ done (requires Ollama) |

The TUI supports two modes:
- **Mock mode** (`make tui-mock`): static fixture data, no services needed.
- **Live mode** (`make tui-live`): wired to real `HALORuntime` + `PlannerAgent.decide()`. Talk panel sends operator messages to the LLM; abort button submits `ABORT_SKILL` commands; results shown in the ActionsPanel. Each session writes a JSONL log to `runs/YYYYMMDD_HHMMSS_<arm_id>.jsonl` (via `halo/tui/run_logger.py`).

To regenerate the screenshot: `uv run python -m halo.tui.app --screenshot runs/halo_tui.svg`

## Project Overview

HALO is a robotic manipulation system. **v1 runs entirely in Isaac Sim/Lab** (no hardware required). Real SO-ARM101 arm support is a later phase. The core design principle is **continuous control decoupled from LLM reasoning**: the robot never pauses motion waiting for the planner. Perception and control are machine-to-machine; numeric control hints never flow through LLM context.

## Repository Structure

```
halo/
  contracts/        # enums.py, snapshots.py, commands.py, events.py, actions.py
                    # + JSON schemas: enums.json, commands.json, events.json, snapshot.json
  runtime/          # state_store.py, event_bus.py, command_router.py, runtime.py
  services/                    # each service has its own CLAUDE.md with detailed docs
    control_service/           # config.py, action_buffer.py, te_buffer.py, safety_guard.py, service.py
    skill_runner_service/      # config.py, fsm.py, service.py
    planner_service/           # config.py, snapshot_serializer.py, tools.py, agent.py, service.py
    target_perception_service/ # config.py, service.py, vlm_parser.py, ollama_vlm_fn.py, mock_fns.py
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
| **PlannerService** | event-driven (30 s watchdog) | Task orchestration, skill selection, retries, high-level recovery. LLM: `gpt-oss` via Ollama. Tick fires on urgent events (SKILL_SUCCEEDED/FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE, SCENE_DESCRIBED, TARGET_ACQUIRED, COMMAND_REJECTED); watchdog ensures a tick every 30 s even if no events arrive. Ticks are serialized — decide_fn is awaited before the next event is processed. |
| **TargetPerceptionService** | 10–30 Hz (fast loop), async (VLM) | Target discovery/tracking, fused target hints, validity/confidence, failure codes. VLM: `qwen2.5vl` via Ollama (scene camera only, async reacquire). Includes VLM parser (`vlm_parser.py`), Ollama VLM client (`ollama_vlm_fn.py`), mock fns (`mock_fns.py`). SAM/SAM2 for segmentation, fast tracker for steady-state, ZED X depth fusion (planned). |
| **SkillRunnerService** | 10–20 Hz (ACT inference) | Pick FSM, phase transitions, ACT chunk buffering, buffer trimming on phase switch, fast success/failure checks. |
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

### HALORuntime (`halo/runtime/runtime.py`)
Top-level entry point. Owns `RuntimeStateStore`, `EventBus`, and `CommandRouter`. Exposes the two planner-facing APIs: `get_latest_runtime_snapshot(arm_id)` and `submit_command(cmd)`.

### RuntimeStateStore / EventBus
Single source of truth (transport TBD: ROS2 topics, ZeroMQ, Redis, shared memory). Partitioned by `arm_id` from day one.

### Pick Skill FSM states
`RESET → APPROACH_PREGRASP → ALIGN → DESCEND_GRASP → CLOSE → [VERIFY_GRASP] → LIFT → DONE`
Recovery: `RECOVER_RETRY_APPROACH`, `RECOVER_RETRY_DESCEND`, `RECOVER_REGRASP`

`CLOSE` is triggered **deterministically** (distance < threshold held for `grasp_persistence_ms`), never by the planner.

### ACT action space
`[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]` in the EE frame, per-timestep servo increments.
Do **not** integrate and play back whole chunks open-loop. Temporal ensembling blends overlapping deltas per-timestep before IK/OSC mapping.

### Planner snapshot fields (compact; no raw telemetry)
`snapshot_id`, `arm_id`, `skill/phase`, `target` (hint_valid, confidence, obs_age_ms, delta_xyz_ee, distance_m), `perception` (tracking_status, failure_code), `act` (buffer_fill_ms, buffer_low), `progress`, `outcome`, `safety`, `command_acks`, `recent_events` (ring of 8).

### Stable enums (define in `contracts/enums`)
- Perception failure codes: `OK`, `OCCLUDED`, `OUT_OF_VIEW`, `DEPTH_INVALID`, `MULTIPLE_CANDIDATES`, `CALIB_INVALID`, `TRACK_JUMP_REJECTED`, `REACQUIRE_FAILED`
- Skill failure codes: `TIMEOUT`, `NO_PROGRESS`, `NO_GRASP`, `DROP_DETECTED`, `PLACE_MISS`, `UNSAFE_ABORT`
- Safety reflex reasons: `JOINT_LIMIT`, `WORKSPACE_LIMIT`, `COLLISION_RISK`, `OVERCURRENT`, `ESTOP`
- Phase IDs: `0 RESET`, `1 APPROACH_PREGRASP`, `2 ALIGN`, `3 DESCEND_GRASP`, `4 CLOSE`, `5 LIFT`, `6 VERIFY_GRASP`, `7 TRANSIT_PREPLACE`, `8 DESCEND_PLACE`, `9 OPEN`, `10 RETREAT`, `11 DONE`, `20-22 RECOVER_*`

## Hardware (real SO-ARM101 phase — not v1)

v1 uses Isaac Sim/Lab only. The real hardware target is SO-ARM101 with:
- **Scene camera**: ZED X (VLM grounding, global target discovery, depth-based 3D, `T_base<-scene_cam`)
- **Wrist camera**: 1080p USB2 UVC (ACT observation, local visual servoing, `T_ee<-wrist_cam`)
- **LLM/VLM**: local via Ollama — planner uses `gpt-oss`, perception uses `qwen2.5vl`

## Timing Budgets

- Fast perception loop (camera frame → `target_hint_vec` published): **≤80–120 ms**
- VLM reacquire: async, **hundreds of ms to several seconds** — never on critical path
- ACT chunk horizon: **200–500 ms** for moving targets; buffer fill target **150–300 ms**
- v0 Isaac Lab: 10 Hz control, 10-step chunks (1 s horizon) for debugging simplicity

## v1: Isaac Sim/Lab bootstrapping

- Teacher: analytic controller (IK + motion generation) generating 10k–50k demo episodes
- Dataset schema must stay **identical** between sim and real (same action space, chunking, observation keys)
- Train/val/test splits by **episode seed / object placement**, not by timestep
- Verify replay parity before training: recorded actions re-executed in sim should reproduce the episode

## Calibration Transforms (real hardware phase)

- `T_base<-scene_cam` (ZED X)
- `T_ee<-wrist_cam` (wrist UVC)
- `T_base<-ee` (kinematics)

Every published hint must carry: `hint_ts`, `robot_state_ts`, `time_skew_ms`, `obs_age_ms`. If either exceeds threshold → `hint_valid = false`, force REACQUIRE/HOLD. In sim these are available from ground-truth state but the same timestamp discipline applies.
