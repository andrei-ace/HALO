# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Always use Context7 MCP when needing library/API documentation, code generation, setup or configuration steps — no need to ask explicitly.

## Commands

```bash
uv sync --extra dev   # install + dev deps (first time or after pyproject.toml changes)
uv run pytest         # run all tests
uv run pytest tests/test_contracts.py   # run a single test file
uv run pytest -k test_snapshot_ids_increment  # run a single test by name
```

## Project Overview

HALO is a robotic manipulation system. **v1 runs entirely in Isaac Sim/Lab** (no hardware required). Real SO-ARM101 arm support is a later phase. The core design principle is **continuous control decoupled from LLM reasoning**: the robot never pauses motion waiting for the planner. Perception and control are machine-to-machine; numeric control hints never flow through LLM context.

## Repository Structure (planned)

The repo is in early development. See `docs/` for the full architecture and plan:
- `docs/halo_architecture.md` — module boundaries, runtime contracts, dataflows, timing, and code-facing interfaces
- `docs/halo_plan_summary.md` — project plan including Isaac Lab sim-to-real bootstrapping strategy

Planned layout (from architecture spec):
```
halo/
  contracts/        # JSON schemas (enums.json, commands.json, snapshot.json, events.json)
                    # + Python dataclasses/enums mirroring those schemas
  runtime/          # RuntimeStateStore, EventBus, CommandRouter, HALORuntime
  services/
    planner_service/           # LLM agent + tool adapter
    target_perception_service/ # VLM + SAM/Tracker + depth fusion
    skill_runner_service/      # FSM + ACT inference + chunk scheduling
    control_service/           # Real-time executor + safety/reflex
    logger_service/            # Episode capture + dataset writer
  models/
    act/            # Action Chunking Transformer policy
    vlm/            # Vision-language model (Ollama: qwen3-vl:30B)
  configs/
    calib/          # Camera and kinematic calibration
    skills/         # Per-skill FSM profiles and thresholds
    safety/         # Safety guard parameters
  tools/
    ollama_clients/
    zed_capture/    # ZED X scene camera
    uvc_capture/    # Wrist UVC camera
  eval/
    sim/            # Isaac Lab evaluation
    real/           # Real hardware evaluation
```

## Architecture

### Five services, strict role separation

| Service | Rate | Owns |
|---|---|---|
| **PlannerService** | event-driven (30 s watchdog) | Task orchestration, skill selection, retries, high-level recovery. LLM: `gpt-oss:20B` via Ollama. Tick fires on urgent events (SKILL_SUCCEEDED/FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE); watchdog ensures a tick every 30 s even if no events arrive. Ticks are serialized — decide_fn is awaited before the next event is processed. |
| **TargetPerceptionService** | 10–30 Hz (fast loop), async (VLM) | Target discovery/tracking, fused target hints, validity/confidence, failure codes. VLM: `qwen3-vl:30B` via Ollama (scene camera only). SAM/SAM2 for segmentation, fast tracker for steady-state, ZED X depth fusion. |
| **SkillRunnerService** | 10–20 Hz (ACT inference) | Pick FSM, phase transitions, ACT chunk buffering, buffer trimming on phase switch, fast success/failure checks. |
| **ControlService** | 50–100 Hz | Real-time action streaming, smoothing, clamps (vel/acc/jerk), safety interlocks. Never waits on LLM or VLM. |
| **SafetyGuard / ReflexLayer** | Hard real-time | Joint/workspace/velocity limits, immediate stop/retract/open-gripper overrides. LLM cannot bypass. |

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
3. Every planner command carries a `command_id` (UUID) and `precondition_snapshot_id`; the router must enforce idempotency and reject stale preconditions.
4. VLM reacquire runs **asynchronously** — it is never on the critical path of the 10–30 Hz hint-publish loop.
5. On phase transition, **trim the ACT buffer** to ~50–100 ms to avoid executing old-phase tail actions.

### HALORuntime (`halo/runtime/runtime.py`)
Top-level entry point. Owns `RuntimeStateStore`, `EventBus`, and `CommandRouter`. Exposes the two planner-facing APIs: `get_latest_runtime_snapshot(arm_id)` and `submit_command(cmd)`.

### RuntimeStateStore / EventBus
Single source of truth (transport TBD: ROS2 topics, ZeroMQ, Redis, shared memory). Partitioned by `arm_id` from day one.

### Pick Skill FSM states
`IDLE → ACQUIRE_TARGET → APPROACH → PREGRASP_ALIGN → GRASP_CLOSE → [VERIFY_GRASP] → LIFT → SUCCESS_PICK`
Recovery: `REACQUIRE_TARGET`, `GRASP_RECOVER`, `FAIL(<reason>)`

`GRASP_CLOSE` is triggered **deterministically** (distance + persistence threshold), never by the planner.

### ACT action space
`[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]` in the EE frame, per-timestep servo increments.
Do **not** integrate and play back whole chunks open-loop. Temporal ensembling blends overlapping deltas per-timestep before IK/OSC mapping.

### Planner snapshot fields (compact; no raw telemetry)
`snapshot_id`, `arm_id`, `skill/phase`, `target` (hint_valid, confidence, obs_age_ms, delta_xyz_ee, distance_m), `perception` (tracking_status, failure_code), `act` (buffer_fill_ms, buffer_low), `progress`, `outcome`, `safety`, `command_ack`, `recent_events` (ring of 3–8).

### Stable enums (define in `contracts/enums`)
- Perception failure codes: `OK`, `OCCLUDED`, `OUT_OF_VIEW`, `DEPTH_INVALID`, `MULTIPLE_CANDIDATES`, `CALIB_INVALID`, `TRACK_JUMP_REJECTED`, `REACQUIRE_FAILED`
- Skill failure codes: `TIMEOUT`, `NO_PROGRESS`, `NO_GRASP`, `DROP_DETECTED`, `PLACE_MISS`, `UNSAFE_ABORT`
- Safety reflex reasons: `JOINT_LIMIT`, `WORKSPACE_LIMIT`, `COLLISION_RISK`, `OVERCURRENT`, `ESTOP`
- Phase IDs: `0 RESET`, `1 APPROACH_PREGRASP`, `2 ALIGN`, `3 DESCEND_GRASP`, `4 CLOSE`, `5 LIFT`, `6 VERIFY_GRASP`, `7 TRANSIT_PREPLACE`, `8 DESCEND_PLACE`, `9 OPEN`, `10 RETREAT`, `11 DONE`, `20-22 RECOVER_*`

## Hardware (real SO-ARM101 phase — not v1)

v1 uses Isaac Sim/Lab only. The real hardware target is SO-ARM101 with:
- **Scene camera**: ZED X (VLM grounding, global target discovery, depth-based 3D, `T_base<-scene_cam`)
- **Wrist camera**: 1080p USB2 UVC (ACT observation, local visual servoing, `T_ee<-wrist_cam`)
- **LLM/VLM**: local via Ollama — planner uses `gpt-oss:20B`, perception uses `qwen3-vl:30B`

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
