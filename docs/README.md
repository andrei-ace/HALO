# HALO Architecture

HALO is a robotic manipulation system built around **continuous control decoupled from LLM reasoning** — the robot never pauses motion waiting for the planner.

The project is developed in phases: **v0** implements the full software backbone (services, contracts, planner agent, TUI) tested with mocks. **v1** wires everything to **Isaac Sim/Lab** for end-to-end sim training and evaluation. Real SO-ARM101 hardware is a later phase.

---

## System Overview

Five services with strict role separation, coordinated through a shared runtime:

| Service | Rate | Role |
|---|---|---|
| **PlannerService** | event-driven (30 s watchdog) | Task orchestration, skill selection, retries, recovery |
| **TargetPerceptionService** | 10–30 Hz + async VLM | Target discovery/tracking, fused hints, failure codes |
| **SkillRunnerService** | 10–20 Hz | Pick FSM, phase transitions, ACT chunk buffering |
| **ControlService** | 50–100 Hz (target); 10 Hz in v0 sim | Real-time action streaming, temporal ensembling, safety |
| **SafetyGuard** | Hard real-time | Delta limits, hint freshness gating, reflexes |

```mermaid
graph TB
    subgraph Runtime["HALORuntime"]
        SS["RuntimeStateStore"]
        EB["EventBus"]
        CR["CommandRouter"]
    end

    PS["PlannerService\n(LLM: gpt-oss)"]
    TPS["TargetPerceptionService\n(VLM: qwen2.5vl)"]
    SRS["SkillRunnerService\n(Pick FSM + ACT)"]
    CS["ControlService\n(50-100 Hz)"]
    SG["SafetyGuard\n(Reflex Layer)"]

    PS -->|commands| CR
    CR -->|acks + events| EB
    PS -->|read snapshot| SS

    TPS -->|target_hint_vec| SS
    SRS -->|action_chunks| CS
    SRS -->|read hints| SS
    CS -->|clamped actions| Robot["Robot"]
    SG -->|reflex override| CS
    SG -->|reflex events| EB
    EB -->|urgent events| PS

    style Runtime fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style PS fill:#0f3460,stroke:#16213e,color:#e0e0e0
    style TPS fill:#533483,stroke:#16213e,color:#e0e0e0
    style SRS fill:#e94560,stroke:#16213e,color:#e0e0e0
    style CS fill:#c84b31,stroke:#16213e,color:#e0e0e0
    style SG fill:#7b2d26,stroke:#16213e,color:#e0e0e0
```

---

## Dataflows

The system has two independent paths — a high-frequency **control path** (machine-to-machine, no LLM) and a low-frequency **decision path** (LLM-driven).

### Control Path

Numeric control hints flow machine-to-machine and never enter LLM context.

```mermaid
flowchart LR
    CAM["Cameras\n+ RobotState"] --> TPS["TargetPerception\nService"]
    TPS -->|"target_hint_vec\n(robot frame + EE deltas)"| SS["RuntimeState\nStore"]
    SS -->|"read hints\ndirectly"| SRS["SkillRunner\nService"]
    SRS -->|"ACT inference\n(10-20 Hz)"| AC["action_chunks"]
    AC --> CS["ControlService\n(50-100 Hz)"]
    CS -->|"clamped deltas\n→ IK/OSC"| R["Robot"]
```

### Decision Path

The planner reads compact snapshots and issues async commands. It never blocks the control loop.

```mermaid
flowchart LR
    SS["RuntimeState\nStore"] -->|"get_latest_runtime_snapshot()"| PS["PlannerService\n(LLM)"]
    PS -->|"async commands\n(start_skill, abort, etc.)"| CR["Command\nRouter"]
    CR -->|"acks + state updates"| SS
    CR -->|"events"| EB["EventBus"]
    EB -->|"urgent events\n(SKILL_FAILED, etc.)"| PS
```

---

## Command Protocol

Every mutating command carries a `command_id` (UUID) and `precondition_snapshot_id`. The router enforces idempotency and uses **strict optimistic concurrency**: `precondition_snapshot_id` must exactly equal the current `snapshot_id` (not >=). If the world has moved on, the command is rejected and the planner must re-read and retry.

```mermaid
sequenceDiagram
    participant P as PlannerService
    participant CR as CommandRouter
    participant SS as StateStore
    participant EB as EventBus

    P->>CR: submit_command(start_skill, snapshot_id=42)

    Note over CR: 1. Idempotency check (first)
    alt Duplicate command_id
        CR-->>P: ALREADY_APPLIED (short-circuit)
    else New command_id
        Note over CR,SS: 2. Precondition check
        CR->>SS: latest snapshot_id == 42?
        alt Exact match
            CR->>SS: apply command
            CR->>EB: publish COMMAND_ACCEPTED
            EB-->>P: event in next snapshot
        else Snapshot has advanced (stale)
            CR->>EB: publish COMMAND_REJECTED (REJECTED_STALE)
            EB-->>P: event in next snapshot
        end
    end
```

### Planner Tools

| Tool | Precondition | Purpose |
|---|---|---|
| `start_skill(skill, target, options)` | snapshot_id | Launch a skill (pick, place) |
| `abort_skill(skill_run_id, reason)` | snapshot_id | Abort a running skill |
| `override_target(skill_run_id, handle)` | snapshot_id | Retarget mid-skill |
| `describe_scene(reason)` | None (stateless) | Trigger async VLM scene analysis |
| `track_object(target_handle)` | None (stateless) | Set perception tracking target |

---

## Pick Skill FSM

The SkillRunnerService drives a deterministic FSM. Phase transitions are fast and local — the planner only starts/aborts skills, never times micro-actions.

```mermaid
stateDiagram-v2
    [*] --> RESET
    RESET --> APPROACH_PREGRASP

    APPROACH_PREGRASP --> ALIGN : position reached
    APPROACH_PREGRASP --> RECOVER_RETRY_APPROACH : timeout / collision

    ALIGN --> DESCEND_GRASP : aligned

    DESCEND_GRASP --> CLOSE : distance < threshold\nheld for grasp_persistence_ms
    DESCEND_GRASP --> RECOVER_RETRY_DESCEND : timeout / no progress

    CLOSE --> VERIFY_GRASP : dwell complete\n(if verify enabled)
    CLOSE --> LIFT : dwell complete\n(if verify skipped)

    VERIFY_GRASP --> LIFT : grasp confirmed
    VERIFY_GRASP --> RECOVER_REGRASP : grasp failed

    LIFT --> DONE : lift complete

    RECOVER_RETRY_APPROACH --> APPROACH_PREGRASP : retry
    RECOVER_RETRY_DESCEND --> APPROACH_PREGRASP : retry
    RECOVER_REGRASP --> APPROACH_PREGRASP : retry

    DONE --> [*]
```

**Key invariant:** `CLOSE` is triggered deterministically when `distance < grasp_distance_threshold_m` held for `grasp_persistence_ms`. The planner never commands "close gripper now".

---

## TargetPerceptionService

Two loops: a fast tracking loop (10–30 Hz) and an async VLM loop for scene analysis/reacquisition.

```mermaid
flowchart TB
    subgraph Fast["Fast Loop (10-30 Hz, ≤80-120ms)"]
        OBS["observe_fn\n(tracker + depth)"] --> PG["Plausibility\nGates"]
        PG -->|valid| PUB["Publish\ntarget_hint_vec"]
        PG -->|invalid| INV["hint_valid = false\n→ HOLD / REACQUIRE"]
    end

    subgraph Async["Async VLM (0.5-5s, off critical path)"]
        VLM["VLM\n(qwen2.5vl)"] --> PARSE["vlm_parser\n→ VlmScene"]
        PARSE --> SEED["Store _vlm_seed"]
    end

    SEED -.->|"consumed when\nobserve returns None"| OBS
    INV -.->|"trigger reacquire"| VLM

    CAM["Scene Camera"] --> VLM
    CAM2["Cameras + Robot State"] --> OBS
```

### Perception Failure Codes

`OK` · `OCCLUDED` · `OUT_OF_VIEW` · `DEPTH_INVALID` · `MULTIPLE_CANDIDATES` · `CALIB_INVALID` · `TRACK_JUMP_REJECTED` · `REACQUIRE_FAILED`

---

## ControlService & Safety

The ControlService applies temporal ensembling to blend overlapping action chunks into smooth per-timestep deltas. Target rate is 50–100 Hz for real hardware; v0 Isaac Lab uses 10 Hz for debugging simplicity.

```mermaid
flowchart LR
    SRS["SkillRunner\naction_chunks"] --> TEB["Temporal\nEnsembling\nBuffer"]
    TEB -->|"blended delta\nper timestep"| SG["SafetyGuard\n(clamp + freshness)"]
    SG -->|"safe action"| IK["IK / OSC\nMapping"]
    IK --> Robot["Robot"]

    SG -->|"reflex triggered"| REF["Reflex\n(stop/retract/open)"]
    REF --> Robot
    REF -->|"SAFETY_REFLEX_TRIGGERED"| EB["EventBus"]
```

### Safety Guards (v0)

- Per-timestep linear delta limit (`max_linear_delta_m`)
- Per-timestep angular delta limit (`max_angular_delta_rad`)
- Hint freshness gating (`obs_age_ms`, `time_skew_ms` thresholds)
- Reflex: immediate stop/retract/open-gripper on unsafe conditions

The LLM cannot bypass safety guards.

---

## Event Flow

Services communicate asynchronously through the EventBus. The planner wakes on urgent events.

```mermaid
flowchart TB
    SRS["SkillRunner"] -->|"SKILL_SUCCEEDED\nSKILL_FAILED\nPHASE_ENTER"| EB["EventBus"]
    TPS["Perception"] -->|"PERCEPTION_FAILURE\nSCENE_DESCRIBED\nTARGET_ACQUIRED"| EB
    SG["SafetyGuard"] -->|"SAFETY_REFLEX_TRIGGERED\nSAFETY_RECOVERED"| EB
    CR["CommandRouter"] -->|"COMMAND_ACCEPTED\nCOMMAND_REJECTED"| EB

    EB -->|"urgent events\nwake planner"| PS["PlannerService"]
    EB -->|"30s watchdog\n(if no events)"| PS

    PS -->|"reads latest\nsnapshot"| SS["StateStore"]
    SS -->|"snapshot includes\nrecent_events ring"| PS
```

### Urgent Events (wake PlannerService)

`SKILL_SUCCEEDED` · `SKILL_FAILED` · `SAFETY_REFLEX_TRIGGERED` · `PERCEPTION_FAILURE` · `SCENE_DESCRIBED` · `TARGET_ACQUIRED` · `COMMAND_REJECTED`

---

## Planner Snapshot

The planner sees exactly **one** compact snapshot (the latest). Old snapshots are replaced, never appended.

```mermaid
graph LR
    subgraph Snapshot["PlannerSnapshot"]
        ID["snapshot_id, arm_id, ts_ms"]
        SK["skill: name, phase, skill_run_id"]
        TG["target: hint_valid, confidence,\nobs_age_ms, delta_xyz_ee, distance_m"]
        PC["perception: tracking_status,\nfailure_code"]
        ACT["act: status, buffer_fill_ms,\nbuffer_low"]
        PR["progress: elapsed_ms,\nno_progress_ms, delta_distance"]
        OUT["outcome: state, reason_code"]
        SAF["safety: state, reflex_active"]
        CMD["command_acks"]
        EVT["recent_events (ring of 8)"]
    end

    style Snapshot fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
```

---

## ACT Action Space

Actions are **per-timestep servo increments** in the end-effector frame:

```
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]
```

- Deltas are applied relative to the **current measured** EE pose (closed-loop)
- Temporal ensembling blends overlapping chunk predictions per-timestep
- On phase transition, the buffer is trimmed to ~50–100 ms to avoid stale tail actions
- **v0 Isaac Lab profile:** 10 Hz control rate, 10-step chunks (1 s horizon) for debugging simplicity. Production target is 50–100 Hz with shorter horizons (200–500 ms).

---

## Timing Budgets

```mermaid
gantt
    title Service Timing Budgets
    dateFormat X
    axisFormat %L ms

    section Control
    ControlService tick (target 50-100 Hz) :0, 20
    ControlService tick (v0 sim 10 Hz)     :0, 100

    section Skill
    ACT inference (10-20 Hz)           :0, 100

    section Perception
    Fast loop (tracker + depth + gates) :0, 120
    VLM reacquire (async)              :0, 5000

    section Planner
    LLM decide (event-driven)          :0, 5000
    Watchdog fallback                  :0, 30000
```

| Path | Target | v0 Sim |
|---|---|---|
| ControlService tick | 50–100 Hz (10–20 ms) | 10 Hz (100 ms) |
| Fast perception loop → hint publish | ≤ 80–120 ms | same |
| VLM reacquire (async, off critical path) | 0.5–5 s | same |
| ACT chunk horizon | 200–500 ms | 1 s (10 steps) |
| ACT buffer fill target | 150–300 ms | ~1 s |
| Planner watchdog | 30 s max between ticks | same |
