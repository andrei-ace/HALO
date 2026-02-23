# HALO Project Plan (Planner + Target Perception + ACT Skill Runner)

Date: 2026-02-23
Scope: HALO — v1: Isaac Sim/Lab, single-arm pick/place, local models via Ollama, smooth continuous control with low-frequency reasoning. Real SO-ARM101 arm (two arms eventually) is a later phase.

---

## 1) High-level architecture (roles and boundaries)

### Goals
- **Smooth continuous motion**: no “one action → think → one action” pauses.
- **Low-frequency high-level reasoning** (agentic planner) with **fresh perception**.
- **Deterministic safety**: no safety-critical logic in the LLM loop.
- **Perception/controls are machine-to-machine**: numeric control hints are **not** transported through LLM context.

### Core modules
1) **Planner (agentic, stateful)**
   - Model: `gpt-oss:20B` (Ollama)
   - Responsibilities:
     - Task-level orchestration (choose next skill / handle failures / retries / target selection).
     - Tool use via commands + cached-state queries.
     - Keeps context clean by retaining only the **latest** runtime snapshot (deprecating older ones), plus a small recent event ring and active command acknowledgements.
   - Must *not*:
     - Stream per-timestep motor actions.
     - Pass raw numeric control hints through text prompts.

2) **TargetPerceptionService (unified perception subsystem)**
   - Internals (hidden from planner):
     - VLM grounding: `qwen3-vl:30B` (scene camera only by default)
     - Segmentation: SAM/SAM2 (mask init/refinement)
     - Tracking: fast tracker (mask/points), continuous updates
     - Depth fusion: ZED X depth + mask → 3D target estimate
     - Plausibility gates: reject impossible motion, depth invalid, etc.
     - Fast perception loop (budgeted): tracker + (SAM refresh when needed) + depth fusion + gates → fused target hints.
       - This loop is the one that should target **<80–120 ms** end-to-end latency (initial goal) and can run at ~10–30 Hz.
     - Semantic fallback (not budgeted): VLM grounding runs **asynchronously** on events (init / lost-target / verification).
       - Expected latency is orders of magnitude higher (e.g., ~0.5–5 s depending on hardware/settings).
       - Output should be a **seed** (bbox/ROI/point + label) for SAM/tracker, never part of the steady hint-publish loop.
   - Planner-facing responsibilities:
     - Maintain a tracked target and publish **fused target hints** (robot-frame + EE-relative).
     - Expose health + failure codes (e.g., OCCLUDED, DEPTH_INVALID).
     - Decide when to use tracker-only vs SAM refresh vs VLM reacquire.

3) **SkillRunner (deterministic state machine executor)**
   - Runs a **fixed FSM** for each skill (e.g., Pick skill).
   - Uses ACT for continuous motion:
     - Model: **ACT (Action Chunking Transformer)** as the low-level policy for approach/alignment/lift behaviors.
     - Executes actions via chunk buffer + controller; transitions between phases using fast success checks.
   - Key point:
     - Phase timing is handled inside SkillRunner (fast), not by the planner (slow).
     - The planner launches skills and reacts to success/failure; it does not time “grasp now”.

4) **ControlLoop (high-rate deterministic execution loop)**
   - Executes buffered actions at fixed rate (e.g., 50–100 Hz).
   - Consumes ACT chunk actions produced at a lower rate (e.g., 10–20 Hz) using interpolation or sample-and-hold.
   - Applies smoothing / interpolation / velocity/accel clamps.
   - Never waits on planner or VLM.

5) **Safety Guard + Reflex Layer (LLM-independent)**
   - Hard guards: joint/workspace bounds, velocity/accel/jerk limits, coarse collision checks.
   - Reflex: immediate overrides (stop/retract/open gripper) on faults/overload/unsafe conditions.
   - Planner handles recovery *after* reflex stabilizes the robot.

---

## 2) Hardware layout

### Cameras
- **Scene camera (primary perception + depth): ZED X**
  - Used for: VLM frames, global target discovery/reacquisition, depth-based 3D.
- **Wrist camera (lightweight): 1080p USB2 UVC module**
  - Used for: ACT/VLA local visual servoing, final approach and grasp stability.
  - Mounted via SO-ARM100/101 wrist-cam mount.

### Principle
- VLM uses **scene camera only** by default to keep semantics consistent and reduce noise.
- Wrist camera is heavily leveraged by ACT for final approach alignment and robustness.

Fallback (optional):
- If scene reacquire fails repeatedly, allow **rare** wrist-VLM calls or use deterministic reseed cues.
- Wrist-camera VLM calls are disabled during steady-state execution and only enabled in explicit recovery/debug modes.

---

## 3) Data/control paths (keep numeric hints out of LLM)

### Control path (machine-to-machine)
- TargetPerceptionService publishes **target_state / target_hint_vec** into shared runtime state.
- SkillRunner reads target hints directly (not through the planner).
- ACT consumes:
  - Wrist RGB (primary), optional scene RGB at lower rate
  - Robot state (joints, gripper)
  - Low-dim target hints (robot frame + EE-relative)
  - `phase_id` (approach/align/grasp/lift)

### Decision path (LLM)
- Planner reads **compact runtime snapshot** and emits high-level commands:
  - Start skill, abort, override target, request refresh (rare)
- Planner does not transport control floats.

---

## 4) Time sync + coordinate frames (explicit and enforced)

### Required transforms (with quality metrics)
Maintain calibrated transforms:
- `T_base<-scene_cam` (ZED X)
- `T_ee<-wrist_cam` (wrist UVC)
- `T_base<-ee` from robot kinematics

### Timestamp discipline
Every published hint must include:
- `hint_timestamp`
- `robot_state_timestamp` (used for EE pose)
- `time_skew_ms = hint_ts - robot_state_ts`
- `obs_age_ms = now - hint_ts`

Safety rule:
- If `obs_age_ms` or `time_skew_ms` exceeds thresholds → `hint_valid = 0` (unsafe), force reacquire/hold.

### Latency budgets (initial targets)
Define and monitor explicit budgets early so tuning is measurable:
- **Fast perception loop** (scene camera capture → tracker/depth fusion → fused target hint publish): initial target (e.g., **<80–120 ms**).
- **Semantic VLM job** (grounding / reacquire / verify): **not** part of the fast-loop budget; runs async and may take ~0.5–5 s.
- Wrist camera capture → ACT observation availability: target budget per control profile
- Snapshot publication cadence (planner-grade): fixed rate (e.g., 2–10 Hz) + event-triggered updates
- Planner reaction latency: best-effort only (not in the control path)
- Phase-specific freshness thresholds (e.g., stricter in PREGRASP_ALIGN / GRASP_CLOSE than in TRANSIT)


### EE-relative hints (recommended)
Publish both:
- `target_xyz_base`
- `delta_xyz_ee` (preferred for ACT learning and robustness)
- `distance_m`, `approach_vec_ee`
- confidence + validity flags

---

## 5) Smooth execution with ACT: chunk buffering + closed-loop behavior

### Why chunking
ACT outputs short-horizon **action chunks**. The controller streams them smoothly while ACT refills asynchronously.

Note: in this project, “ACT” means an **ACT-style chunked imitation policy** optionally conditioned on `phase_id` and low-dimensional target hints (not only raw vision+proprio). Keep a simpler baseline for comparison.


### Action space semantics (avoid drift)
- This plan uses **EE-frame delta pose + gripper** actions (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd).
- Interpret deltas as **per-timestep servo increments** applied relative to the *current measured* EE pose (closed-loop via the controller), not as a long open-loop trajectory to integrate blindly for the entire chunk.
- Temporal ensembling (below) should blend **per-timestep actions** from overlapping chunks. Do not ensemble by integrating whole-chunk deltas first.
- Alternative (often robust on real hardware): predict **absolute EE poses relative to chunk start** (still local, but avoids compounding integration).

### Closed-loop, receding-horizon rule
- Do not execute long open-loop chunks when the target can move.
- Use shorter horizons for moving targets:
  - chunk horizon ~ 200–500 ms (tunable)
  - buffer fill target ~ 150–300 ms
  - refill before buffer low (MPC-like behavior)
- On phase switch, **trim buffer** (keep only ~50–100ms) to avoid executing old-phase actions.

### Temporal ensembling (recommended)
Blend overlapping actions from consecutive chunks over an overlap window to reduce boundary artifacts and tail drift.

---

## 6) Planner tools design: cached snapshot + async commands

### Snapshot tool (planner-facing)
- **`get_latest_runtime_snapshot()`** returns a compact, planner-grade snapshot:
  - arm_id, current skill/phase
  - target validity/freshness/confidence
  - tracking status + failure codes
  - progress deltas (`delta_distance`, `no_progress_ms`)
  - ACT buffer health (buffer low/fill ms)
  - outcome state (in_progress/success/fail/uncertain)
  - safety state + reflex flags
  - small recent event ring (3–8 items)

**Old snapshots are deprecated** in LLM context when a new one arrives.

Hard rule for prompt assembly:
- The planner must see **exactly one** `get_latest_runtime_snapshot()` result: the latest one.
- Implement this as a middleware behavior that **replaces** the previous snapshot tool message (do not append a history of snapshots), otherwise the LLM will invent temporal logic.


### Debug snapshot tool (rare)
- **`get_runtime_debug_snapshot(section=...)`** for deep dives (qpos arrays, transforms, detailed tracker stats).
- Not used in steady-state planning.

### Command tools (async, non-blocking)
Planner sends high-level commands; results appear in subsequent snapshots/events:
- `start_skill(skill, target_handle, options/profile)`
- `abort_skill(skill_run_id, reason)`
- `override_target(skill_run_id, target_handle)`
- `request_perception_refresh(mode, reason)` (rare)

#### Command sequencing + idempotency (must-have)
Each planner command should carry:
- `command_id` (UUID)
- `arm_id`
- `issued_at`
- `expected_skill_run_id` (when applicable)
- optional `precondition_snapshot_id` (helps reject stale commands)

Execution contract:
- duplicate `command_id` → idempotent response (`already_applied`)
- stale / mismatched precondition → reject with explicit reason (`stale`, `wrong_skill_run`, etc.)
- command acceptance/rejection is surfaced via events and the next snapshot (`accepted`, `rejected`, `stale`, `already_applied`)


---

## 7) TargetPerceptionService contract

### Planner → Perception
- `set_tracking_target(arm_id, target_spec | target_handle, task_role, optional_roi, tolerance)`
- (optional) `request_refresh(reason)` when planner wants a forced reacquire/verify

### Perception → Runtime state (for SkillRunner + planner summary)
Publishes:
- `arm_id`
- `tracking_status`: tracking / lost / relocalizing / reacquiring
- `target_handle`, `target_id` (if known)
- `target_hint_vec`: target pose + EE-relative deltas + confidence
- `hint_valid`, `obs_age_ms`, `time_skew_ms`, `calib_valid`
- `failure_code` (explicit): OCCLUDED, OUT_OF_VIEW, DEPTH_INVALID, MULTIPLE_CANDIDATES, CALIB_INVALID, TRACK_JUMP_REJECTED, etc.

### Deterministic plausibility gates (must-have)
Invalidate hints and trigger reacquire when:
- target motion exceeds physical bounds (velocity/accel)
- target drifts outside workspace AABB
- depth-valid ratio below threshold
- time skew / obs age too high

---

## 8) SkillRunner FSM (Pick) — phases decided locally

### Principle
Planner starts the skill; SkillRunner runs the micro-state machine with fast success checks.
This prevents pauses between approach→grasp for moving targets.

### Pick FSM states (summary)
- IDLE
- ACQUIRE_TARGET
- REACQUIRE_TARGET
- APPROACH
- PREGRASP_ALIGN
- GRASP_CLOSE
- VERIFY_GRASP (optional)
- WAIT_VERIFY (optional, VLM verify)
- LIFT
- GRASP_RECOVER (retry)
- SUCCESS_PICK
- FAIL(reason)

Each state has:
- entry actions (set `phase_id`, trigger gripper close, request reacquire, etc.)
- success predicates (distance thresholds, gripper occupancy band, lift delta-z, etc.)
- failure predicates (timeout, no progress, target lost, safety reflex)

### Moving target critical detail
- GRASP_CLOSE should be triggered deterministically when in grasp window (distance + persistence), without waiting for planner.
- Optional immediate gripper close and buffer trimming on phase transition.

### Gripper timing / dwell modeling (important for real transfer)
Model and expose configurable timing terms in the SkillRunner profile:
- `gripper_close_timeout_ms`
- `gripper_settle_dwell_ms`
- optional force/current threshold for “object contacted / grasp likely”
- phase-specific timing compensation (especially close→lift transitions)


---

## 9) Outcome monitoring (fast, deterministic, phase-aware)

A dedicated **SkillOutcomeMonitor** computes:
- `in_progress` / `success` / `failure` / `uncertain`
- `reason_code` and `needs_verify` when ambiguous

Signals:
- TargetPerceptionService (object follows EE, object height change, zone containment)
- Wrist cues for final alignment and presence
- Gripper width/current/effort (if available)
- Progress watchdogs (`delta_distance`, `no_progress_ms`)
- Timeouts and retry budgets

### Canonical failure / decision taxonomy (recommended)
Define stable enums (or canonical strings) shared across services and logs:
- `perception_failure_code`: `OCCLUDED`, `OUT_OF_VIEW`, `DEPTH_INVALID`, `MULTIPLE_CANDIDATES`, `CALIB_INVALID`, `TRACK_JUMP_REJECTED`, ...
- `skill_failure_code`: `TIMEOUT`, `NO_PROGRESS`, `NO_GRASP`, `DROP_DETECTED`, `PLACE_MISS`, `UNSAFE_ABORT`, ...
- `safety_reflex_reason`: `JOINT_LIMIT`, `WORKSPACE_LIMIT`, `COLLISION_RISK`, `OVERCURRENT`, `ESTOP`, ...
- `planner_recovery_decision`: `RETRY_SAME_TARGET`, `REACQUIRE`, `RETARGET`, `ABORT_TASK`


---

## 10) Logging + dataset creation (ACT)

### Logging (for debugging + dataset)
Log for every chunk and decision:
- `chunk_id -> hint_version -> snapshot_id`
- chunk start/end timestamps
- state transitions with reason codes
- outcome monitor signals at transition points
- perception failure codes and reacquire attempts

### Dataset (imitation learning)
Record:
- wrist RGB (primary), optional scene RGB
- robot state
- action stream (what was executed)
  - If using deltas: store deltas computed between consecutive measured EE poses (servo increments), plus controller clamps (if any).
- phase_id / skill state label
- target hints (EE-relative strongly recommended)
- success/failure labels per episode (optional but useful)

Because target hints are produced deterministically, dataset generation is mostly automatic.

### Dataset QA / filtering (must-have)
Reject or flag episodes / segments with:
- prolonged control saturation / clipping
- timestamp gaps or dropped frames
- replay mismatch beyond threshold (recorded actions do not reproduce behavior)
- mislabeled successes (e.g., failed grasp marked success)
- corrupted images / missing sensor packets

Track dataset stats continuously:
- per-phase duration distributions
- success/failure ratios and failure reasons
- action magnitude histograms / clipping frequency
- image corruption / missing frame counts

### Train / val / test split protocol
- Split by **episode seed / object placement**, not by random timesteps.
- Keep a held-out evaluation set with unseen cube/bin placements.
- Add a “hard” bucket (edge placements, camera jitter, lighting changes).
- Report metrics separately for in-distribution, randomized, and hard scenarios.


---

## 11) Recommended implementation structure (processes/threads)

Suggested separation (processes if possible):
- `planner_service` (LLM agent + tools)
- `target_perception_service` (VLM+SAM+tracker+depth fusion)
- `skill_runner_service` (FSM + ACT inference + chunk planner)
- `control_service` (realtime executor + safety/reflex)
- shared state store + event bus (queues, ROS2 topics, or similar)

Planner reads snapshots and emits commands; no blocking in the motion loop.

Implementation note: namespace runtime state, commands, snapshots, and events by `arm_id` from day one (even if v0 runs one arm).

---

## 12) Minimal planner snapshot (example shape)

Planner-grade fields only:
- identity: snapshot_id, ts, last_event_id, arm_id
- goal: task_id, skill, phase
- target: handle, hint_valid, confidence, obs_age_ms, time_skew_ms, calib_valid, delta_xyz_ee, distance, delta_distance
- perception: tracking_status, failure_code, reacquire_fail_count, vlm_job_pending
- act: status, buffer_low, buffer_fill_ms
- progress: elapsed_ms, no_progress_ms
- outcome: state, reason_code, needs_verify
- safety: state, reflex_active, reason_codes
- command_ack: recent command statuses (accepted/rejected/stale/already_applied)
- recent_events: small ring

---

## 13) Naming

- **Planner** (agentic orchestrator)
- **TargetPerceptionService** (VLM+SAM+tracker+depth → fused target state)
- **SkillRunner** (FSM executor + ACT)
- **ControlLoop** (high-rate deterministic execution)
- **SafetyGuard / ReflexLayer**
- **SkillOutcomeMonitor**

---

## 14) Next build steps (suggested order)

1. Implement shared runtime state + event IDs + compact snapshot.
2. Implement SkillRunner Pick FSM with deterministic phase transitions and fast success checks.
3. Integrate ACT with chunk buffering + buffer trimming on phase switches.
4. Integrate TargetPerceptionService in basic mode:
   - scene VLM for target init
   - tracker + depth → target hints
   - plausibility gates
5. Add verification path (WAIT_VERIFY) and fallback reacquire strategies.
6. Add logging and episode capture for dataset.
7. Iterate thresholds/profiles for moving targets (short horizons, strict freshness).

## Isaac Sim/Lab-first bootstrapping for HALO: Teacher demos → ACT (teleop-aligned)

### Target task (v0)
**Pick cube from table → place into bin** (single cube, single bin, no clutter at first).

Goal: build the **same data + training + eval loop** we’ll use later on real SO-ARM101 hardware with teleoperation, but in sim the “teleop operator” is replaced by an automated teacher.

---

## Teacher (demo generator) to avoid teleop

### Recommended teacher (v0): Analytic / planner-backed controller
Use an automated teacher based on **motion generation + IK / operational-space control**, producing smooth end-effector trajectories:
- Free-space motion: target poses (pregrasp, grasp, preplace, place)
- Contact logic: gripper close/open + lift/verify heuristics
- Collision handling: replans + retreat on failure

Why: fast to generate thousands of clean episodes; naturally yields **phase/skill labels** that match the later skill-conditioned ACT setup.

### Optional later teacher: RL teacher → distill to ACT
Train an RL policy using privileged state for robustness; record rollouts as demonstrations; distill to ACT (pixels + proprio).

---

## ACT-aligned interfaces (keep identical between sim and real)

### Action space (ACT output)
**End-effector delta pose + gripper command**, per timestep:
- `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_cmd]`
- Action deltas are represented in the **end-effector frame** (preferred for local servoing / transfer robustness).
- `gripper_cmd`: binary open/close or scalar (0..1)
- Apply per-dimension clipping + normalization consistently in both dataset generation and training.
- Execution mapping (must be explicit and semantically identical across sim/real):
  - sim: EE-delta (EE frame) → IK / operational-space controller / resolved-rate controller
  - real: EE-delta (same frame + scaling) → your low-level controller

### Control rate + chunking
- Control rate: **10 Hz** to start (easy debugging)
- ACT chunk length: **10 steps** (predict 1 second of actions)
- Training sample at time `t`: predict actions `t : t+chunk_len-1`
- **Note:** this v0 Isaac setup intentionally uses a longer horizon (1.0 s) for simplicity/debugging. For moving targets and real-hardware deployment, expect shorter receding horizons (e.g., 200–500 ms) with tighter refill thresholds.

### Observations (what the student sees)
Record exactly what we’ll have on real SO101:
- `wrist_rgb[t]` (required)
- optional `scene_rgb[t]` (nice-to-have)
- `q[t]` joint positions
- optional `qd[t]` joint velocities
- `gripper_state[t]` (position/force if available)
- optional `prev_action[t-1]` (stability)

Teacher may use privileged state for transitions (cube/bin pose), but **student inputs stay non-privileged**.

### Optional teacher quality / provenance labels (useful later)
Record whether each segment/action came from:
- nominal teacher path
- recovery path / retry
- replan after collision risk
- fallback heuristic

These labels can be used for filtering or weighting during training/fine-tuning.

---

## Demonstration dataset format (teleop-like)

### Per episode metadata
- `episode_id`, `seed`
- `success` bool
- `failure_reason` (timeout, collision, no_grasp, drop, etc.)
- randomization params (lighting/texture/physics jitter) for reproducibility

### Per timestep arrays
Observations:
- `wrist_rgb[t]` (uint8 H×W×3)
- optional `scene_rgb[t]`
- `q[t]`, optional `qd[t]`
- `gripper_state[t]`

Actions (targets):
- `a[t] = [Δx,Δy,Δz, Δroll,Δpitch,Δyaw, gripper_cmd]`

Labels (very important early):
- `phase_id[t]` (FSM state label)
- `done[t]`

---

## FSM: states + skills (teacher + later skill-conditioned ACT)

### Skills (reusable primitives)
Each skill outputs actions in the ACT action space at 10 Hz:
1. **MoveEE(target_pose, tol, speed_profile)**: go to a pose smoothly (free-space)
2. **AlignYaw(target_yaw)**: align yaw before descent (optional initially)
3. **DescendTo(target_pose)**: controlled descent to grasp/place pose
4. **CloseGripper()** / **OpenGripper()**
5. **Lift(Δz)**: lift straight up after grasp
6. **VerifyGrasp()**: success if cube lifts above threshold and stays stable N steps
7. **RetreatUp(Δz)**: retreat to safe height

### Phase IDs (stable labels for training)
Use a small, fixed enum:
- `0 RESET`
- `1 APPROACH_PREGRASP`
- `2 ALIGN`
- `3 DESCEND_GRASP`
- `4 CLOSE`
- `5 LIFT`
- `6 VERIFY_GRASP`
- `7 TRANSIT_PREPLACE`
- `8 DESCEND_PLACE`
- `9 OPEN`
- `10 RETREAT`
- `11 DONE`
Recovery:
- `20 RECOVER_RETRY_APPROACH`
- `21 RECOVER_RETRY_DESCEND`
- `22 RECOVER_REGRASP`

### FSM transitions (v0)
**RESET**
- open gripper, go home, randomize cube pose → `APPROACH_PREGRASP`

**APPROACH_PREGRASP**
- target: cube_pose + z_pregrasp → reached? `ALIGN` else timeout/collision → `RECOVER_RETRY_APPROACH`

**ALIGN**
- optional yaw align → `DESCEND_GRASP`

**DESCEND_GRASP**
- target: cube_pose + z_grasp → reached? `CLOSE` else → `RECOVER_RETRY_DESCEND`

**CLOSE**
- close + short dwell (`gripper_settle_dwell_ms`) → `LIFT`

**LIFT**
- lift Δz_lift → `VERIFY_GRASP`

**VERIFY_GRASP**
- if cube height > threshold + stable N steps → `TRANSIT_PREPLACE`
- else → `RECOVER_REGRASP`

**TRANSIT_PREPLACE**
- target: bin_pose + z_preplace → `DESCEND_PLACE`

**DESCEND_PLACE**
- target: bin_pose + z_place → `OPEN`

**OPEN**
- open + short dwell (configurable) → `RETREAT`

**RETREAT**
- retreat up → `DONE`

**DONE**
- mark success → end episode

Recovery:
- `RECOVER_RETRY_APPROACH`: retreat up, replan approach → `APPROACH_PREGRASP`
- `RECOVER_RETRY_DESCEND`: retreat up, re-approach/descend → `APPROACH_PREGRASP` or `DESCEND_GRASP`
- `RECOVER_REGRASP`: open, retreat, re-approach → `APPROACH_PREGRASP`

---

## Training plan (step-by-step, teleop-aligned)

### Step 1 — Generate clean teacher demos (no randomization / mild randomization)
- Run FSM teacher for **10k–50k episodes**
- Record: wrist RGB + proprio + actions + phase_id + success/failure
- Validate “teleop parity” in sim:
  - replay recorded actions → episode should succeed (otherwise control/timing mismatch)
- Create train/val/test splits by episode seed / placement (not by timestep) before training starts.

### Step 2 — Train skill-conditioned ACT (most stable starting point)
Train an ACT-style chunked imitation policy to predict action chunks conditioned on `phase_id`:
- Input: (wrist_rgb, proprio, phase token)
- Output: `chunk_len × action_dim`
- Loss: Huber/MSE on deltas + BCE for gripper if binary
- Expectation: near-perfect imitation on free-space phases first; then grasp/place timing.

Baselines (recommended, cheap sanity checks):
- one-step BC policy (CNN/ViT encoder + MLP head)
- short-horizon chunked BC without ACT temporal machinery
- no-phase baseline (same observations, no `phase_id`) to quantify value of phase conditioning

### Step 3 — Closed-loop evaluation with FSM orchestrator
- Keep FSM for high-level sequencing (phase selection + transitions)
- Replace teacher actions with ACT-predicted actions inside each phase
- Ensure reproducible rollouts: lock simulator version/physics settings, save episode seeds, and support deterministic replay mode when possible.
- Report over N seeds with confidence intervals (not only single-run success).
- Track metrics:
  - success rate, time-to-success
  - grasp success rate, drop rate
  - placement success (in-bin)
  - collisions / joint limit hits
  - “stuck” timeouts (no progress)

### Step 4 — Scale difficulty (domain randomization + hard initializations)
Gradually ramp:
- lighting/textures, camera jitter
- cube pose distribution (near edges, rotated)
- friction/mass perturbations

Fine-tune ACT on the expanded dataset (same schema, same code).

### Step 5 — Dataset improvement via teacher corrections (DAgger-like)
- Roll out current ACT policy
- When it deviates/fails, query teacher for corrective actions
- Append corrected segments to dataset
- Fine-tune ACT (small LR)

### Step 6 — Optional: remove phase conditioning (end-to-end ACT)
After skill-conditioned is robust:
- train an end-to-end ACT without `phase_id`
- compare success/stability vs skill-conditioned
- keep whichever is more reliable (skill-conditioned is usually easier to debug + recover).

---

## Direct mapping to real SO-ARM101 hardware teleoperation later
When moving to real:
- replace teacher actions with teleop operator actions
- keep identical dataset schema, action space, chunking, training/eval scripts
- only swap sensor sources (real wrist cam + joint encoders) and action execution (real controller)

This preserves the “teleop → ACT fine-tune → eval → iterate” loop end-to-end.

## Sim-to-real alignment checklist (must match early)
- Wrist camera FOV/resolution and mounting geometry (`T_ee<-wrist_cam`)
- Camera intrinsics/extrinsics conventions and timestamping
- Control rate and action semantics (EE-frame delta interpretation, scaling, clipping)
- Gripper command semantics + latency + dwell behavior
- Observation synchronization model (camera / robot state skew)
- Noise/blur/compression characteristics and exposure behavior
- Joint limits, speed limits, and controller behavior (including saturation)
- Safety constraints and stop/retract semantics

## 15) Acceptance criteria (v0 / v1)
### v0 sim (single cube, no clutter)
- Stable teacher demo generation and replay parity passes on sampled episodes.
- Skill-conditioned ACT closed-loop success meets target threshold over N fixed seeds.
- No uncontrolled collisions / repeated safety faults in nominal scenarios.

### v0.5 sim (randomized / harder)
- Maintain acceptable success under lighting/texture/camera jitter and edge placements.
- Failure taxonomy and logs are complete enough to diagnose top failure modes quickly.

### v1 real (static cube first)
- Dry-run motion checks pass (smoothness, bounds, latency, timestamp freshness).
- Pick-and-place success reaches target threshold over N trials with logged outcomes.
- Planner intervention rate remains low in nominal runs; safety/reflex events are rare and explainable.