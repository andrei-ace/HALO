# mujoco_sim/ — SO-101 MuJoCo Simulation for HALO

## Overview

Raw MuJoCo simulation module for SO-101 PICK demo generation, episode recording,
and ZMQ-based bridge to the HALO runtime. Uses trajectory-planned pick with
jerk-limited ruckig profiles and damped least-squares IK. No robosuite dependency.

## Structure

```
mujoco_sim/
  __init__.py
  constants.py          # synced from halo.contracts (verified by tests)
  scene_info.py         # single source of truth: entity names, TCP offset, all grasp/teacher defaults
  config/
    env_config.py       # EnvConfig dataclass (SO-101 defaults)
  env/
    so101_env.py        # SO101Env wrapper (raw MuJoCo, dual cameras, joint-position control)
    robosuite_env.py    # (legacy, not exported)
  dataset/
    raw_episode.py      # Timestep (with phase_id, joint_pos) + RawEpisode in-memory buffer
    writer_hdf5.py      # write_episode() — one HDF5 per episode, gzip images
    reader_hdf5.py      # read_episode() — load HDF5 → RawEpisode
  teacher/
    pick_teacher.py     # PickTeacher trajectory-planned policy (pre-computes full trajectory on first step)
    ik_helper.py        # Damped least-squares IK: EE position → 5 arm joint angles
    keyframe_planner.py # SE(3) keyframe planner: cube pose → 5 Cartesian keyframes with phase tags
    waypoint_generator.py # Cartesian keyframes → joint-space waypoints via orientation-aware IK
    trajectory.py       # Jerk-limited trajectory planning via ruckig (joint waypoints → smooth profiles)
  runner/
    run_teacher.py      # run_teacher() — stabilize → teacher loop → write HDF5
  server/
    __init__.py         # SimServer: standalone ZMQ server owning SO101Env + PickTeacher
    __main__.py         # CLI: python -m mujoco_sim.server
    config.py           # SimServerConfig (host, ports, render FPS, JPEG quality)
    handlers.py         # dispatch_command() — routes CommandRPC messages
    protocol.py         # Message types, msgpack + JPEG serialization helpers
  scripts/
    test_env.py             # acceptance: dump scene.png + wrist.png
    generate_episodes.py    # CLI: --num-episodes, --output-dir, --stabilize, --standalone
    inspect_episode.py      # CLI: inspect HDF5 episode
    measure_pinch_offset.py # measure TCP pinch-point offset (gripperframe → jaw contact surface)
    visualize_ik_pose.py    # render IK-solved poses as static FK snapshots
  assets/
    so101/                  # MJCF + STL meshes (from SO-ARM100 repo)
      so101_new_calib.xml   # SO-101 robot model (position-controlled STS3215 servos)
      pick_scene.xml        # Pick scene: robot + floor + cube + scene_cam
      assets/               # 13 STL mesh files
  tests/
    test_constants_sync.py      # verify constants match halo.contracts (6 tests)
    test_raw_episode.py         # Timestep, RawEpisode, HDF5 roundtrip, phase_id, joint_pos (29 tests)
    test_pick_teacher.py        # PickTeacher phases, actions, full episode (20 tests)
    test_grasp_planner.py       # grasp enumeration, filtering, scoring (18 tests)
    test_trajectory_pipeline.py # keyframes → IK → ruckig integration (24 tests)
    test_server.py              # protocol serialization + command dispatch (15 tests)
```

## Commands

mujoco_sim is a uv workspace member — all commands run from the **repo root** using the shared venv.

```bash
# From repo root
uv run python -m pytest mujoco_sim/mujoco_sim/tests/ -v   # all mujoco_sim tests (112 tests)
uv run python -m pytest tests/test_mujoco_sim_contract_sync.py -v  # contract sync
uv run python -m pytest -v  # includes mujoco_sim tests (in root testpaths)

# Requires mujoco (uv sync --extra sim)
uv run python -m mujoco_sim.scripts.test_env  # acceptance: dump scene.png + wrist.png
uv run python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes  # generate demos
uv run python -m mujoco_sim.scripts.generate_episodes --standalone --num-episodes 10  # connect to running sim server
uv run python -m mujoco_sim.scripts.inspect_episode  # inspect latest episode
uv run python -m mujoco_sim.scripts.inspect_episode data/episodes/20260301_004533/ep_000000.hdf5  # specific file
uv run python -m mujoco_sim.scripts.visualize_ik_pose  # render IK waypoint snapshots
uv run python -m mujoco_sim.scripts.visualize_ik_pose --seed 7 -v  # with seed + verbose IK logging
uv run python -m mujoco_sim.scripts.measure_pinch_offset  # measure jaw contact offset
uv run python -m mujoco_sim.server  # start ZMQ sim server
```

### inspect_episode — episode debug inspector

`inspect_episode.py` prints a diagnostic summary of an HDF5 episode. Use it to verify teacher behaviour after generation. With no arguments it finds the latest episode in `data/episodes/`.

What it reports:
- **Step count, seed, duration, control freq**
- **Phase sequence** with per-phase step ranges and durations
- **Lift check** — verifies cube actually rose >=5 mm (max Z during LIFT phase, not final Z)
- **EE and cube positions** (first/last) + EE-cube distance over time (start, min, final)
- **Distance at each phase transition** — useful for spotting approach/grasp timing issues
- **Gripper** range (first -> last)
- **Action norms** (mean, max, last-50 mean, action dim)
- **Tracker** status (if tracking data present)
- **Video** presence and size

## SO-101 Joint/Actuator Names

| Idx | Joint | Range (rad) |
|-----|-------|-------------|
| 0 | `shoulder_pan` | +/-1.92 |
| 1 | `shoulder_lift` | +/-1.75 |
| 2 | `elbow_flex` | +/-1.69 |
| 3 | `wrist_flex` | +/-1.66 |
| 4 | `wrist_roll` | -2.74 to 2.84 |
| 5 | `gripper` | -0.17 to 1.75 |

EE site: `gripperframe` on the `gripper` body.

## Action Space

6D joint-position targets: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

Written directly to `data.ctrl[:]`. Position actuators with `kp=998.22` track targets.

Gripper semantics: `-0.17` = fully open, `1.75` = fully closed (joint angle, rad).

## Trajectory Planning Pipeline

PickTeacher pre-computes a full trajectory on the first `step()` call, then samples it in real time:

```
keyframe_planner.plan_pick_keyframes()  →  5 SE(3) keyframes (pos + ori + gripper + phase_id)
    ↓
waypoint_generator.generate_joint_waypoints()  →  5 joint-space waypoints via IK (with yaw-retry fallbacks)
    ↓
trajectory.plan_trajectory()  →  TrajectoryPlan (jerk-limited ruckig segments)
    ↓
pick_teacher.step()  →  samples plan at elapsed time → (action, phase_id, done)
```

### Keyframe planner

Generates 5 Cartesian keyframes from cube pose + home joints: `home → pregrasp → grasp → grasp_closed → lift`. Each keyframe is a `Keyframe` dataclass with position (3,), orientation (3x3 rot matrix), gripper, phase_id, label.

Orientation rule: gripperframe Z-axis aligned with world -Z (gripper pointing down). Yaw extracted from cube quaternion.

TCP pinch-point offset: `[0.0, 0.0, 0.0]` (zeroed — 3-4 mm actual offset causes more IK error than it corrects). Face standoff (3 mm) compensates instead.

### Waypoint generator

Solves IK for each keyframe independently, seeded from the previous waypoint for continuity. Falls back to yaw-rotated retries `[0, 90, -90, 180]` degrees on IK failure. Raises `IKFailure` if position error exceeds tolerance.

### Trajectory (ruckig)

Each waypoint pair becomes one ruckig segment with per-joint velocity/acceleration/jerk limits. All segments start/end at rest (v=0, a=0). Gripper interpolated linearly within each segment.

Default limits: vel `[1.0, 1.0, 1.0, 1.0, 1.5]`, accel `[3.0, 3.0, 3.0, 3.0, 4.0]`, jerk `[10.0, 10.0, 10.0, 10.0, 15.0]`.

### TeacherConfig

All defaults are named constants in `scene_info.py` (single source of truth):

```python
TeacherConfig(
    pregrasp_height_offset=DEFAULT_PREGRASP_STANDOFF,  # 0.08 m
    lift_height=DEFAULT_LIFT_HEIGHT,                    # 0.08 m
    ori_tol_deg=DEFAULT_ORI_TOL_DEG,                    # 55°
    grasp_n_candidates=DEFAULT_GRASP_N_CANDIDATES,      # 64
    grasp_max_cone_deg=DEFAULT_GRASP_MAX_CONE_DEG,      # 5°
    grasp_face_contact_span=DEFAULT_CUBE_FACE_CONTACT_SPAN,  # 0.10
    grasp_face_standoff=DEFAULT_FACE_STANDOFF,          # 0.003 m (3 mm)
    max_velocity=None,            # defaults in JointLimits
    max_acceleration=None,
    max_jerk=None,
    ik_pos_weight=DEFAULT_IK_POS_WEIGHT,    # 1.0
    ik_ori_weight=DEFAULT_IK_ORI_WEIGHT,    # 0.1
    ik_max_iters=DEFAULT_IK_MAX_ITERS,      # 200
    ik_tol=DEFAULT_IK_TOL,                  # 1e-3
    ik_pos_tol=DEFAULT_IK_POS_TOL,          # 0.03 m
)
```

## ZMQ Sim Server

Standalone process owning SO101Env + PickTeacher, bridged to HALO runtime via ZMQ.

### Channels

| Channel | ZMQ Pattern | Default Port | Purpose |
|---------|-------------|--------------|---------|
| TelemetryStream | PUB | 5560 | Frames + state at render_fps (10 Hz) |
| CommandRPC | REP | 5561 | step, reset, teacher_step, configure, set_hint, shutdown |

Single-threaded main loop (required for macOS OpenGL rendering). Protocol uses msgpack + raw bytes for numpy arrays, JPEG for camera frames.

### SimServerConfig

```python
SimServerConfig(
    host="127.0.0.1",
    telemetry_port=5560,
    command_port=5561,
    render_fps=10,
    jpeg_quality=85,
    env_config=EnvConfig(),
    teacher_config=TeacherConfig(),
)
```

## Usage

### Python API — environment only

```python
from mujoco_sim.env import SO101Env
from mujoco_sim.config import EnvConfig
import numpy as np

env = SO101Env(EnvConfig())
obs = env.reset(seed=42)
# obs keys: rgb_scene (480,640,3), rgb_wrist (240,320,3),
#           qpos (13,), qvel (12,), gripper (float),
#           ee_pose (7,), object_pose (7,), joint_pos (6,)

action = np.zeros(6)  # [shoulder_pan, ..., gripper]
obs, reward, done, info = env.step(action)

env.close()
```

### Python API — teacher + recording

```python
from mujoco_sim.env import SO101Env
from mujoco_sim.config import EnvConfig
from mujoco_sim.teacher import PickTeacher, TeacherConfig
from mujoco_sim.dataset import RawEpisode, Timestep, EpisodeMetadata, write_episode
import numpy as np

env = SO101Env(EnvConfig())
teacher = PickTeacher(TeacherConfig())
obs = env.reset(seed=42)

# Stabilize
for _ in range(100):
    obs, _, _, _ = env.step(env.home_qpos)

teacher.reset()
episode = RawEpisode(metadata=EpisodeMetadata(seed=42))

for step in range(500):
    action, phase_id, done = teacher.step(obs, env.mujoco_model, env.mujoco_data)
    episode.append(Timestep(
        rgb_scene=obs["rgb_scene"], rgb_wrist=obs["rgb_wrist"],
        qpos=obs["qpos"], qvel=obs["qvel"],
        gripper=float(obs["gripper"]), ee_pose=obs["ee_pose"],
        action=np.array(action, copy=True), phase_id=phase_id,
        object_pose=obs.get("object_pose"), joint_pos=obs.get("joint_pos"),
    ))
    if done:
        break
    obs, _, env_done, _ = env.step(action)
    if env_done:
        break

write_episode(episode, "episodes/ep_000000.hdf5")
env.close()
```

### Teacher phase sequence (executed in trajectory)

```
IDLE(0) → MOVE_PREGRASP(3) → EXECUTE_APPROACH(5) → CLOSE_GRIPPER(6) → LIFT(8) → DONE(9)
```

SELECT_GRASP, PLAN_APPROACH, and VISUAL_ALIGN are folded into the planning step (instantaneous). VERIFY_GRASP is implicit in the gripper-close segment.

### HDF5 file layout

```
ep_000000.hdf5
+-- obs/
|   +-- rgb_scene    (T, H, W, 3) uint8  [gzip]
|   +-- rgb_wrist    (T, H, W, 3) uint8  [gzip]
|   +-- qpos         (T, 13)
|   +-- qvel         (T, 12)
|   +-- gripper      (T,)
|   +-- ee_pose      (T, 7)
|   +-- joint_pos    (T, 6)             [if present]
|   +-- phase_id     (T,) int32         [if present]
|   +-- object_pose  (T, 7)             [if present]
|   +-- contacts/    step_NNNNNN (N,)   [if present]
+-- action           (T, 6)
+-- attrs: seed, env_name, robot, control_freq, num_steps, created_at
```

## Dependencies

```
numpy, h5py>=3.0, tqdm>=4.60, ruckig>=0.14
Optional: mujoco>=3.1.6 (sim), pyzmq>=25 + msgpack>=1.0 (server), opencv-python>=4.8 (viewer/server)
```

## Contact Solver (pick_scene.xml)

MuJoCo's soft contact model allows friction slip by default. The pick scene uses three solver-level settings to suppress slippage during grasping:

```xml
<option impratio="10" cone="elliptic" noslip_iterations="3"/>
```

- **`impratio=10`** — friction constraints 10× stiffer than normal force (prevents gradual slip)
- **`cone="elliptic"`** — accurate friction cones (required for high impratio to work well)
- **`noslip_iterations=3`** — post-processing solver that entirely prevents residual slip

Additionally, gripper collision geoms use `friction="2.0 0.1 0.001" condim="4"` (matching the cube), and the gripper actuator force is `±6.0 N` (increased from default `±3.35 N`).

Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#cslippage

## Constants (scene_info.py)

All grasp planner and teacher defaults are centralized as named constants in `scene_info.py`. Never use magic numbers in `grasp_planner.py` or `pick_teacher.py` — import from `scene_info`.

Key constants:
- `TCP_PINCH_OFFSET_LOCAL = [0, 0, 0]` — zeroed (3-4 mm actual offset hurts more via IK error)
- `DEFAULT_FACE_STANDOFF = 0.003` — 3 mm outward offset to prevent jaw overshoot
- `DEFAULT_CUBE_FACE_CONTACT_SPAN = 0.10` — tangential sampling range on cube faces
- `DEFAULT_GRASP_N_CANDIDATES = 64` — grasp candidates (16 per side face)
- `DEFAULT_PREGRASP_STANDOFF = 0.08` — pregrasp distance along approach
- `DEFAULT_LIFT_HEIGHT = 0.08` — lift distance above grasp contact
- `DEFAULT_ORI_TOL_DEG = 55.0` — IK orientation tolerance
- `DEFAULT_IK_POS_TOL = 0.03` — IK position tolerance (metres)

## Key Design

- **SO-101** arm (5 DOF + 1 gripper) via raw MuJoCo (no robosuite)
- **Position actuators** — joint targets written to `data.ctrl[:]`
- **Trajectory-planned pick** — full trajectory computed on first step via keyframe planner + IK + ruckig
- **Damped least-squares IK** — Cartesian targets -> 5 arm joint angles, with yaw-retry fallbacks
- **Dual cameras**: `scene_cam` (overhead) + `wrist_cam` (on gripper body, TBD)
- **20 Hz control freq**, 10 physics substeps per step (`dt=0.005`)
- **Seeded resets** with cube position randomization
- **Stabilization**: 5 s home-pose settling before recording (configurable)
- **Phase tracking**: each Timestep records `phase_id` from teacher
- **ZMQ bridge**: SimServer exposes env + teacher to HALO runtime (TelemetryStream PUB + CommandRPC REP)
- **Contact slippage suppression**: impratio=10, elliptic cones, noslip solver (see Contact Solver above)
- **Episode generation**: 100% success rate (10/10) with `make generate-episodes`
