# mujoco_sim/ — SO-101 MuJoCo Simulation for HALO

## Overview

Raw MuJoCo simulation module for SO-101 PICK demo generation, episode recording,
offline phase detection, and VCR replay with manual annotation. Uses damped
least-squares IK for joint-position control. No robosuite dependency.

## Structure

```
mujoco_sim/
  __init__.py
  constants.py          # synced from halo.contracts (verified by tests)
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
    pick_teacher.py     # PickTeacher scripted policy (IK-based, joint-position output)
    ik_helper.py        # Damped least-squares IK: EE position → 5 arm joint angles
  runner/
    run_teacher.py      # run_teacher() — stabilize → teacher loop → write HDF5
  scripts/
    test_env.py             # acceptance: dump scene.png + wrist.png
    generate_episodes.py    # CLI: --num-episodes, --output-dir, --stabilize
    inspect_episode.py      # CLI: inspect HDF5 episode
  assets/
    so101/                  # MJCF + STL meshes (from SO-ARM100 repo)
      so101_new_calib.xml   # SO-101 robot model (position-controlled STS3215 servos)
      pick_scene.xml        # Pick scene: robot + floor + cube + scene_cam
      assets/               # 13 STL mesh files
  tests/
    test_constants_sync.py  # verify constants match halo.contracts
    test_raw_episode.py     # Timestep, RawEpisode, HDF5 roundtrip, phase_id, joint_pos
    test_pick_teacher.py    # PickTeacher phases, actions, full episode (uses real MuJoCo model)
```

## Commands

mujoco_sim is a uv workspace member — all commands run from the **repo root** using the shared venv.

```bash
# From repo root
uv run python -m pytest mujoco_sim/mujoco_sim/tests/ -v   # all mujoco_sim tests
uv run python -m pytest tests/test_mujoco_sim_contract_sync.py -v  # contract sync
uv run python -m pytest -v  # includes mujoco_sim tests (in root testpaths)

# Requires mujoco (uv sync --extra sim)
uv run python -m mujoco_sim.scripts.test_env  # acceptance: dump scene.png + wrist.png
uv run python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes  # generate demos
uv run python -m mujoco_sim.scripts.inspect_episode  # inspect latest episode
uv run python -m mujoco_sim.scripts.inspect_episode data/episodes/20260301_004533/ep_000000.hdf5  # specific file
```

### inspect_episode — episode debug inspector

`inspect_episode.py` prints a diagnostic summary of an HDF5 episode. Use it to verify teacher behaviour after generation. With no arguments it finds the latest episode in `data/episodes/`.

What it reports:
- **Step count, seed, duration, control freq**
- **Phase sequence** with per-phase step ranges and durations
- **Lift check** — verifies cube actually moved up ≥5 cm during LIFT phase (FAILED = grasp problem)
- **EE and cube positions** (first/last) + EE-cube distance over time (start, min, final)
- **Distance at each phase transition** — useful for spotting approach/grasp timing issues
- **Gripper** range (first → last)
- **Action norms** (mean, max, last-50 mean, action dim)
- **Tracker** status (if tracking data present)
- **Video** presence and size

## SO-101 Joint/Actuator Names

| Idx | Joint | Range (rad) |
|-----|-------|-------------|
| 0 | `shoulder_pan` | ±1.92 |
| 1 | `shoulder_lift` | ±1.75 |
| 2 | `elbow_flex` | ±1.69 |
| 3 | `wrist_flex` | ±1.66 |
| 4 | `wrist_roll` | -2.74 to 2.84 |
| 5 | `gripper` | -0.17 to 1.75 |

EE site: `gripperframe` on the `gripper` body.

## Action Space

6D joint-position targets: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

Written directly to `data.ctrl[:]`. Position actuators with `kp=998.22` track targets.

Gripper semantics: `-0.17` = fully open, `1.75` = fully closed (joint angle, rad).

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

### Teacher phase sequence

```
IDLE(0) → SELECT_GRASP(1) → PLAN_APPROACH(2) → MOVE_PREGRASP(3) → VISUAL_ALIGN(4)
→ EXECUTE_APPROACH(5) → CLOSE_GRIPPER(6) → VERIFY_GRASP(7) → LIFT(8) → DONE(9)
```

### HDF5 file layout

```
ep_000000.hdf5
├── obs/
│   ├── rgb_scene    (T, H, W, 3) uint8  [gzip]
│   ├── rgb_wrist    (T, H, W, 3) uint8  [gzip]
│   ├── qpos         (T, 13)
│   ├── qvel         (T, 12)
│   ├── gripper      (T,)
│   ├── ee_pose      (T, 7)
│   ├── joint_pos    (T, 6)             [if present]
│   ├── phase_id     (T,) int32         [if present]
│   ├── object_pose  (T, 7)             [if present]
│   └── contacts/    step_NNNNNN (N,)   [if present]
├── action           (T, 6)
└── attrs: seed, env_name, robot, control_freq, num_steps, created_at
```

## Key Design

- **SO-101** arm (5 DOF + 1 gripper) via raw MuJoCo (no robosuite)
- **Position actuators** — joint targets written to `data.ctrl[:]`
- **Damped least-squares IK** — Cartesian targets → 5 arm joint angles
- **Dual cameras**: `scene_cam` (overhead) + `wrist_cam` (on gripper body, TBD)
- **20 Hz control freq**, 10 physics substeps per step (`dt=0.005`)
- **Seeded resets** with cube position randomization
- **Stabilization**: 5 s home-pose settling before recording (configurable)
- **Phase tracking**: each Timestep records `phase_id` from teacher
