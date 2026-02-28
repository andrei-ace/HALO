# mujoco_sim/ — MuJoCo/Robosuite Simulation for HALO

## Overview

MuJoCo/robosuite simulation module for PICK demo generation, episode recording,
offline phase detection, and VCR replay with manual annotation. Sibling to `sim/`
(Isaac Lab) — no bridge to HALO runtime, no ACT training.

See `ROADMAP.md` for the full 6-PR roadmap.

## Structure

```
mujoco_sim/
  __init__.py
  constants.py          # synced from halo.contracts (verified by tests)
  config/
    env_config.py       # EnvConfig dataclass
  env/
    robosuite_env.py    # RobosuiteEnv wrapper (dual cameras, OSC_POSE, seeded resets)
  dataset/
    raw_episode.py      # Timestep (with phase_id) + RawEpisode in-memory buffer
    writer_hdf5.py      # write_episode() — one HDF5 per episode, gzip images
    reader_hdf5.py      # read_episode() — load HDF5 → RawEpisode
  teacher/
    pick_teacher.py     # PickTeacher scripted policy (proportional control, same thresholds as SkillRunnerConfig)
  runner/
    run_teacher.py      # run_teacher() — stabilize → teacher loop → write HDF5
  scripts/
    test_env.py             # acceptance: dump scene.png + wrist.png
    generate_episodes.py    # CLI: --num-episodes, --output-dir, --stabilize
  tests/
    test_constants_sync.py  # verify constants match halo.contracts
    test_raw_episode.py     # Timestep, RawEpisode, HDF5 roundtrip, phase_id
    test_pick_teacher.py    # PickTeacher phases, actions, full episode
```

## Commands

mujoco_sim is a uv workspace member — all commands run from the **repo root** using the shared venv.

```bash
# From repo root
uv run python -m pytest mujoco_sim/mujoco_sim/tests/ -v   # all mujoco_sim tests
uv run python -m pytest tests/test_mujoco_sim_contract_sync.py -v  # contract sync
uv run python -m pytest -v  # includes mujoco_sim tests (in root testpaths)

# Requires robosuite (uv sync --extra mujoco)
uv run python -m mujoco_sim.scripts.test_env  # acceptance: dump scene.png + wrist.png
uv run python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes  # generate demos
```

## Usage

### Quick start — generate demo episodes (CLI)

```bash
# Generate 10 PICK episodes with default settings (5 s stabilization, seed 0-9)
python -m mujoco_sim.scripts.generate_episodes \
    --num-episodes 10 \
    --output-dir episodes \
    --seed-base 0

# Custom stabilization time and control frequency
python -m mujoco_sim.scripts.generate_episodes \
    --num-episodes 50 \
    --output-dir episodes \
    --stabilize 3.0 \
    --control-freq 20 \
    -v  # verbose logging
```

Output: `episodes/ep_000000.hdf5`, `episodes/ep_000001.hdf5`, ...

### Python API — environment only

```python
from mujoco_sim.env import RobosuiteEnv
from mujoco_sim.config import EnvConfig
import numpy as np

# Create environment (Panda arm, Lift task, dual cameras)
env = RobosuiteEnv(EnvConfig())

# Reset with a seed for reproducibility
obs = env.reset(seed=42)
# obs keys: rgb_scene (480,640,3), rgb_wrist (240,320,3),
#           qpos (nq,), qvel (nv,), gripper (float),
#           ee_pose (7,), object_pose (7,)

# Step with a 7-DOF EE-delta action
action = np.zeros(7)         # [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
action[2] = -0.01            # move down
action[6] = 0.0              # gripper open (1.0 = close)
obs, reward, done, info = env.step(action)

# Render from a specific camera
scene_img = env.render()                           # scene camera (default)
wrist_img = env.render(camera="robot0_eye_in_hand") # wrist camera

# State checkpoint / restore (for VCR replay)
state = env.get_state()   # {"qpos": ..., "qvel": ...}
env.set_state(state)      # restore + mj_forward

env.close()
```

### Python API — teacher + recording

```python
from mujoco_sim.env import RobosuiteEnv
from mujoco_sim.config import EnvConfig
from mujoco_sim.teacher import PickTeacher, TeacherConfig
from mujoco_sim.dataset import (
    RawEpisode, Timestep, EpisodeMetadata,
    write_episode, read_episode, episode_path,
)
import numpy as np

env = RobosuiteEnv(EnvConfig())
teacher = PickTeacher(TeacherConfig())

obs = env.reset(seed=42)

# Stabilize: let physics settle for 5 s before recording
for _ in range(100):  # 5 s at 20 Hz
    obs, _, _, _ = env.step(np.zeros(7))

# Record episode
teacher.reset()
episode = RawEpisode(metadata=EpisodeMetadata(seed=42))

for step in range(500):
    action, phase_id, done = teacher.step(obs)

    episode.append(Timestep(
        rgb_scene=obs["rgb_scene"],
        rgb_wrist=obs["rgb_wrist"],
        qpos=obs["qpos"],
        qvel=obs["qvel"],
        gripper=float(obs["gripper"]),
        ee_pose=obs["ee_pose"],
        action=np.array(action, copy=True),
        phase_id=phase_id,
        object_pose=obs.get("object_pose"),
    ))

    if done:
        break
    obs, _, env_done, _ = env.step(action)
    if env_done:
        break

# Save to HDF5
path = write_episode(episode, episode_path("episodes", 0))
print(f"Saved {len(episode)} steps to {path}")

# Load back
loaded = read_episode(path)
print(f"Loaded {len(loaded)} steps, phases: {loaded.phase_ids}")

env.close()
```

### Python API — batch generation (high-level)

```python
from mujoco_sim.runner import run_teacher

# Generate 50 episodes, returns list of HDF5 paths
paths = run_teacher(
    num_episodes=50,
    output_dir="episodes",
    seed_base=0,
    stabilize_seconds=5.0,  # 5 s settling before each episode
    max_steps=500,
)
```

### Inspecting saved episodes

```python
from mujoco_sim.dataset import read_episode

ep = read_episode("episodes/ep_000000.hdf5")

print(f"Steps:        {len(ep)}")
print(f"Seed:         {ep.metadata.seed}")
print(f"Control freq: {ep.metadata.control_freq} Hz")
print(f"Scene shape:  {ep.rgb_scenes.shape}")    # (T, 480, 640, 3)
print(f"Actions:      {ep.actions.shape}")         # (T, 7)
print(f"Phase IDs:    {ep.phase_ids}")             # (T,) int32
print(f"EE poses:     {ep.ee_poses.shape}")        # (T, 7)
print(f"Object poses: {ep.object_poses.shape}")    # (T, 7)

# Access single timestep
ts = ep[0]
print(f"Step 0 — phase={ts.phase_id}, gripper={ts.gripper:.3f}, ee={ts.ee_pose[:3]}")
```

### HDF5 file layout

```
ep_000000.hdf5
├── obs/
│   ├── rgb_scene    (T, H, W, 3) uint8  [gzip]
│   ├── rgb_wrist    (T, H, W, 3) uint8  [gzip]
│   ├── qpos         (T, nq)
│   ├── qvel         (T, nv)
│   ├── gripper      (T,)
│   ├── ee_pose      (T, 7)
│   ├── phase_id     (T,) int32           [if present]
│   ├── object_pose  (T, 7)              [if present]
│   └── contacts/    step_NNNNNN (N,)    [if present]
├── action           (T, 7)
└── attrs: seed, env_name, robot, control_freq, num_steps, created_at
```

### Teacher phase sequence

```
IDLE(0) → SELECT_GRASP(1) → PLAN_APPROACH(2) → MOVE_PREGRASP(3) → VISUAL_ALIGN(4)
→ EXECUTE_APPROACH(5) → CLOSE_GRIPPER(6) → VERIFY_GRASP(7) → LIFT(8) → DONE(9)
```

Phases 1-2 are instant pass-throughs. Distance thresholds (same as SkillRunnerConfig):
- MOVE_PREGRASP → VISUAL_ALIGN: `0.15 m`
- VISUAL_ALIGN → EXECUTE_APPROACH: `0.05 m`
- EXECUTE_APPROACH → CLOSE_GRIPPER: `0.01 m`

Timed phases: CLOSE_GRIPPER (1 s), VERIFY_GRASP (0.5 s), LIFT (2 s).

## Key Design

- **Panda** arm via robosuite (same as sim/ Isaac Lab module)
- **OSC_POSE** controller — 7-DOF `[dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]` EE-frame deltas
- **Dual cameras**: `agentview` (scene) + `robot0_eye_in_hand` (wrist)
- **20 Hz control freq** (robosuite default for OSC_POSE)
- **Seeded resets** for reproducibility
- **constants.py** is a verbatim copy of `sim/halo_sim/constants.py`
- **Stabilization**: 5 s zero-action settling before recording (configurable via `--stabilize`)
- **Phase tracking**: each Timestep records `phase_id` from teacher, persisted in HDF5
