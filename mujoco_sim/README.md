# mujoco_sim — SO-101 MuJoCo Simulation

Raw MuJoCo simulation for the SO-101 arm: trajectory-planned pick & place demos, HDF5 episode recording, and a ZMQ server bridging to the HALO runtime.

## Quick start

```bash
uv sync --extra sim                    # install
uv run python -m pytest mujoco_sim/mujoco_sim/tests/ -v  # 116 tests

# Generate episodes
uv run python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes
uv run python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --pick-and-place

# Inspect an episode
uv run python -m mujoco_sim.scripts.inspect_episode

# Start ZMQ sim server
uv run python -m mujoco_sim.server
```

## Robot

SO-101: 5-DOF arm + 1 gripper. 6D joint-position action space (`[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`), written to `data.ctrl[:]` at 20 Hz.

## Skills

**PICK:** `IDLE → MOVE_PREGRASP → EXECUTE_APPROACH → CLOSE_GRIPPER → LIFT → DONE`
Full trajectory pre-computed on first step: 64-candidate grasp planner → SE(3) keyframes → damped least-squares IK → jerk-limited ruckig profiles → clearance validation.

**PLACE:** `SELECT_PLACE → TRANSIT_PREPLACE → DESCEND_PLACE → OPEN → RETREAT → DONE`
Keyframe-planned placement into tray target.

## ZMQ server

| Channel | Pattern | Port | Purpose |
|---------|---------|------|---------|
| Telemetry | PUB | 5560 | Frames + state at 10 Hz |
| Command | REP | 5561 | step, reset, start_pick, start_place, shutdown |

## Episode format (HDF5)

Per episode: `obs/{rgb_scene, rgb_wrist, qpos, qvel, gripper, ee_pose, joint_pos, phase_id, object_pose}` + `action (T, 6)`. Images gzip-compressed.

## Dependencies

Core: `numpy, h5py, tqdm, ruckig`. Optional: `mujoco` (sim), `pyzmq + msgpack` (server), `opencv-python` (viewer).

See [CLAUDE.md](CLAUDE.md) for detailed architecture and API docs.
