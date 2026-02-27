# sim/ — Isaac Lab Extension for HALO

## Overview

This package contains the Isaac Lab environment, teacher pipeline, and ZeroMQ bridge
for running HALO's PICK skill in simulation. It runs on a remote A6000 48GB GPU machine
inside Isaac Sim's Python environment.

## Pinned versions

- **Isaac Sim 4.5.0**
- **Isaac Lab 2.1.0**

Install:
```bash
pip install isaacsim==4.5.0
git clone https://github.com/isaac-sim/IsaacLab.git -b v2.1.0
cd IsaacLab && ./isaaclab.sh --install
```

## Structure

```
halo_sim/
  constants.py          # synced from halo.contracts (verified by tests/test_sim_contract_sync.py)
  cfg/
    robot_cfg.py        # Franka Panda placeholder (swap SO-ARM101 URDF later)
    scene_cfg.py        # table + cube + cameras
    env_cfg.py          # PickPlaceEnvCfg (DirectRLEnvCfg)
    teacher_cfg.py      # teacher tolerances/speeds
    domain_rand.py      # V0 mild randomization
  envs/
    pick_env.py         # DirectRLEnv — PICK only
  teacher/
    teacher_fsm.py      # mirrors HALO PickFSM phases (privileged sim state)
    ik_teacher.py       # target pose -> EE-frame delta actions
  data/
    schema.py           # dataset field definitions
    recorder.py         # sharded HDF5 recorder
  bridge/
    sim_server.py       # ZeroMQ server (runs in Isaac Sim Python)
    protocol.py         # message schema (shared between both sides)
  scripts/
    generate_demos.py   # teacher -> sharded HDF5
    replay_demos.py     # replay parity verification
    test_robot_spawn.py # minimal spawn + wiggle test
    run_bridge.py       # launch sim server for HALO connection
```

## Key design

- **Franka Panda** is the placeholder arm (swap SO-ARM101 URDF later)
- **10 Hz control rate** (physics dt=0.02, decimation=5)
- **Action space**: 7-DOF `[dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]` EE-frame deltas
- **64 parallel envs** for teacher demo generation (A6000 48GB)
- **Wrist camera gating**: black image when phase not in WRIST_ACTIVE_PHASES
- **Single-env mode** for HALO bridge (HALO drives one arm at a time)

## Commands

All scripts below are **stubs** that raise `NotImplementedError` until wired to
a live Isaac Lab session. They document the intended CLI interface.

```bash
# Teacher demo generation (batched, no HALO bridge)  [STUB]
python -m halo_sim.scripts.generate_demos --num-episodes 10000 --num-envs 64

# Replay verification  [STUB]
python -m halo_sim.scripts.replay_demos --dataset datasets/demos_v0_manifest.json --samples 100

# Spawn test  [STUB]
python -m halo_sim.scripts.test_robot_spawn

# Bridge mode (HALO connects via ZeroMQ)  [STUB]
python -m halo_sim.scripts.run_bridge --action-port 5555 --obs-port 5556
```
