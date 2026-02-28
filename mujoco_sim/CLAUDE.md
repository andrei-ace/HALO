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
    raw_episode.py      # Timestep dataclass + RawEpisode in-memory buffer
    writer_hdf5.py      # write_episode() — one HDF5 per episode, gzip images
    reader_hdf5.py      # read_episode() — load HDF5 → RawEpisode
  scripts/
    test_env.py         # acceptance: dump scene.png + wrist.png
  tests/
    test_constants_sync.py  # verify constants match halo.contracts
    test_raw_episode.py     # Timestep, RawEpisode, HDF5 roundtrip
```

## Commands

```bash
uv sync --extra dev               # install + dev deps
uv run python -m pytest tests/    # run tests
uv run python -m mujoco_sim.scripts.test_env  # acceptance script

# From repo root — verify contract sync (no robosuite needed)
uv run python -m pytest tests/test_mujoco_sim_contract_sync.py -v
```

## Key Design

- **Panda** arm via robosuite (same as sim/ Isaac Lab module)
- **OSC_POSE** controller — 7-DOF `[dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]` EE-frame deltas
- **Dual cameras**: `agentview` (scene) + `robot0_eye_in_hand` (wrist)
- **20 Hz control freq** (robosuite default for OSC_POSE)
- **Seeded resets** for reproducibility
- **constants.py** is a verbatim copy of `sim/halo_sim/constants.py`
