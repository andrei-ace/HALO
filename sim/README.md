# sim/ — Isaac Lab Integration (Planned)

Isaac Lab sim support is a future phase. The current sim integration uses **raw MuJoCo + SO-101** (see `mujoco_sim/`).

## Phased Sim Strategy

1. **MuJoCo + SO-101** (current) — `mujoco_sim/` package. Teacher demos, ACT training, closed-loop eval.
2. **Isaac Lab** (future) — this directory. GPU-accelerated parallel envs, domain randomization at scale, sim-to-real transfer.
3. **Real SO-ARM101 hardware** (later) — same dataset schema and action space, swap sensor sources and controller.

## What Was Here

This directory previously contained scaffolded stubs for an Isaac Lab extension (`halo_sim/` package):

- `envs/pick_env.py` — DirectRLEnv stub for PICK skill
- `teacher/teacher_fsm.py` — analytic teacher FSM (mirrors HALO PickFSM phases using privileged sim state)
- `teacher/ik_teacher.py` — target EE pose to joint-position action converter
- `data/schema.py` + `recorder.py` — dataset field definitions + sharded HDF5 recorder
- `bridge/protocol.py` + `sim_server.py` — ZeroMQ message schema + sim-side server
- `cfg/` — env, robot (SO-101 placeholder), scene, teacher, domain randomization configs
- `scripts/` — generate_demos, replay_demos, test_robot_spawn, run_bridge (all stubs)
- `constants.py` — phase IDs, action fields synced from `halo.contracts` (verified by `test_sim_contract_sync.py`)

All scripts were stubs raising `NotImplementedError`. The scaffolding was removed in favor of completing the MuJoCo integration first. The design patterns are captured in `docs/halo_architecture.md` (sim strategy section) and the MuJoCo implementation in `mujoco_sim/`.

## When to Implement

After the MuJoCo pipeline is validated end-to-end (teacher demos → ACT training → closed-loop eval), the Isaac Lab integration will be re-scaffolded here with:

- Isaac Sim 4.5.0 + Isaac Lab 2.1.0 (or latest stable)
- 64 parallel envs on A6000 48GB for batched demo generation
- Same action space: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` 6D joint-position targets at 10 Hz
- Wrist camera gating (black image when phase not in WRIST_ACTIVE_PHASES)
- ZeroMQ bridge reusing `halo/bridge/` adapters (SimClient, SimSource)
- Sharded HDF5 dataset recording with manifest

## Key Contracts (preserved in halo.contracts)

The contracts that the Isaac Lab extension must satisfy are defined in:
- `halo/contracts/enums.py` — PhaseId, WRIST_ACTIVE_PHASES
- `halo/contracts/actions.py` — JointPositionAction, JointPositionChunk, ZERO_JOINT_ACTION, SO101_DOF
- `halo/bridge/` — ZeroMQ transport protocol (config, apply, observe, chunk, transforms)
