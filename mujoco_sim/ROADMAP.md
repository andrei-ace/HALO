# MuJoCo Simulation Module — Full Roadmap

## Context

HALO needs a MuJoCo simulation module for data generation and episode replay/annotation. This is a sibling to `sim/` (Isaac Lab) but focused on: teacher generates PICK demos → episodes recorded to HDF5 → offline phase detection → VCR replay with manual annotation. No ACT training, no tracking migration, no bridge to HALO runtime. Module lives at `mujoco_sim/` at the repo root with its own `pyproject.toml`.

## Directory Structure (all PRs)

```
mujoco_sim/
├── CLAUDE.md
├── pyproject.toml
└── mujoco_sim/
    ├── __init__.py
    ├── constants.py
    ├── config/
    │   ├── __init__.py
    │   └── env_config.py
    ├── env/
    │   ├── __init__.py
    │   ├── so101_env.py
    │   └── robosuite_env.py          # legacy, not exported
    ├── dataset/                       # PR2
    │   ├── __init__.py
    │   ├── raw_episode.py
    │   ├── writer_hdf5.py
    │   └── reader_hdf5.py
    ├── teacher/                       # PR3
    │   ├── __init__.py
    │   ├── ik_helper.py
    │   └── pick_teacher.py
    ├── runner/                        # PR3
    │   ├── __init__.py
    │   └── run_teacher.py
    ├── fsm/                           # PR4
    │   ├── __init__.py
    │   ├── guards.py
    │   └── pick_phase_detector.py
    ├── replay/                        # PR5
    │   ├── __init__.py
    │   ├── vcr_player.py
    │   ├── viewer_app.py
    │   └── phase_overlay.py
    ├── annotator/                     # PR6
    │   ├── __init__.py
    │   ├── phase_track.py
    │   ├── annotation_ui.py
    │   └── annotation_io.py
    ├── server/                        # ZMQ sim server
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── config.py
    │   ├── handlers.py
    │   └── protocol.py
    ├── scripts/
    │   ├── __init__.py
    │   ├── test_env.py                # PR1
    │   ├── generate_episodes.py       # PR3
    │   ├── inspect_episode.py         # PR3
    │   ├── detect_phases.py           # PR4
    │   ├── replay_episode.py          # PR5
    │   └── annotate_episode.py        # PR6
    └── tests/
        ├── __init__.py
        ├── test_constants_sync.py     # PR1
        ├── test_raw_episode.py        # PR2
        ├── test_pick_teacher.py       # PR3
        ├── test_server.py             # server tests
        ├── test_guards.py            # PR4
        ├── test_phase_detector.py     # PR4
        ├── test_phase_track.py        # PR6
        └── test_annotation_io.py      # PR6
```

## PR Dependency Graph

```
PR1 (env + constants)
 ├─ PR2 (recording format)
 │   ├─ PR3 (teacher)     [PR1 + PR2]
 │   └─ PR4 (phase FSM)   [PR2]
 │        ├─ PR5 (VCR)    [PR1 + PR2 + PR4]
 │        └─ PR6 (annot)  [PR4 + PR5]
```

PR3 and PR4 can be developed in parallel after PR2.

## PR Summaries

### PR1 — Environment + Constants ✅

Foundation: `SO101Env` wrapper with dual cameras, seeded resets, 6-DOF joint-position action space for SO-101.

**Delivered:**
- `constants.py` — phase IDs, action fields, gripper semantics, timing (synced from `halo.contracts`)
- `config/env_config.py` — `EnvConfig` dataclass (SO-101, position actuators, dual cameras, 20 Hz)
- `env/so101_env.py` — `SO101Env` wrapper: reset, step, render, state get/set (raw MuJoCo)
- `scripts/test_env.py` — acceptance script (dump images, verify seeded reproducibility)
- `tests/test_constants_sync.py` — 5 tests validating all constants
- Root-level `tests/test_mujoco_sim_contract_sync.py` — cross-module contract sync

### PR2 — Episode Recording Format ✅

**Timestep** dataclass: `rgb_scene(H,W,3)`, `rgb_wrist(H,W,3)`, `qpos(nq)`, `qvel(nv)`, `gripper(float)`, `ee_pose(7)`, `action(7)`, optional `phase_id(int)`, `object_pose(7)`, `contacts(N)`

**RawEpisode** — in-memory buffer with `append(Timestep)`, indexing, bulk numpy accessors

**HDF5Writer** — one episode per file (`episodes/ep_NNNNNN.hdf5`), gzip on images, metadata as attrs

**HDF5Reader** — `load(path) → RawEpisode`

**Delivered:**
- `dataset/raw_episode.py` — `Timestep` dataclass + `EpisodeMetadata` + `RawEpisode` with bulk accessors
- `dataset/writer_hdf5.py` — `write_episode()` + `episode_path()` helper
- `dataset/reader_hdf5.py` — `read_episode()` (inverse of writer)
- `dataset/__init__.py` — public exports
- `tests/test_raw_episode.py` — 24 tests (Timestep, RawEpisode, HDF5 roundtrip, gzip, metadata)
- Fixed `pyproject.toml` — added `[tool.hatch.build.targets.wheel]` packages + correct testpaths

### PR3 — Teacher Runner ✅

**PickTeacher** — scripted PICK using privileged sim state (ground-truth cube_pos, ee_pos) with damped least-squares IK. `step(obs, model, data) → (action[6], phase_id, done)`

**run_teacher** — loop: reset env → stabilize (5 s) → run teacher → write HDF5

**Delivered:**
- `teacher/pick_teacher.py` — `PickTeacher` + `TeacherConfig` (proportional control, same thresholds as SkillRunnerConfig)
- `runner/run_teacher.py` — `run_teacher()` with stabilization phase + `_run_single_episode()` helper
- `scripts/generate_episodes.py` — CLI: `python -m mujoco_sim.scripts.generate_episodes --num-episodes 10`
- `tests/test_pick_teacher.py` — 19 tests (init, phase transitions, action output, full episode)
- `Timestep.phase_id` field added — persisted in HDF5, roundtripped in reader
- Stabilization: 5 s of zero-action steps before recording to let physics settle

### PR4 — Offline Phase Detection (current)

**PickPhaseDetector** — forward pass over episode timeline. Uses distance thresholds from SkillRunnerConfig. Produces `phase_ids[T]` + `segments`. Saves as sidecar `ep_NNNNNN.phase.json`.

**guards.py** — pure functions: `ee_to_object_distance`, `gripper_is_closed`, `object_is_lifted`

### PR5 — VCR Replay Engine

**VCRPlayer** — cursor-based navigation, state injection via qpos/qvel + mj_forward

**ViewerApp** — `mujoco.viewer.launch_passive` with key_callback. SPACE=play/pause, arrows=step, J/L=speed, R=rewind

**PhaseOverlay** — text overlay + timeline bar via separate OpenCV window

### PR6 — Manual Phase Annotation

**PhaseTrack** — full-length phase_ids array + sorted boundary list. "Paint forward" semantic.

**AnnotationUI** — 0-9 keys assign phase, B=boundary, U=undo, T=toggle auto/manual, S=save

**annotation_io** — `.annotation.json` with both auto and manual tracks

## End-to-End Pipeline (after all PRs)

```bash
python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes
python -m mujoco_sim.scripts.detect_phases --input-dir episodes
python -m mujoco_sim.scripts.replay_episode --episode episodes/ep_000001.hdf5
python -m mujoco_sim.scripts.annotate_episode --episode episodes/ep_000001.hdf5
```
