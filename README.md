![HALO — Hierarchical Adaptive LLM-Operated Robot](docs/imgs/Logo.png)

▶️ [**Watch the demo on YouTube**](https://www.youtube.com/watch?v=hIvHln6MW2w)

HALO is a robotic manipulation system that decouples continuous motor control from LLM-based task reasoning. The robot never pauses motion waiting for the planner — perception and control run machine-to-machine at 10-100 Hz, while an LLM agent orchestrates skills asynchronously. Safety-critical decisions live outside the LLM loop entirely.

The architecture is robot-agnostic — any 5+ DOF arm with a gripper can be integrated by providing an IK solver and controller mapping. The current development target is the [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) (5-DOF + 1-DOF gripper), validated in MuJoCo simulation.

Operators interact with the robot through **natural voice and text conversation** powered by the **Gemini Live API**. A Live Agent narrates what the robot is doing, answers questions about the scene, and translates spoken instructions into planner actions — no programming or GUI needed. The system supports both local inference (Ollama) and cloud backends (Google Gemini via Cloud Run), with automatic failover between them.

## Key Features

- **Live Agent (Gemini Live API)** — conversational voice/text interface for operator interaction; narrates robot actions, answers scene questions, forwards intents to the planner via proxy-tool architecture; multilingual with session memory
- **Cognitive backend switching** — Switchboard routes LLM/VLM calls to LOCAL (Ollama) or CLOUD (Gemini), with automatic failover/failback and split-brain prevention via LeaseManager
- **Voice interaction** — bidirectional audio streaming (16 kHz capture / 24 kHz playback) with barge-in support and real-time transcription
- **Small, fast models** — runs on modest hardware: planner uses a 20B-parameter LLM (`gpt-oss:20b`), perception uses a 3B-parameter VLM (`qwen2.5vl:3b`), cloud uses Gemini 3.1 Flash-Lite — no large frontier models needed
- **LLM task planner** — ADK ReAct agent that orchestrates pick/place/track skills via async commands
- **Continuous control** — 50-100 Hz action streaming with temporal ensembling, independent of LLM latency
- **Dual perception pipeline** — fast tracking loop (10-30 Hz) + async VLM scene analysis (off critical path)
- **Deterministic safety** — per-timestep delta clamping, hint freshness gating, reflex layer; LLM cannot bypass
- **Visual FSM engine** — skill state machines defined as Mermaid diagrams, executed by a generic FSM engine
- **MuJoCo simulation** — robot-agnostic env with trajectory-planned teachers, 64-candidate grasp planner, jerk-limited motion, autonomous ZMQ sim server
- **Terminal UI** — Textual-based TUI with mock, live-local, and live-cloud modes
- **JSONL observability** — per-session run logs with full event and VLM result capture

## System Overview

![HALO System Architecture](docs/imgs/architecture.svg)

## Screenshots

![MuJoCo simulation — SO-101 arm with scene objects](docs/imgs/mujoco-sim-scene.png)

![TUI during a pick skill with embedded sim view and FSM progress](docs/imgs/tui-live-pick-lift.png)

![TUI during a place skill — arm carrying cube to tray](docs/imgs/tui-live-place-transit.png)

## Prerequisites

- **Python 3.13+** and [**uv**](https://docs.astral.sh/uv/) package manager
- `make install` — installs all dependencies (including MuJoCo)

## Quickstart

```bash
make install
```

### Ollama setup (local inference)

Live-local and integration tests require [Ollama](https://ollama.com). Install it, then pull the two models:

```bash
# macOS / Linux — install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the planner LLM and the perception VLM
ollama pull gpt-oss:20b        # 20B planner (used by PlannerService)
ollama pull qwen2.5vl:3b       # 3B VLM (used by TargetPerceptionService)
```

Verify Ollama is running and models are available:

```bash
ollama list   # should show both models
```

### Running

#### Cloud mode — deployed Cloud Run (recommended)

```bash
make tui-live-cloud    # reads URL + SA from terraform outputs
```

See [GCP Deployment](#gcp-deployment) below for setup.

#### Cloud mode — local development (2 terminals)

```bash
GOOGLE_API_KEY=<key> make run-cloud-service   # terminal 1: cloud service (Gemini)
make tui-live-cloud-local                      # terminal 2: TUI (sim server auto-started)
```

#### Local mode (Ollama + MuJoCo sim)

```bash
make tui-live          # starts sim server automatically, connects to local Ollama
```

#### Mock mode (no external services needed)

```bash
make tui-mock
```

The MuJoCo sim server is spawned automatically by the TUI in managed mode (`--source mujoco`). Use `make sim-server` only if you need to run it standalone.

## GCP Deployment

The cloud service deploys to Google Cloud Run with Terraform. It uses **Gemini 3.1 Flash-Lite** for both planner decisions and VLM scene analysis, plus **Gemini 2.5 Flash Live Preview** for the Live Agent's bidirectional audio — fast, cheap models that keep latency low and costs minimal.

### Prerequisites

- GCP account with billing enabled, [gcloud CLI](https://cloud.google.com/sdk/docs/install), [Terraform >= 1.5](https://developer.hashicorp.com/terraform/install)
- A [Gemini API key](https://aistudio.google.com/apikey)
- GCP permissions: **Owner**, or at minimum `serviceUsageAdmin`, `iam.serviceAccountAdmin`, `artifactregistry.admin`, `secretmanager.admin`, `run.admin`, `datastore.owner`

Authenticate and enable the bootstrap API (required before Terraform can manage the rest):

```bash
gcloud auth application-default login

PROJECT_ID=your-project-id
gcloud services enable serviceusage.googleapis.com    --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com              --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com    --project=$PROJECT_ID
gcloud services enable firestore.googleapis.com        --project=$PROJECT_ID
gcloud services enable iam.googleapis.com              --project=$PROJECT_ID
```

### Deploy

```bash
# 1. Configure Terraform variables
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Edit terraform.tfvars — set project_id (and optionally invoker_impersonators)

# 2. Bootstrap infrastructure (registry, secrets, SAs, Firestore — no Cloud Run yet)
make tf-init
make tf-bootstrap

# 3. Configure Docker auth for Artifact Registry (one-time)
gcloud auth configure-docker $(cd infra && terraform output -raw artifact_registry | cut -d/ -f1)

# 4. Add your Gemini API key to Secret Manager (one-time)
echo -n "YOUR_GEMINI_KEY" | gcloud secrets versions add google-api-key \
  --project=$(cd infra && terraform output -raw project_id) --data-file=-

# 5. Build image, push to registry, deploy Cloud Run service
make deploy-cloud
```

Subsequent deploys only need `make deploy-cloud`. To rotate the API key, repeat step 4.

### Connect the TUI

The TUI authenticates to Cloud Run by impersonating the invoker service account. Add your email to `invoker_impersonators` in `infra/terraform.tfvars`:

```hcl
invoker_impersonators = ["user:you@example.com"]
```

Then apply and connect:

```bash
cd infra && terraform apply && cd ..
make tui-live-cloud    # reads URL + SA from terraform outputs
```

See [cloud_service/README.md](cloud_service/README.md) for endpoints, env vars, local testing, and key-file auth. See [infra/README.md](infra/README.md) for Terraform variables and resource details.

## Training Pipeline

Adding a new skill to HALO follows a repeatable workflow:

1. **Define the FSM** — author a Mermaid stateDiagram-v2 in `configs/skills/<skill>/default.mmd`
2. **Write a teacher** — MuJoCo solver that produces trajectories and signals the FSM phase per timestep
3. **Generate episodes** — run the teacher across randomised scenes to produce HDF5 datasets
4. **Train ACT** — train an Action Chunking with Transformers model on the episodes
5. **Build detectors** — replace teacher phase signals with sensor-based detectors for runtime FSM transitions

### Teacher Episodes

Teacher episodes are generated in MuJoCo using trajectory-planned demonstrations. The teacher solves the full trajectory (grasp planning → IK → jerk-limited motion profiles) and labels each timestep with the corresponding FSM `phase_id`.

```bash
# Generate 16 pick episodes (default)
make generate-episodes

# Generate with video previews (requires opencv)
make generate-episodes-video

# Generate pick-and-place episodes with video
make generate-episodes-place
```

Tune with environment variables:

```bash
make generate-episodes EPISODES=100 SEED_BASE=0 EPISODE_DIR=data/episodes
```

- `EPISODES` — number of episodes to generate (default: 16)
- `SEED_BASE` — starting random seed for reproducibility (default: 0)
- `EPISODE_DIR` — output directory (default: `data/episodes`)

### Episode Format

All episode sources share the same observation and action schema:

```
ep_NNNNNN.hdf5
├── obs/
│   ├── rgb_scene     (T, H, W, 3) uint8   — scene camera image
│   ├── rgb_wrist     (T, H, W, 3) uint8   — wrist camera image
│   ├── qpos          (T, nq)      float64 — full joint positions (arm + objects)
│   ├── qvel          (T, nv)      float64 — joint velocities
│   ├── gripper       (T,)         float64 — gripper angle
│   ├── ee_pose       (T, 7)       float64 — end-effector [x,y,z, qx,qy,qz,qw]
│   ├── joint_pos     (T, 6)       float64 — actuated arm joints (optional)
│   ├── phase_id      (T,)         int32   — FSM state label per timestep (optional)
│   ├── object_pose   (T, 7)       float64 — target object pose (optional)
│   ├── bbox_xywh     (T, 4)       int32   — tracker bounding box (optional)
│   ├── tracker_ok    (T,)         bool    — tracker status (optional)
│   └── contacts/                          — per-step contact data (optional, sparse)
├── action            (T, 6)       float64 — joint-position targets (see below)
└── attrs: seed, env_name, robot, control_freq, num_steps, created_at
```

**Action format:** `(T, 6)` — joint-position targets `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`. The same 6D joint-position action space is used everywhere: MuJoCo teacher episodes, ACT training, runtime inference, and temporal ensembling.

RGB datasets use gzip-4 compression. All poses are position + quaternion (scipy convention). See [mujoco_sim/CLAUDE.md](mujoco_sim/CLAUDE.md) for full field details.

### Teleoperation Recording (planned)

An operator will drive the robot via keyboard or gamepad through the ZMQ bridge, recording episodes in the same HDF5 format as teacher demos. The `phase_id` field is left blank during recording and annotated post-hoc using VCR (below) or with live hotkeys. Combining teacher and teleop episodes produces richer training data with more diverse strategies.

### VCR — Episode Replay & Annotation (planned)

VCR is a Textual TUI tool for replaying and annotating recorded episodes:

- **Replay** — scrub through episodes frame-by-frame with scene camera, wrist camera, joint states, gripper position, and phase timeline
- **Annotate** — label FSM phase boundaries on each timestep; annotations are stored in a sidecar JSON file (HDF5 stays immutable)
- **Use cases** — review and correct teacher labels, annotate teleop episodes, create detector training data

### ACT Model Training (planned)

[Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/) (ACT) is the imitation learning backbone:

- **Inputs**: wrist camera + scene camera + proprioception (joint positions + gripper)
- **Output**: action chunks — a sequence of future joint-position targets
- **Training data**: teacher + teleop episodes in the HDF5 format above
- **v0 config**: 6D joint-space actions, chunk length 10 at 20 Hz (0.5 s horizon)

### Phase Detectors (planned)

During teacher episodes, `phase_id` comes from the teacher solver (ground truth). During ACT inference, **detectors** replace teacher signals to drive FSM state transitions at runtime:

- **Gripper current/position** — gripping something? Fully closed on air?
- **Wrist camera** — object visible? Grip looks successful?
- **Joint state** — reached target position within tolerance?
- **Contact forces** — touching the object? (sim only initially)

Heuristic detectors first, learned detectors later (trained on VCR-annotated episodes).

### Closed-Loop Evaluation (planned)

Run ACT + detectors in MuJoCo sim with the FSM engine orchestrating phase transitions from detector signals. Measure pick success rate, place accuracy, and cycle time against the teacher baseline.

## Project Status

| Component | Status |
|---|---|
| Contracts, Runtime, EventBus, CommandRouter | Done |
| ControlService + TemporalEnsembling + SafetyGuard | Done |
| SkillRunnerService + Mermaid FSM engine | Done |
| PlannerService + ADK ReAct agent | Done |
| TargetPerceptionService (mock + VLM pipeline) | Done |
| Cognitive backend switching (Switchboard, LeaseManager) | Done |
| TUI (mock + live modes) + RunLogger | Done |
| ZMQ bridge to MuJoCo sim | Done |
| MuJoCo sim (SO-101 env, teachers, grasp planner, SimServer) | Done |
| Integration tests (Ollama-backed) | Done |
| ACT model + training pipeline | Planned |
| Phase detectors (runtime FSM state detection) | Planned |
| VCR (episode replay + annotation tool) | Planned |
| Teleoperation recording | Planned |
| Isaac Lab extension (GPU-accelerated parallel envs) | Planned |
| Sim-to-real transfer + real hardware deployment | Planned |

## Repository Structure

```
halo/                  # Core runtime, services, contracts, TUI
  contracts/           # Enums, snapshots, commands, events, actions + JSON schemas
  runtime/             # StateStore, EventBus, CommandRouter, HALORuntime
  services/            # PlannerService, SkillRunnerService, ControlService, TargetPerceptionService
  cognitive/           # Switchboard, LeaseManager, ContextStore, local/remote backends
  bridge/              # ZMQ 2-channel bridge to MuJoCo sim
  tui/                 # Textual TUI app + RunLogger
  configs/             # Planner/perception prompts, Mermaid FSM definitions
  models/              # ACT model, training, inference (planned)
  detectors/           # Phase detectors for runtime FSM state detection (planned)
vcr/                   # Episode replay + annotation TUI (planned)
mujoco_sim/            # MuJoCo + SO-101 sim (env, teachers, SimServer)
  teleop/              # Teleoperation recording (planned)
cloud_service/         # Cloud Run service (Gemini planner + VLM + Live Agent)
infra/                 # Terraform GCP configuration
tests/                 # Unit tests (~740 HALO + 116 sim + 20 cloud)
integration/           # LLM integration tests (require Ollama)
docs/                  # Architecture and developer reference
```

## Documentation

- [Architecture](docs/halo_architecture.md) — system design, dataflows, safety, cloud integration
- [Developer Reference](docs/README.md) — repo structure, service internals, testing, workflow
- [MuJoCo Sim](mujoco_sim/CLAUDE.md) — env, dataset format, grasp planner, SimServer
- [Cloud Service](cloud_service/README.md) — endpoints, Live Agent, deployment
- [Infrastructure](infra/README.md) — Terraform GCP setup
