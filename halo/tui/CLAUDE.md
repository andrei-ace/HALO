# TUI

Textual-based operator terminal for HALO. Primary interface for monitoring robot state, talking to the planner, and controlling skills in mock, local, and cloud modes.

## Files

| File | Purpose |
|------|---------|
| `app.py` | `HALOApp` (Textual `App`), all panel widgets, CLI entry point, live wiring, agent loop |
| `run_logger.py` | `RunLogger` — writes JSONL session logs + VLM images + scene text to `runs/` |
| `feed_viewer.py` | `FeedViewer` — OpenCV `imshow` window in a subprocess, tracker bbox overlay, wrist PIP |
| `fsm_overlay.py` | `render_fsm_overlay()` — pure function: FSM dict → BGR numpy image (composited below camera feed) |

## Modes

| Mode | Flag / make target | Runtime | Agent | Services |
|------|-------------------|---------|-------|----------|
| **Mock** | `python -m halo.tui.app` / `make tui-mock` | None | None | None — static `_DATA` fixture |
| **Live local** | `--live` / `make tui-live` | HALORuntime | Switchboard (LOCAL) | Perception + SkillRunner (if mujoco) |
| **Live cloud** | `--live --cloud-url <url>` / `make tui-live-cloud` | HALORuntime | Switchboard (CLOUD→LOCAL failover) | Perception + SkillRunner (if mujoco) |
| **Live cloud local** | `make tui-live-cloud-local` | HALORuntime | Switchboard (CLOUD) | Same as cloud, localhost URL |

The `--live-agent` flag adds `LiveAgentClient` + audio capture/playback for voice interaction via the cloud service WebSocket.

## CLI Args (live mode)

| Arg | Default | Description |
|-----|---------|-------------|
| `--arm` | `arm0` | Arm ID |
| `--model` | (from `LocalConfig`) | Planner LLM model name |
| `--vlm-model` | (from `LocalConfig`) | VLM model name |
| `--base-url` | `http://localhost:11434` | Ollama base URL |
| `--source` | `videoloop` | Video source type (`videoloop` or `mujoco`) |
| `--cloud-url` | (none) | Cloud service URL — enables cloud mode |
| `--sa-key-file` | (none) | GCP service account key file for IAM auth |
| `--sa-email` | (none) | GCP service account email |
| `--live-agent` | (flag) | Enable Live Agent voice/text via cloud WS |
| `--screenshot <path>` | (none) | Headless SVG screenshot then exit |

## UI Layout

Two-column layout with title bar and hint bar:

```
┌─────────────────────────────────────────────────────┐
│                HALO — arm0                          │  TitleBar
├───────────────────────┬─────────────────────────────┤
│  Skill Runner         │  System                     │
│  Target Perception    │  Joints (6DOF)              │
│                       │  Control Service            │
│                       │  Events                     │
│  Talk to Planner      ├─────────────────────────────┤
│  [Voice] (if live-agent)│  Panic                    │
├───────────────────────┴─────────────────────────────┤
│  [ ? ] legend · Tab navigate · T type · ...         │  HintBar
└─────────────────────────────────────────────────────┘
```

Left column is `3fr`, right column is `2fr`. Info panels shrink first; Talk + Panic hold their space.

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Navigate between input and buttons |
| `T` | Focus the message input directly |
| `Enter` | Send message to planner |
| `Esc` | Clear input and return to monitoring mode |
| `R` | Show full planner reasoning (modal) |
| `Y` | Yank — copy last reasoning to clipboard |
| `F` | Toggle OpenCV camera feed viewer (live only) |
| `M` | Toggle microphone mute (live agent only) |
| `Ctrl+A` | Emergency abort — blows the safety fuse |
| `Ctrl+R` | Reset fuse — re-enable FSM / skill execution |
| `Ctrl+Q` | Quit |
| `?` | Show / hide keyboard legend |

## Panel Widgets

| Widget | ID | Shows |
|--------|----|-------|
| `PlannerPanel` | `#planner-panel` | Skill name/run ID, phase, ACT status/buffer, outcome, elapsed |
| `TargetPerceptionPanel` | `#perception-panel` | Tracking status, target handle, center px, distance, confidence, obs age, failure code, active/pending tracker buffers |
| `ControlServicePanel` | `#control-panel` | Control status, safety state, Δ EE action + gripper |
| `ServosPanel` | `#servos-panel` | 6-DOF joint positions + velocity bars (SO-101) |
| `TalkPanel` | `#talk-panel` | Prompt history (scrollable) + text input |
| `SystemPanel` | `#system-panel` | Service status dots + backend indicator (LOCAL/CLOUD) |
| `EventsPanel` | `#events-panel` | Last 8 EventBus events (newest at bottom, trims automatically) |
| `AudioPanel` | `#audio-panel` | Mic/speaker status, VU bars, transcription in/out (hidden unless live-agent/live-backend) |
| `PanicPanel` | `#panic-panel` | ABORT NOW! + RESET buttons |

Modals: `LegendScreen` (keybinding help), `ReasoningScreen` (full LLM reasoning text, Y to copy).

## Feed Viewer

- Runs in a **subprocess** (`multiprocessing.Process`, spawn context) so macOS Cocoa gets its own main thread.
- Parent pushes frames + annotations via `Pipe` (no Queue/Event/Lock — avoids POSIX semaphore issues on Python 3.14+).
- A **pusher thread** (`_push_state`) reads `RuntimeStateStore` target/perception info + `SkillRunnerService.get_view_model()` at 30 Hz and sends `(frame, wrist_thumb, target, perception, fsm_dict)` tuples.
- Subprocess draws tracker bbox annotations (`_draw_annotations`), wrist PIP thumbnail (`_composite_wrist_pip`), and FSM graph overlay (`render_fsm_overlay` from `fsm_overlay.py`).
- Toggle with **F** key; requires `uv sync --extra viewer` (opencv-python).

## FSM Overlay

Pure function `render_fsm_overlay(fsm_dict, width, height) → np.ndarray` (BGR).

- Topological sort of main-path nodes, recovery nodes placed in source column +1.
- Layout cached by `(skill_name, variant, node_count, width)`.
- Colour-coded nodes (PENDING grey, ACTIVE green, COMPLETED dim green, FAILED red) + edges (taken green, recovery yellow, fail red).
- Header shows skill name + target + outcome badge.
- Previous skill recap (last 3 transitions) rendered at bottom.

## Run Logger

Creates `runs/YYYYMMDD_HHMMSS_<arm_id>/` with:

| File | Content |
|------|---------|
| `run.jsonl` | Planner interactions (kind=`planner`), VLM inferences (kind=`vlm`), scene descriptions (kind=`scene`), tracker events (kind=`tracker`), compaction events (kind=`compaction`) |
| `events.jsonl` | All EventBus events (strips `vlm_image` key for serialization) |
| `vlm_NNN.jpg` | VLM input image with detection bboxes overlaid (red rectangles + handle labels) |
| `scene_NNN.txt` | SCENE_DESCRIBED text + detection list |
| `tui.log` | Redirected Python logging (all levels, including third-party) |
| `sim.log` | MuJoCo SimServer output (mujoco source only) |

## Integration

### Wires to (live mode)

- **HALORuntime** — `get_latest_runtime_snapshot()`, `submit_command()`, `bus.subscribe()`
- **Switchboard** (via `cognitive_stack`) — `decide()` for planner, `vlm_scene()` for perception; `active_type` for backend display; `start()`/`stop()` lifecycle
- **LeaseManager** — `_stamp_lease()` stamps `epoch` + `lease_token` on every command
- **TargetPerceptionService** — `start()`/`stop()`, `request_refresh()` on startup
- **SkillRunnerService** — `start()`/`stop()`, `start_skill()`/`abort_skill()` via command routing, `get_view_model()` for feed viewer
- **LiveAgentClient** — `connect()`, `send_tool_result()`, `send_audio()`, `send_monitor_update()`; audio callbacks wired to playback
- **VideoSource / SimSource** — `start()`/`stop()`, `make_capture_fn()`, joint state (`latest_qpos`/`latest_qvel`)

### Command routing

In live mode with SkillRunnerService, the TUI intercepts `submit_command` to capture commands, then a `_route_commands` worker listens for `COMMAND_ACCEPTED` events and dispatches `START_SKILL`/`ABORT_SKILL` to the SkillRunnerService.

## Implementation Patterns

### Mock detection
`self._runtime is None` → mock mode (static `_DATA` dict). All live workers and service wiring are skipped.

### Safety fuse
`_fuse_blown` flag blocks all commands except `ABORT_SKILL` and `DESCRIBE_SCENE`. Ctrl+A blows, Ctrl+R resets. The fuse also clears `_agent_queue` and `_pending_commands` on abort.

### Agent processor loop
Single `_agent_processor_loop` worker reads from `_agent_queue`. Brief 50ms yield after first message to batch burst events. Messages are concatenated with `\n`. Event-wake messages (start with `[`) get task context appended via `_with_task_context()`.

### Lease stamping
`_stamp_lease(cmd)` reads `current_epoch` + `current_token` from `LeaseManager` and stamps them on every `CommandEnvelope` before submission.

### Event listening
`_listen_events` worker drains the EventBus subscription. For each event:
1. Logs to `events.jsonl`
2. Appends to `EventsPanel` (skips `PHASE_ENTER`/`PHASE_EXIT` — too frequent)
3. Forwards narration events to Live Agent
4. Wakes agent on `_AGENT_WAKE_EVENTS` (skips operator-initiated aborts)

### Cloud startup
`_startup_cloud_and_perception` runs in background: polls cloud health for `startup_cloud_wait_s`, falls back to LOCAL on timeout, then triggers initial `DESCRIBE_SCENE`.

### Logging redirection
In live mode, all Python logging goes to `tui.log` (file handler replaces root handlers). stdout/stderr are also redirected to prevent third-party print() from corrupting the Textual display.

## Testing

- `make tui-mock` — launch with static fixture data, no services needed
- `--screenshot <path>` — headless mode, renders one frame, saves SVG, exits (used for docs)
- Panels have `refresh_live(data)` methods for in-place DOM updates (no churn)
- No dedicated unit tests for TUI widgets currently; tested via mock mode visual inspection
