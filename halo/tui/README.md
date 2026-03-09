# HALO TUI

Operator terminal for the HALO robotic manipulation system. Built with [Textual](https://textual.textualize.io/) — runs in any terminal.

## Quick Start

```bash
make tui-mock              # static fixture data, no services needed
make tui-live              # Ollama planner + VLM + MuJoCo sim
make tui-live-cloud        # via cloud service (set HALO_CLOUD_URL)
make tui-live-cloud-local  # cloud service on localhost
```

## Modes

| Mode | What it does |
|------|-------------|
| **Mock** | Renders the full UI with static sample data. No Ollama, no sim, no network. Good for layout/styling work. |
| **Live local** | Connects to local Ollama for planner + VLM. Optional MuJoCo sim (`--source mujoco`). |
| **Live cloud** | Routes planner/VLM through the cloud service via Switchboard. Automatic failover to local on 3 consecutive failures. |

Add `--live-agent` (with `--cloud-url`) for voice interaction via the Live Agent WebSocket.

## Layout

Two-column dashboard:

- **Left**: Skill Runner status, Target Perception, Talk to Planner (text input + history), Voice panel (if live-agent)
- **Right**: System health (service status dots + backend), Joints (6-DOF positions + velocity bars), Control Service, Events log, Panic (abort/reset)

## Keybindings

| Key | Action |
|-----|--------|
| `T` | Focus message input |
| `Enter` | Send message to planner |
| `Esc` | Clear input |
| `R` | View full planner reasoning |
| `Y` | Copy reasoning to clipboard |
| `F` | Toggle OpenCV feed viewer (live only, requires `uv sync --extra viewer`) |
| `M` | Toggle mic mute (live agent) |
| `Ctrl+A` | Emergency abort (blow safety fuse) |
| `Ctrl+R` | Reset fuse |
| `Ctrl+Q` | Quit |
| `?` | Keyboard legend |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HALO_CLOUD_URL` | Cloud service URL (used by make targets) |
| `HALO_OLLAMA_URL` | Ollama base URL (default `http://localhost:11434`) |
| `HALO_MODEL_NAME` | Planner model (default `gpt-oss:20b`) |

## Session Logs

Each live session writes to `runs/YYYYMMDD_HHMMSS_<arm_id>/`:

- `run.jsonl` — planner interactions, VLM inferences, scene descriptions
- `events.jsonl` — all EventBus events
- `vlm_NNN.jpg` — annotated VLM input images
- `tui.log` — redirected Python logging

## Screenshot

Generate an SVG screenshot (headless):

```bash
uv run python -m halo.tui.app --screenshot runs/halo_tui.svg
```
