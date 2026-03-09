# Bridge

2-channel ZeroMQ bridge connecting HALO runtime to MuJoCo sim server. Decouples real-time telemetry streaming from command RPC.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `SimBridgeConfig` — URLs, timeouts, managed mode flag, protocol version (v2 only) |
| `sim_client.py` | `SimClient` — HALO-side ZMQ client: background telemetry receiver thread + command RPC with retry |
| `sim_source.py` | `SimSource` — wraps `SimClient` as a camera video source; frame queue + `make_capture_fn()` for `TargetPerceptionService` |
| `transforms.py` | `world_to_ee_frame(world_delta, ee_quat)` — quaternion-based world → EE rotation |
| `sim_tracker_service.py` | `SimTrackerService` — ZMQ REP service for VLM/tracker queries from sim server during episode generation |

## 2-Channel Architecture

| Channel | Socket | Direction | Purpose |
|---------|--------|-----------|---------|
| TelemetryStream | SUB | Sim → HALO | High-frequency state: RGB frames, qpos, qvel, phase_id, done flag |
| CommandRPC | REQ/REP | HALO → Sim | Sequential commands: step, reset, start_pick, start_place, set_hint, etc. |

Optional **Ch4** (REP): Sim server → HALO. VLM/tracker queries during episode generation (used by `SimTrackerService`).

## SimClient

**Lifecycle:** `start(timeout)` → connect (or spawn if managed) + start telemetry thread → `stop()`

**CommandRPC methods:** `step()`, `reset()`, `get_state()`/`set_state()`, `start_pick()`/`start_place()`, `abort_pick()`, `configure()`, `shutdown()`, `set_hint()`

**Threading:** Background daemon thread polls TelemetryStream SUB. `_cmd_lock` serializes REQ/REP (ZMQ constraint). Retryable commands (reset, get_state, set_state, configure, set_hint) auto-reconnect on ZMQ error.

**Managed mode:** `managed=True` spawns sim server subprocess; `start()` waits for it to bind, `stop()` kills it.

## SimSource

Drop-in wrapper providing camera-compatible interface. Polling thread at ~100 Hz converts telemetry RGB → BGR, stores in frame deque (max 30).

`make_capture_fn(arm_id)` → async `capture_fn(arm_id) → CapturedFrame` compatible with `TargetPerceptionService`.

Properties: `latest_frame`, `latest_wrist_frame`, `latest_qpos`/`latest_qvel`, `latest_phase_id`/`latest_done`/`latest_error`.

## SimTrackerService

Runs asyncio event loop in background thread. Binds REP socket, dispatches VLM/tracker queries from sim server.

Handles: `QUERY_VLM_DETECT` → run VLM + init tracker, `QUERY_TRACKER_INIT` → init tracker, `QUERY_TRACKER_UPDATE` → feed frame to active tracker.

## BridgeTransportError

Raised on ZMQ communication failures. `ControlService` catches this to write `ActStatus.STALE`.

## Integration

- **SimClient** used by `SkillRunnerService` (sim mode) and `ControlService` (apply_fn)
- **SimSource** used by `TargetPerceptionService` (capture_fn)
- **SimTrackerService** used during MuJoCo episode generation
- **BridgeTransportError** caught by `ControlService` for graceful degradation
