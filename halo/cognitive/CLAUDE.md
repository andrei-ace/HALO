# Cognitive

Backend abstraction and switching for the HALO planner. Supports local (Ollama) and cloud (Gemini/Cloud Run) backends with automatic failover, context preservation, and session management.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | `CognitiveStack` dataclass + `make_cognitive_stack()` factory |
| `config.py` | `BackendType`, `BackendReadiness`, `CognitiveConfig`, `LocalConfig`, `CloudConfig`, `RemoteCloudConfig` (`vlm_timeout_s`), `CompactionConfig`, `LiveAgentConfig` |
| `backend.py` | `CognitiveBackend` protocol: `decide()`, `vlm_scene()`, `health_check()`, `reset_loop_state()` |
| `local_backend.py` | `LocalCognitiveBackend` — wraps `PlannerAgent` + Ollama VLM |
| `remote_backend.py` | `RemoteCognitiveBackend` — HTTP client to Cloud Run; IAM auth, `last_msg_id` sync, multipart VLM |
| `switchboard.py` | `Switchboard` — proxy routing calls to active backend, retry (3 attempts), failover (3 consecutive failures), failback on recovery |
| `lease.py` | `Lease`, `LeaseManager` — epoch-based split-brain prevention (grant/renew/revoke/validate) |
| `context_store.py` | `ContextEntry`, `ContextSnapshot`, `CognitiveState`, `ContextStore` — append-only journal for context handoff between backends |
| `compactor.py` | `MessageRecord`, `CompactionResult`, `MessageHistory` — UUID-tagged message tracking with compaction support |
| `compaction_plugin.py` | `CompactionPlugin` (ADK BasePlugin) — enforces exactly-one-snapshot invariant, triggers session compaction |
| `live_session.py` | `LivePlannerSession` — persistent Gemini Live API session bridging streaming audio + request-response `decide()` |
| `live_agent_client.py` | `LiveAgentClient` — TUI-side WebSocket client for cloud live agent (audio, text, tool calls, commands) |
| `audio_io.py` | `AudioCapture` (16kHz), `AudioPlayback` (24kHz), `AudioComponents`, `make_audio_components()` |

## Switchboard Proxy Pattern

`Switchboard` is the main entry point for planner/VLM calls:

```python
switchboard.decide(snapshot, operator_cmd=None) → list[CommandEnvelope]
switchboard.vlm_scene(arm_id, image, known_handles) → VlmScene
switchboard.switch_to(backend_type) → None
```

- **Retry:** up to 3 attempts with exponential delays (0.5s, 1s, 2s); skips retries for non-retryable errors (429, RESOURCE_EXHAUSTED, quota)
- **Failover:** 3 consecutive failures → automatic switch to other backend
- **Failback:** background health loop checks inactive backend; switches back to preferred if it recovers
- Stamps `epoch` + `lease_token` on all returned commands
- Records decisions and scenes in `ContextStore`
- Mirrors message history to inactive backend for seamless failback

## Lease Lifecycle

`LeaseManager` prevents split-brain by gating all commands on epoch + token:

1. `grant(holder)` — increment epoch, create new lease with UUID token
2. `renew(epoch)` — refresh TTL on current lease
3. `revoke(epoch)` — clear lease
4. `is_valid(epoch)` / `is_valid_token(epoch, token)` — validate

`CommandRouter` rejects commands with stale/missing epoch+token when LeaseManager is active.

## Context Handoff

On backend switch, `ContextStore` provides handoff context:

1. `ContextStore.take_snapshot()` → `ContextSnapshot` (current state summary)
2. `get_handoff_context()` → text summary for new backend's first message
3. `apply_entries()` — replay journal entries from remote (validates cursor monotonicity)

Entry types: `decision`, `scene`, `event`, `operator`. Max 200 entries with auto-trim.

## Message History & Compaction

`MessageHistory` tracks UUID-tagged messages for precise compaction:
- `apply_compaction(summary, up_to_msg_id)` — replace old messages with summary, preserve remainder
- Cross-backend mirroring via `replace_all()` ensures inactive backend stays in sync

`CompactionPlugin` (ADK before_model_callback):
- Strips JSON snapshot blocks from all but the most recent user message
- Triggers session compaction at configured intervals
- Defers history compaction until after model response is tracked

## Audio Data Flow

```
Microphone → AudioCapture (16kHz) → on_audio callback → LivePlannerSession/LiveAgentClient → cloud
Cloud → audio_out → AudioPlayback (24kHz) → Speaker
Cloud → interrupt → AudioPlayback.clear()
```

`make_audio_components()` gracefully handles missing `sounddevice` package.

## LiveAgentClient (TUI-side)

WebSocket client connecting to `/ws/live/{arm_id}`:
- `send_audio()` — base64-encoded PCM from sounddevice thread
- `send_text()` — operator text input
- `send_monitor_update()` — categories: event, planner_decision, scene_description
- `send_tool_result()` / `on_tool_call` callback — proxy-tool protocol
- `drain_commands()` — retrieve accumulated commands
- Auto-reconnect with exponential backoff (max 30s)

## Integration

- **Switchboard** used by `PlannerService` (via `decide_fn`) and `TargetPerceptionService` (via `vlm_fn`)
- **LeaseManager** injected into `HALORuntime` → `CommandRouter` for epoch/token validation
- **ContextStore** feeds handoff context on backend switch
- **LiveAgentClient** used by TUI for voice/text interaction with cloud live agent
- **AudioCapture/Playback** managed by TUI via `AudioComponents`

## Testing

Backend protocol is mockable. `Switchboard` tests inject mock backends and verify failover/retry behaviour. `LeaseManager` is fully deterministic. `ContextStore` tests verify journal append, snapshot, and handoff text generation.
