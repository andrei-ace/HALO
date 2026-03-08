"""WebSocket handler for Live Agent bidirectional audio+text streaming.

Message framing (JSON):
    TUI → Cloud:
        {"type": "audio_in", "data": "<base64 PCM>"}
        {"type": "text_in", "text": "..."}
        {"type": "event", "data": {...}}   — robot events for narration
        {"type": "tool_result", "call_id": "...", "result": "..."}

    Cloud → TUI:
        {"type": "audio_out", "data": "<base64 PCM>"}
        {"type": "text_out", "text": "..."}
        {"type": "transcription_in", "text": "..."}
        {"type": "transcription_out", "text": "..."}
        {"type": "status", "text": "..."}
        {"type": "interrupt"}
        {"type": "tool_call", "call_id": "...", "name": "...", "args": {...}}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cloud_service.live_agent_manager import LiveAgentManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Notable robot events worth narrating
_NARRATION_EVENTS = frozenset(
    {
        "SKILL_STARTED",
        "SKILL_SUCCEEDED",
        "SKILL_FAILED",
        "SAFETY_REFLEX_TRIGGERED",
        "PERCEPTION_FAILURE",
    }
)


@router.websocket("/ws/live/{arm_id}")
async def live_agent_ws(websocket: WebSocket, arm_id: str) -> None:
    """Bidirectional WebSocket endpoint for the Live Agent."""
    # Resolve LiveAgentManager from app state
    live_mgr: LiveAgentManager | None = getattr(websocket.app.state, "live_agent_manager", None)
    if live_mgr is None:
        await websocket.close(code=1011, reason="Live agent not enabled")
        return

    await websocket.accept()
    logger.info("Live agent WS connected for arm_id=%s", arm_id)

    session = await live_mgr.get_or_create(arm_id)

    # Set up outbound callbacks that send JSON messages over the WebSocket
    send_lock = asyncio.Lock()

    async def _ws_send(msg: dict) -> None:
        try:
            async with send_lock:
                await websocket.send_json(msg)
        except Exception:
            pass  # Connection may have closed

    loop = asyncio.get_running_loop()

    def on_audio_out(pcm_bytes: bytes) -> None:
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        asyncio.run_coroutine_threadsafe(_ws_send({"type": "audio_out", "data": encoded}), loop)

    def on_text_out(text: str) -> None:
        asyncio.run_coroutine_threadsafe(_ws_send({"type": "text_out", "text": text}), loop)

    def on_transcription_in(text: str) -> None:
        asyncio.run_coroutine_threadsafe(_ws_send({"type": "transcription_in", "text": text}), loop)

    def on_transcription_out(text: str) -> None:
        asyncio.run_coroutine_threadsafe(_ws_send({"type": "transcription_out", "text": text}), loop)

    def on_interrupted() -> None:
        asyncio.run_coroutine_threadsafe(_ws_send({"type": "interrupt"}), loop)

    def on_tool_call(call_id: str, name: str, args: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _ws_send({"type": "tool_call", "call_id": call_id, "name": name, "args": args}), loop
        )

    session.set_callbacks(
        on_audio_out=on_audio_out,
        on_text_out=on_text_out,
        on_transcription_in=on_transcription_in,
        on_transcription_out=on_transcription_out,
        on_interrupted=on_interrupted,
        on_tool_call=on_tool_call,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from WS client: %.100s", raw)
                continue

            msg_type = msg.get("type")

            if msg_type == "audio_in":
                # Decode base64 PCM and forward to Gemini
                data = msg.get("data", "")
                try:
                    pcm = base64.b64decode(data)
                    session.on_audio_chunk(pcm)
                except Exception:
                    logger.warning("Failed to decode audio_in data")

            elif msg_type == "text_in":
                text = msg.get("text", "").strip()
                if text:
                    session.send_text(text)

            elif msg_type == "event":
                # Robot event for narration
                event_data = msg.get("data", {})
                event_type = event_data.get("type", "")
                if event_type in _NARRATION_EVENTS:
                    # Build a brief status line for the agent
                    summary = event_data.get("summary", event_type)
                    session.inject_status(summary)

            elif msg_type == "tool_result":
                # TUI responding to a proxy tool call
                call_id = msg.get("call_id", "")
                result = msg.get("result", "")
                if call_id:
                    session.resolve_tool_call(call_id, result)

            else:
                logger.debug("Unknown WS message type: %s", msg_type)

    except WebSocketDisconnect:
        logger.info("Live agent WS disconnected for arm_id=%s", arm_id)
    except Exception:
        logger.exception("Live agent WS error for arm_id=%s", arm_id)
    finally:
        session.set_callbacks()
        # Close the live request queue to unblock any pending run_live iteration
        session.close_queue()
        await live_mgr.remove(arm_id)
