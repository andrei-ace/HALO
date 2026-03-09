"""FastAPI app for the HALO cloud cognitive service.

Endpoints:
    POST /decide         — planner decision (snapshot JSON → commands JSON)
    POST /vlm/scene      — VLM scene analysis (JPEG image → VlmScene JSON)
    GET  /state/{arm_id} — session readiness and cursor (debug)
    GET  /health         — health check
    WS   /ws/live/{arm_id} — Live Agent bidirectional audio+text streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from halo.contracts.serde import command_envelope_to_dict, message_record_to_dict, snapshot_from_dict, vlm_scene_to_dict

from cloud_service.deps import get_session_manager, get_vlm_fn, lifespan
from cloud_service.ws_handler import router as ws_router

logger = logging.getLogger(__name__)

app = FastAPI(title="HALO Cloud Cognitive Service", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
async def health(session_mgr=Depends(get_session_manager)) -> dict:
    return {
        "status": "ok",
        "sessions": session_mgr.active_arm_ids,
    }


@app.post("/decide")
async def decide(
    body: dict,
    session_mgr=Depends(get_session_manager),
) -> dict:
    snapshot = snapshot_from_dict(body["snapshot"])
    operator_cmd = body.get("operator_cmd")
    epoch = body.get("epoch")
    client_session_id = body.get("session_id")
    last_msg_id = body.get("last_msg_id")
    msg_history = body.get("msg_history")
    handoff_context = body.get("handoff_context")
    arm_id = snapshot.arm_id

    # Sync session using last_msg_id protocol
    sync_result = await session_mgr.sync_session(
        arm_id,
        last_msg_id=last_msg_id,
        msg_history=msg_history,
        client_session_id=client_session_id,
    )

    if sync_result.status == "need_history":
        logger.info("decide arm_id=%s session_id=%s: need_history (sync failed)", arm_id, client_session_id)
        return {"status": "need_history", "commands": [], "reasoning": "", "compacted": False}

    logger.info("decide arm_id=%s session_id=%s: session synced", arm_id, client_session_id)
    session = sync_result.session

    # Client-provided handoff takes priority over any persisted handoff
    if handoff_context:
        session.pending_handoff = handoff_context

    handoff = session.pending_handoff
    if handoff:
        await session.agent.inject_handoff_context(handoff)
        session.pending_handoff = None
    try:
        commands = await session.agent.decide(snapshot, operator_cmd=operator_cmd, epoch=epoch)
    except Exception:
        # Restore handoff so same-instance retries still get the context.
        if handoff:
            session.pending_handoff = handoff
        raise
    # Mark session ready after first successful decide (was previously done by warm-up)
    session.readiness = "ready"
    session.cursor = len(session.agent.msg_history.get_all())

    try:
        await session_mgr.persist_session(arm_id)
    except Exception:
        # Evict tainted in-memory session so the next request rehydrates
        # from the last good Firestore snapshot.
        session_mgr.evict_session(arm_id)
        raise
    reasoning = session.agent.last_reasoning
    compaction = session.agent.last_compaction
    compacted = compaction is not None
    if compacted:
        logger.info("Session compacted for arm_id=%s: %d messages summarized", arm_id, compaction.compacted_count)

    # Always include msg_history in response
    all_records = session.agent.msg_history.get_all()
    resp_history = [message_record_to_dict(r) for r in all_records]

    result: dict = {
        "status": "ok",
        "commands": [command_envelope_to_dict(c) for c in commands],
        "reasoning": reasoning,
        "compacted": compacted,
        "token_usage": session.agent.last_token_usage,
        "msg_history": resp_history,
    }
    if compacted and compaction is not None:
        result["compaction"] = {
            "summary": compaction.summary,
            "up_to_msg_id": compaction.up_to_msg_id,
            "compacted_count": compaction.compacted_count,
            "retained_count": compaction.retained_count,
        }
    return result


@app.post("/vlm/scene")
async def vlm_scene(
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
    vlm_fn=Depends(get_vlm_fn),
) -> dict:
    meta = json.loads(metadata)
    arm_id = meta.get("arm_id", "arm0")
    known_handles = meta.get("known_handles")
    target_handle = meta.get("target_handle")

    image_bytes = await image.read()

    # Decode JPEG in thread pool to avoid blocking the event loop
    def _decode_jpeg(data: bytes):
        import cv2

        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    img = await asyncio.to_thread(_decode_jpeg, image_bytes)
    if img is None:
        img = image_bytes  # fallback: pass raw bytes

    t0 = time.monotonic()
    scene = await vlm_fn(arm_id, img, known_handles, target_handle=target_handle)
    vlm_ms = int((time.monotonic() - t0) * 1000)

    result = vlm_scene_to_dict(scene)
    result["token_usage"] = scene.token_usage
    result["vlm_ms"] = vlm_ms
    logger.info("vlm/scene arm_id=%s target=%s vlm_ms=%d", arm_id, target_handle or "(scene)", vlm_ms)
    return result


@app.get("/state/{arm_id}")
async def get_state(arm_id: str, session_mgr=Depends(get_session_manager)) -> dict:
    """Return session readiness and cursor for a specific arm."""
    session = session_mgr.get_session(arm_id)
    if session is None:
        return {"readiness": "cold", "cursor": -1, "exists": False}
    return {
        "readiness": session.readiness,
        "cursor": session.cursor,
        "exists": True,
    }
