"""FastAPI app for the HALO cloud cognitive service.

Endpoints:
    POST /decide         — planner decision (snapshot JSON → commands JSON)
    POST /vlm/scene      — VLM scene analysis (JPEG image → VlmScene JSON)
    POST /warm-up        — warm-up session with state + journal
    GET  /state/{arm_id} — session readiness and cursor
    POST /reset/{arm_id} — reset specific arm session
    POST /reset          — reset default (arm0) session (backward compat)
    GET  /health         — health check
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from halo.contracts.serde import command_envelope_to_dict, snapshot_from_dict, vlm_scene_to_dict

from cloud_service.deps import get_session_manager, get_vlm_fn, lifespan, verify_api_key

logger = logging.getLogger(__name__)

app = FastAPI(title="HALO Cloud Cognitive Service", lifespan=lifespan)


@app.get("/health")
async def health(session_mgr=Depends(get_session_manager)) -> dict:
    return {
        "status": "ok",
        "nonce": session_mgr.nonce,
        "sessions": session_mgr.active_arm_ids,
    }


@app.post("/decide", dependencies=[Depends(verify_api_key)])
async def decide(body: dict, session_mgr=Depends(get_session_manager)) -> dict:
    snapshot = snapshot_from_dict(body["snapshot"])
    operator_cmd = body.get("operator_cmd")
    epoch = body.get("epoch")
    arm_id = snapshot.arm_id

    session = session_mgr.get_or_create(arm_id)
    if session.pending_handoff:
        await session.agent.inject_handoff_context(session.pending_handoff)
        session.pending_handoff = None
    commands = await session.agent.decide(snapshot, operator_cmd=operator_cmd, epoch=epoch)
    reasoning = session.agent.last_reasoning

    return {
        "commands": [command_envelope_to_dict(c) for c in commands],
        "reasoning": reasoning,
    }


@app.post("/vlm/scene", dependencies=[Depends(verify_api_key)])
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
    # Decode JPEG to numpy BGR for VLM pipeline
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    import cv2

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = image_bytes  # fallback: pass raw bytes

    scene = await vlm_fn(arm_id, img, known_handles, target_handle=target_handle)
    return vlm_scene_to_dict(scene)


@app.post("/warm-up", dependencies=[Depends(verify_api_key)])
async def warm_up(body: dict, session_mgr=Depends(get_session_manager)) -> dict:
    """Warm up a session with CognitiveState + journal entries."""
    state_dict = body.get("state")
    journal_dicts = body.get("journal", [])
    # Read arm_id from top-level body, fall back to state, then default
    arm_id = body.get("arm_id") or (state_dict.get("last_arm_id") if state_dict else None) or "arm0"

    session = session_mgr.warm_up_session(arm_id, state_dict, journal_dicts)
    return {
        "readiness": session.readiness,
        "cursor": session.cursor,
    }


@app.get("/state/{arm_id}", dependencies=[Depends(verify_api_key)])
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


@app.post("/reset/{arm_id}", dependencies=[Depends(verify_api_key)])
async def reset_arm(arm_id: str, session_mgr=Depends(get_session_manager)) -> dict:
    """Reset a specific arm session."""
    session_mgr.reset_session(arm_id)
    return {"status": "ok"}


@app.post("/reset", dependencies=[Depends(verify_api_key)])
async def reset(session_mgr=Depends(get_session_manager)) -> dict:
    """Reset the default arm0 session (backward compat)."""
    session_mgr.reset_session("arm0")
    return {"status": "ok"}
