"""FastAPI app for the HALO cloud cognitive service.

Endpoints:
    POST /decide      — planner decision (snapshot JSON → commands JSON)
    POST /vlm/scene   — VLM scene analysis (JPEG image → VlmScene JSON)
    GET  /health      — health check
    POST /reset       — reset planner session state
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from halo.contracts.serde import (
    command_envelope_to_dict,
    snapshot_from_dict,
    vlm_scene_to_dict,
)

from cloud_service.deps import get_agent, get_vlm_fn, lifespan, verify_api_key

logger = logging.getLogger(__name__)

app = FastAPI(title="HALO Cloud Cognitive Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/decide", dependencies=[Depends(verify_api_key)])
async def decide(body: dict, agent=Depends(get_agent)) -> dict:
    snapshot = snapshot_from_dict(body["snapshot"])
    operator_cmd = body.get("operator_cmd")

    commands = await agent.decide(snapshot, operator_cmd=operator_cmd)
    reasoning = agent.last_reasoning

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


@app.post("/reset", dependencies=[Depends(verify_api_key)])
async def reset(agent=Depends(get_agent)) -> dict:
    agent.reset_loop_state()
    return {"status": "ok"}
