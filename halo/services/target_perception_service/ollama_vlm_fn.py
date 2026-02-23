from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.vlm_parser import parse_vlm_response

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_IMAGE = Path(__file__).parents[3] / "docs" / "data" / "mock" / "mock.png"

# Prompt instructs qwen3-vl to return coordinates in 0-1000 space so that
# vlm_parser can normalise them — consistent with the mock JSON contract.
_PROMPT = """\
You are a robotic manipulation perception system.
Analyse the scene and locate every graspable object on the workspace surface.

Return ONLY a JSON object in this exact format (no markdown, no extra keys):
{{
  "detections": [
    {{
      "handle": "<slug>",
      "label": "<human name>",
      "bounding_box": [x1, y1, x2, y2],
      "centroid_xy": [cx, cy],
      "estimated_depth_m": 0.0,
      "confidence": 0.0,
      "is_graspable": true
    }}
  ]
}}

Coordinate rules:
- All coordinates are integers in 0-1000 space (top-left = 0,0; bottom-right = 1000,1000).
- estimated_depth_m: approximate distance from the camera in metres.
- confidence: 0.0 – 1.0.
- handle: a short snake_case slug you invent (e.g. "red_cube_01").

If no objects are visible return {{"detections": []}}.
"""


def _call_ollama_sync(
    base_url: str,
    model: str,
    image_b64: str,
) -> dict:
    """Blocking Ollama /api/generate call — run via asyncio.to_thread."""
    payload = json.dumps({
        "model": model,
        "prompt": _PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": "json",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def make_ollama_vlm_fn(
    base_url: str = "http://localhost:11434",
    model: str = "qwen3-vl:30B",
    image_path: Path = _DEFAULT_IMAGE,
    run_logger: RunLogger | None = None,
):
    """
    Return a vlm_fn that sends *image_path* to *model* via Ollama.

    The image is base64-encoded once at construction time.  Each call is
    non-blocking (asyncio.to_thread) so it never stalls the fast loop.

    Returns a TargetInfo seeded from the first detection whose handle matches
    target_handle, or None if the model finds nothing.

    If *run_logger* is provided every inference (success or failure) is appended
    to the session JSONL log alongside the full raw Ollama response.
    """
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    async def vlm_fn(arm_id: str, target_handle: str) -> TargetInfo | None:
        t0 = time.monotonic()
        result: dict = {}
        target_info: TargetInfo | None = None
        error: str | None = None

        try:
            result = await asyncio.to_thread(
                _call_ollama_sync, base_url, model, image_b64
            )
        except Exception as exc:
            error = str(exc)
            if run_logger is not None:
                run_logger.log_vlm_inference(
                    arm_id=arm_id,
                    target_handle=target_handle,
                    model=model,
                    raw_response=result,
                    target_info=None,
                    inference_ms=int((time.monotonic() - t0) * 1000),
                    error=error,
                )
            return None

        # qwen3-vl is a thinking model: JSON output lands in "thinking" when
        # "response" is empty.
        raw_text = result.get("response") or result.get("thinking", "{}")
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            raw = {}

        detections = parse_vlm_response(raw)
        if detections:
            # Prefer an exact handle match; fall back to the most confident detection.
            match = next((d for d in detections if d.handle == target_handle), None)
            if match is None:
                match = max(detections, key=lambda d: d.confidence)

            total_ns = result.get("total_duration", 0)
            obs_age_ms = int(total_ns / 1_000_000) if total_ns else 0

            target_info = TargetInfo(
                handle=target_handle,
                hint_valid=match.confidence >= 0.5,
                confidence=match.confidence,
                obs_age_ms=obs_age_ms,
                time_skew_ms=0,
                # VLM gives depth only — lateral EE offset is unknown here.
                delta_xyz_ee=(0.0, 0.0, -match.estimated_depth_m),
                distance_m=match.estimated_depth_m,
            )

        if run_logger is not None:
            run_logger.log_vlm_inference(
                arm_id=arm_id,
                target_handle=target_handle,
                model=model,
                raw_response=result,
                target_info=(
                    {
                        "handle": target_info.handle,
                        "hint_valid": target_info.hint_valid,
                        "confidence": target_info.confidence,
                        "obs_age_ms": target_info.obs_age_ms,
                        "distance_m": target_info.distance_m,
                        "delta_xyz_ee": list(target_info.delta_xyz_ee),
                    }
                    if target_info is not None
                    else None
                ),
                inference_ms=int((time.monotonic() - t0) * 1000),
                error=error,
            )

        return target_info

    return vlm_fn
