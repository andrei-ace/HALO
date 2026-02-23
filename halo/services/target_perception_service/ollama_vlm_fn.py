from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.vlm_parser import parse_vlm_response

# Resize input images to this width before sending to Ollama.
# Height is computed to preserve aspect ratio.  All model bbox
# coordinates will be in this known pixel space.
_VLM_INPUT_WIDTH = 1024

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_IMAGE = Path(__file__).parents[3] / "docs" / "data" / "mock" / "mock.png"
_DEFAULT_PROMPT = Path(__file__).parents[3] / "configs" / "perception" / "scene_analysis.md"


def _extract_json(text: str) -> dict:
    """
    Parse JSON from a model response string.

    Handles three common formats:
    - Bare JSON object
    - JSON wrapped in ```json ... ``` fences
    - JSON embedded somewhere inside prose
    """
    text = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.rstrip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: find the first {...} block in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _call_ollama_sync(base_url: str, model: str, prompt: str, image_b64: str) -> dict:
    """Blocking Ollama /api/generate call — run via asyncio.to_thread."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def make_ollama_vlm_fn(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5vl",
    image_path: Path = _DEFAULT_IMAGE,
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    """
    Return a vlm_fn that sends *image_path* to *model* via Ollama.

    The image and prompt are loaded once at construction time.  Each call is
    non-blocking (asyncio.to_thread) so it never stalls the fast loop.

    Returns a TargetInfo seeded from the detection whose handle matches
    target_handle, or the most-confident detection if no exact match.
    Returns None if the model finds nothing.

    If *run_logger* is provided every inference (success or failure) is
    appended to the session JSONL log alongside the full raw Ollama response.
    """
    # Resize to a known width so model bbox coords map to a known pixel space.
    img = Image.open(image_path)
    aspect = img.height / img.width
    new_w = _VLM_INPUT_WIDTH
    new_h = int(new_w * aspect)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = prompt_path.read_text(encoding="utf-8")

    async def vlm_fn(arm_id: str, target_handle: str) -> TargetInfo | None:
        t0 = time.monotonic()
        result: dict = {}
        target_info: TargetInfo | None = None
        error: str | None = None

        try:
            result = await asyncio.to_thread(_call_ollama_sync, base_url, model, prompt, image_b64)
        except Exception as exc:
            error = str(exc)
            if run_logger is not None:
                run_logger.log_vlm_inference(
                    arm_id=arm_id,
                    target_handle=target_handle,
                    model=model,
                    raw_response={k: v for k, v in result.items() if k != "context"},
                    target_info=None,
                    inference_ms=int((time.monotonic() - t0) * 1000),
                    error=error,
                )
            return None

        raw = _extract_json(result.get("response", ""))
        vlm_scene = parse_vlm_response(raw)

        if vlm_scene.detections:
            # Prefer exact handle match; fall back to first graspable detection.
            match = next((d for d in vlm_scene.detections if d.handle == target_handle), None)
            if match is None:
                match = next(
                    (d for d in vlm_scene.detections if d.is_graspable),
                    vlm_scene.detections[0],
                )

            total_ns = result.get("total_duration", 0)
            obs_age_ms = int(total_ns / 1_000_000) if total_ns else 0

            target_info = TargetInfo(
                handle=target_handle,
                hint_valid=True,
                confidence=1.0,  # VLM detected it — present is 1.0
                obs_age_ms=obs_age_ms,
                time_skew_ms=0,
                # Depth not available from VLM; filled by depth fusion later.
                delta_xyz_ee=(0.0, 0.0, 0.0),
                distance_m=0.0,
            )

        if run_logger is not None:
            # Drop the context token array — it's large and not useful in logs.
            loggable = {k: v for k, v in result.items() if k != "context"}
            run_logger.log_vlm_inference(
                arm_id=arm_id,
                target_handle=target_handle,
                model=model,
                raw_response=loggable,
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
