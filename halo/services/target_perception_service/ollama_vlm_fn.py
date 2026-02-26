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

import numpy as np
from PIL import Image

from halo.services.target_perception_service.vlm_parser import VlmScene, parse_vlm_response

# Resize input images to this width before sending to Ollama.
# Height is computed to preserve aspect ratio.  All model bbox
# coordinates will be in this known pixel space.
_VLM_INPUT_WIDTH = 1024

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

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


def _image_to_b64(image: object) -> str:
    """Convert an image (numpy BGR HWC, PIL Image, or bytes) to base64 PNG."""
    if isinstance(image, np.ndarray):
        # OpenCV BGR → RGB → PIL
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
    elif isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Resize to known width for consistent bbox coordinate space.
    aspect = pil.height / pil.width
    new_w = _VLM_INPUT_WIDTH
    new_h = int(new_w * aspect)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    """
    Return an async VlmFn that sends a camera image to *model* via Ollama
    and returns a VlmScene (scene description + list of detections).

    The image is provided per-call by the service (captured from the camera).
    The prompt is loaded once at construction time.  Each call is non-blocking
    (asyncio.to_thread) so it never stalls the event loop.

    If *run_logger* is provided every inference (success or failure) is
    appended to the session JSONL log alongside the full raw Ollama response.
    """
    prompt = prompt_path.read_text(encoding="utf-8")

    async def vlm_fn(
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        if image is None:
            return VlmScene(scene="", detections=[])

        image_b64 = _image_to_b64(image)

        effective_prompt = prompt
        if known_handles:
            handles_str = ", ".join(known_handles)
            effective_prompt += f"\nPreviously known: {handles_str}. Reuse these handles for the same objects."

        t0 = time.monotonic()
        result: dict = {}
        error: str | None = None

        try:
            result = await asyncio.to_thread(_call_ollama_sync, base_url, model, effective_prompt, image_b64)
        except Exception as exc:
            error = str(exc)
            if run_logger is not None:
                await asyncio.to_thread(
                    run_logger.log_vlm_inference,
                    arm_id=arm_id,
                    target_handle=target_handle or "",
                    model=model,
                    raw_response={k: v for k, v in result.items() if k != "context"},
                    target_info=None,
                    inference_ms=int((time.monotonic() - t0) * 1000),
                    error=error,
                    image=image,
                )
            return VlmScene(scene="", detections=[])

        raw = _extract_json(result.get("response", ""))
        vlm_scene = parse_vlm_response(raw)

        if run_logger is not None:
            loggable = {k: v for k, v in result.items() if k != "context"}
            det_dicts = [{"handle": d.handle, "bbox": d.bbox} for d in vlm_scene.detections]
            await asyncio.to_thread(
                run_logger.log_vlm_inference,
                arm_id=arm_id,
                target_handle=target_handle or "",
                model=model,
                raw_response=loggable,
                target_info=None,
                inference_ms=int((time.monotonic() - t0) * 1000),
                error=error,
                image=image,
                detections=det_dicts,
            )

        return vlm_scene

    return vlm_fn
