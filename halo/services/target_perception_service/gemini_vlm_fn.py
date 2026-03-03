"""Gemini VLM function — drop-in replacement for ``make_ollama_vlm_fn``.

Returns an async ``VlmFn`` with the same signature: ``(arm_id, image,
known_handles?, target_handle?) -> VlmScene``.  Calls the Gemini API
via ``google.genai`` (already installed as a transitive dep of
``google-adk``).

Usage::

    from halo.services.target_perception_service.gemini_vlm_fn import make_gemini_vlm_fn
    vlm_fn = make_gemini_vlm_fn(model="gemini-2.5-flash")
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from halo.services.target_perception_service.vlm_parser import VlmScene, parse_vlm_response

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_PROMPT = Path(__file__).parents[3] / "configs" / "perception" / "scene_analysis.md"

# Resize input images to this width before sending to Gemini.
_VLM_INPUT_WIDTH = 1024

# JSON schema that Gemini must conform to (response_mime_type="application/json").
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string"},
                    "label": {"type": "string"},
                    "bounding_box": {"type": "array", "items": {"type": "number"}},
                    "is_graspable": {"type": "boolean"},
                },
                "required": ["handle", "label", "bounding_box", "is_graspable"],
            },
        },
    },
    "required": ["scene", "detections"],
}


def _to_pil(image: object) -> Image.Image:
    """Convert numpy BGR HWC, PIL Image, or bytes to a PIL Image."""
    if isinstance(image, np.ndarray):
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _resize_image(pil: Image.Image) -> tuple[Image.Image, int, int]:
    """Resize if wider than _VLM_INPUT_WIDTH. Returns (image, sent_w, sent_h)."""
    if pil.width > _VLM_INPUT_WIDTH:
        aspect = pil.height / pil.width
        new_w = _VLM_INPUT_WIDTH
        new_h = int(new_w * aspect)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return pil, pil.width, pil.height


def _extract_json(text: str) -> dict:
    """Parse JSON from a model response, handling fences and embedded JSON."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.rstrip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _call_gemini_sync(
    api_key: str,
    model: str,
    prompt: str,
    pil_image: Image.Image,
) -> dict:
    """Blocking Gemini generate_content call — run via asyncio.to_thread."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[pil_image, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _RESPONSE_SCHEMA,
        },
    )
    return _extract_json(response.text or "")


def make_gemini_vlm_fn(
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    """Return an async VlmFn backed by Gemini.

    Same contract as ``make_ollama_vlm_fn``: accepts ``(arm_id, image,
    known_handles, target_handle)`` and returns a ``VlmScene``.

    If *api_key* is None, reads ``GOOGLE_API_KEY`` from the environment.
    """
    resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    prompt = prompt_path.read_text(encoding="utf-8")

    async def vlm_fn(
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        if image is None:
            return VlmScene(scene="", detections=[])

        pil_image = _to_pil(image)
        pil_image, sent_w, sent_h = _resize_image(pil_image)

        effective_prompt = prompt
        if known_handles:
            handles_str = ", ".join(known_handles)
            effective_prompt += f"\nPreviously known: {handles_str}. Reuse these handles for the same objects."

        t0 = time.monotonic()
        raw: dict = {}
        error: str | None = None

        try:
            raw = await asyncio.to_thread(
                _call_gemini_sync,
                resolved_key,
                model,
                effective_prompt,
                pil_image,
            )
        except Exception as exc:
            error = str(exc)
            if run_logger is not None:
                await asyncio.to_thread(
                    run_logger.log_vlm_inference,
                    arm_id=arm_id,
                    target_handle=target_handle or "",
                    model=model,
                    raw_response={},
                    target_info=None,
                    inference_ms=int((time.monotonic() - t0) * 1000),
                    error=error,
                    image=image,
                )
            return VlmScene(scene="", detections=[])

        vlm_scene = parse_vlm_response(raw, img_w=sent_w, img_h=sent_h)

        if run_logger is not None:
            det_dicts = [{"handle": d.handle, "bbox": d.bbox} for d in vlm_scene.detections]
            await asyncio.to_thread(
                run_logger.log_vlm_inference,
                arm_id=arm_id,
                target_handle=target_handle or "",
                model=model,
                raw_response=raw,
                target_info=None,
                inference_ms=int((time.monotonic() - t0) * 1000),
                error=error,
                image=image,
                detections=det_dicts,
            )

        return vlm_scene

    return vlm_fn
