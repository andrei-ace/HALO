"""Unified VLM function — supports Gemini and Ollama backends.

Returns an async ``VlmFn`` with signature: ``(arm_id, image,
known_handles?, target_handle?) -> VlmScene``.

Usage::

    from halo.services.target_perception_service.vlm_fn import make_vlm_fn
    vlm_fn = make_vlm_fn(provider="gemini", model="gemini-3.1-flash-lite-preview")
    vlm_fn = make_vlm_fn(provider="ollama", model="qwen2.5vl:3b")
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from halo.services.target_perception_service.vlm_parser import VlmScene, parse_vlm_response

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_PROMPT = Path(__file__).parents[3] / "configs" / "perception" / "scene_analysis.md"

_VLM_INPUT_WIDTH_GEMINI = 768
_VLM_INPUT_WIDTH_OLLAMA = 1024

# Shared JSON schema — Ollama uses bounding_box (pixel ints), Gemini uses box_2d (0-1000 ints).
_OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "string",
            "description": "One sentence describing the workspace layout.",
        },
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Unique stable ID as {color}_{type}_{nn}, e.g. green_cube_01 or yellow_tray_01.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short description (e.g. 'small green cube'). Not a copy of handle.",
                    },
                    "bounding_box": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[x1, y1, x2, y2] in pixels.",
                    },
                    "is_graspable": {
                        "type": "boolean",
                        "description": "true for pickable items, false for containers and robot hand.",
                    },
                },
                "required": ["handle", "label", "bounding_box", "is_graspable"],
            },
        },
    },
    "required": ["scene", "detections"],
}

_GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "string",
            "description": "One sentence describing the workspace layout.",
        },
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Unique stable ID as {color}_{type}_{nn}, e.g. green_cube_01 or yellow_tray_01.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short description (e.g. 'small green cube'). Not a copy of handle.",
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[y_min, x_min, y_max, x_max] in 0-1000 range.",
                    },
                    "is_graspable": {
                        "type": "boolean",
                        "description": "true for pickable items, false for containers and robot hand.",
                    },
                },
                "required": ["handle", "label", "box_2d", "is_graspable"],
            },
        },
    },
    "required": ["scene", "detections"],
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


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


def _resize_image(pil: Image.Image, max_width: int = _VLM_INPUT_WIDTH_GEMINI) -> tuple[Image.Image, int, int]:
    """Resize if wider than *max_width*. Returns (image, sent_w, sent_h)."""
    if pil.width > max_width:
        aspect = pil.height / pil.width
        new_w = max_width
        new_h = int(new_w * aspect)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return pil, pil.width, pil.height


def _pil_to_b64(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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


# ---------------------------------------------------------------------------
# Provider-specific API calls (blocking — run via to_thread)
# ---------------------------------------------------------------------------


def _call_gemini_sync(api_key: str, model: str, prompt: str, pil_image: Image.Image) -> tuple[dict, dict]:
    """Call Gemini VLM. Returns (parsed_json, token_usage_dict)."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[pil_image, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _GEMINI_SCHEMA,
        },
    )
    token_usage: dict[str, int] = {}
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        if getattr(um, "prompt_token_count", None) is not None:
            token_usage["prompt_tokens"] = um.prompt_token_count
        if getattr(um, "candidates_token_count", None) is not None:
            token_usage["completion_tokens"] = um.candidates_token_count
        if getattr(um, "total_token_count", None) is not None:
            token_usage["total_tokens"] = um.total_token_count
        if getattr(um, "cached_content_token_count", None):
            token_usage["cached_tokens"] = um.cached_content_token_count
    return _extract_json(response.text or ""), token_usage


def _call_ollama_sync(base_url: str, model: str, prompt: str, image_b64: str) -> tuple[dict, dict]:
    """Call Ollama VLM. Returns (full_response_dict, token_usage_dict)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": _OLLAMA_SCHEMA,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    token_usage: dict[str, int] = {}
    if result.get("prompt_eval_count") is not None:
        token_usage["prompt_tokens"] = result["prompt_eval_count"]
    if result.get("eval_count") is not None:
        token_usage["completion_tokens"] = result["eval_count"]
    if token_usage.get("prompt_tokens") is not None and token_usage.get("completion_tokens") is not None:
        token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
    return result, token_usage


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


def _normalize_gemini(raw: dict) -> None:
    """Convert Gemini box_2d [y_min,x_min,y_max,x_max] 0-1000 → bounding_box [x1,y1,x2,y2] 0..1."""
    for det in raw.get("detections", []):
        bb = det.pop("box_2d", None) or det.get("bounding_box")
        if bb and len(bb) == 4:
            y_min, x_min, y_max, x_max = bb
            if any(v > 1.0 for v in bb):
                det["bounding_box"] = [x_min / 1000.0, y_min / 1000.0, x_max / 1000.0, y_max / 1000.0]
            else:
                det["bounding_box"] = [x_min, y_min, x_max, y_max]


def _normalize_ollama(raw: dict) -> None:
    """Normalize any non-standard Ollama response shapes."""
    # Fallback for models that ignore structured output
    if "detections" not in raw and "graspable_objects" in raw:
        objects = raw.pop("graspable_objects")
        robot_hand = raw.pop("robot_hand", None)
        dets = []
        counter: dict[str, int] = {}
        for obj in objects:
            label = obj.get("label", "object")
            key = label.lower().replace(" ", "_")
            counter[key] = counter.get(key, 0) + 1
            handle = f"{key}_{counter[key]:02d}"
            bb = obj.get("bbox_2d") or obj.get("bounding_box") or obj.get("box_2d")
            dets.append(
                {
                    "handle": obj.get("handle", handle),
                    "label": label,
                    "bounding_box": bb or [0, 0, 0, 0],
                    "is_graspable": obj.get("is_graspable", True),
                }
            )
        if robot_hand and isinstance(robot_hand, dict):
            bb = robot_hand.get("bbox_2d") or robot_hand.get("bounding_box") or robot_hand.get("box_2d")
            if bb:
                dets.append(
                    {
                        "handle": "robot_hand_01",
                        "label": "robot hand",
                        "bounding_box": bb,
                        "is_graspable": False,
                    }
                )
        raw["detections"] = dets
        if "scene" not in raw:
            raw["scene"] = ""

    for det in raw.get("detections", []):
        if "bounding_box" not in det:
            bb = det.pop("bbox_2d", None) or det.pop("box_2d", None)
            if bb:
                det["bounding_box"] = bb


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------


def make_vlm_fn(
    provider: str = "ollama",
    model: str = "qwen2.5vl:3b",
    api_key: str | None = None,
    base_url: str = "http://localhost:11434",
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    """Return an async VlmFn backed by *provider* ("gemini" or "ollama").

    Gemini: bboxes arrive as box_2d [y_min,x_min,y_max,x_max] 0-1000,
    normalized to [x1,y1,x2,y2] 0..1. Pass img_w=img_h=1.

    Ollama: bboxes arrive as bounding_box in pixel coords,
    normalized via img_w/img_h from the sent image dimensions.
    """
    prompt = prompt_path.read_text(encoding="utf-8")
    is_gemini = provider == "gemini"

    if is_gemini:
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def vlm_fn(
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        if image is None:
            return VlmScene(scene="", detections=[])

        pil_image = _to_pil(image)
        max_w = _VLM_INPUT_WIDTH_GEMINI if is_gemini else _VLM_INPUT_WIDTH_OLLAMA
        pil_image, sent_w, sent_h = _resize_image(pil_image, max_width=max_w)

        effective_prompt = prompt
        if known_handles:
            handles_str = ", ".join(known_handles)
            effective_prompt += f"\nPreviously known: {handles_str}. Reuse these handles for the same objects."

        t0 = time.monotonic()
        raw: dict = {}
        vlm_token_usage: dict[str, int] = {}
        error: str | None = None

        try:
            if is_gemini:
                raw, vlm_token_usage = await asyncio.to_thread(
                    _call_gemini_sync, resolved_key, model, effective_prompt, pil_image
                )
            else:
                image_b64 = _pil_to_b64(pil_image)
                result, vlm_token_usage = await asyncio.to_thread(
                    _call_ollama_sync, base_url, model, effective_prompt, image_b64
                )
                raw = _extract_json(result.get("response", ""))
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
            raise

        # Normalize provider-specific response into canonical bounding_box format
        if is_gemini:
            _normalize_gemini(raw)
            img_w, img_h = 1, 1  # already 0..1 after normalization
        else:
            _normalize_ollama(raw)
            img_w, img_h = sent_w, sent_h  # pixel coords, parse_vlm_response normalizes

        vlm_scene = parse_vlm_response(raw, img_w=img_w, img_h=img_h)

        if run_logger is not None:
            det_dicts = [{"handle": d.handle, "bbox": d.bbox} for d in vlm_scene.detections]
            loggable = raw if is_gemini else {k: v for k, v in raw.items() if k != "context"}
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
                token_usage=vlm_token_usage,
            )

        vlm_fn.last_token_usage = vlm_token_usage  # type: ignore[attr-defined]
        return dataclasses.replace(vlm_scene, token_usage=vlm_token_usage)

    vlm_fn.last_token_usage = {}  # type: ignore[attr-defined]
    return vlm_fn
