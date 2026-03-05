"""Unit tests for Gemini VLM function (mocked — no API key needed)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from halo.services.target_perception_service.gemini_vlm_fn import (
    _extract_json,
    _resize_image,
    _to_pil,
    make_gemini_vlm_fn,
)
from halo.services.target_perception_service.vlm_parser import VlmScene

# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_bare():
    assert _extract_json('{"scene": "table"}') == {"scene": "table"}


def test_extract_json_fenced():
    text = '```json\n{"scene": "table"}\n```'
    assert _extract_json(text) == {"scene": "table"}


def test_extract_json_embedded():
    text = 'Here is the result: {"scene": "table"} end.'
    assert _extract_json(text) == {"scene": "table"}


def test_extract_json_invalid():
    assert _extract_json("not json at all") == {}


# ---------------------------------------------------------------------------
# _to_pil / _resize_image
# ---------------------------------------------------------------------------


def test_to_pil_numpy():
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    pil = _to_pil(arr)
    assert pil.size == (200, 100)


def test_resize_image_no_op():
    from PIL import Image

    pil = Image.new("RGB", (512, 256))
    resized, w, h = _resize_image(pil)
    assert w == 512
    assert h == 256


def test_resize_image_downscale():
    from PIL import Image

    pil = Image.new("RGB", (2048, 1024))
    resized, w, h = _resize_image(pil)
    assert w == 768
    assert h == 384


# ---------------------------------------------------------------------------
# make_gemini_vlm_fn — mocked
# ---------------------------------------------------------------------------

# Gemini returns box_2d: [y_min, x_min, y_max, x_max] in 0-1000.
_MOCK_RESPONSE_BOX2D = {
    "scene": "A table with a red cube.",
    "detections": [
        {
            "handle": "red_cube_01",
            "label": "red cube",
            "box_2d": [100, 200, 300, 400],
            "is_graspable": True,
        }
    ],
}

# Fallback: legacy bounding_box field (e.g. from Ollama) already 0..1 [y,x,y,x].
_MOCK_RESPONSE_LEGACY_01 = {
    "scene": "A table with a red cube.",
    "detections": [
        {
            "handle": "red_cube_01",
            "label": "red cube",
            "bounding_box": [0.48, 0.28, 0.54, 0.34],
            "is_graspable": True,
        }
    ],
}


@pytest.mark.asyncio
async def test_gemini_vlm_fn_box2d_1000():
    """box_2d [y_min,x_min,y_max,x_max] 0-1000 → bbox [x1,y1,x2,y2] 0..1."""
    with patch(
        "halo.services.target_perception_service.vlm_fn._call_gemini_sync",
        return_value=_MOCK_RESPONSE_BOX2D,
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        scene = await vlm_fn("arm0", img)

    assert isinstance(scene, VlmScene)
    assert scene.scene == "A table with a red cube."
    assert len(scene.detections) == 1
    det = scene.detections[0]
    assert det.handle == "red_cube_01"
    assert det.is_graspable is True
    # box_2d [y_min=100, x_min=200, y_max=300, x_max=400]
    # → bounding_box [x1=0.2, y1=0.1, x2=0.4, y2=0.3]
    assert det.bbox[0] == pytest.approx(0.2)
    assert det.bbox[1] == pytest.approx(0.1)
    assert det.bbox[2] == pytest.approx(0.4)
    assert det.bbox[3] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_gemini_vlm_fn_legacy_bbox_0_1():
    """Legacy bounding_box in 0..1 [y,x,y,x] is reordered to [x,y,x,y]."""
    with patch(
        "halo.services.target_perception_service.vlm_fn._call_gemini_sync",
        return_value=_MOCK_RESPONSE_LEGACY_01,
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        scene = await vlm_fn("arm0", img)

    det = scene.detections[0]
    # bounding_box [y_min=0.48, x_min=0.28, y_max=0.54, x_max=0.34]
    # → [x1=0.28, y1=0.48, x2=0.34, y2=0.54]
    assert det.bbox[0] == pytest.approx(0.28)
    assert det.bbox[1] == pytest.approx(0.48)
    assert det.bbox[2] == pytest.approx(0.34)
    assert det.bbox[3] == pytest.approx(0.54)


@pytest.mark.asyncio
async def test_gemini_vlm_fn_none_image():
    """None image returns empty VlmScene."""
    vlm_fn = make_gemini_vlm_fn(api_key="test-key")
    scene = await vlm_fn("arm0", None)
    assert scene == VlmScene(scene="", detections=[])


@pytest.mark.asyncio
async def test_gemini_vlm_fn_error_raises():
    """API errors are raised so the switchboard can catch and failover."""
    with patch(
        "halo.services.target_perception_service.vlm_fn._call_gemini_sync",
        side_effect=RuntimeError("API quota exceeded"),
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="API quota exceeded"):
            await vlm_fn("arm0", img)


@pytest.mark.asyncio
async def test_gemini_vlm_fn_known_handles():
    """Known handles are appended to the prompt."""
    call_args = {}

    def fake_call(api_key, model, prompt, pil_image):
        call_args["prompt"] = prompt
        return {"scene": "table", "detections": []}

    with patch(
        "halo.services.target_perception_service.vlm_fn._call_gemini_sync",
        side_effect=fake_call,
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        await vlm_fn("arm0", img, known_handles=["red_cube_01", "green_cube_01"])

    assert "red_cube_01" in call_args["prompt"]
    assert "green_cube_01" in call_args["prompt"]
