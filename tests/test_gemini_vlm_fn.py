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
    assert w == 1024
    assert h == 512


# ---------------------------------------------------------------------------
# make_gemini_vlm_fn — mocked
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = {
    "scene": "A table with a red cube.",
    "detections": [
        {
            "handle": "red_cube_01",
            "label": "red cube",
            "bounding_box": [100, 200, 300, 400],
            "is_graspable": True,
        }
    ],
}


@pytest.mark.asyncio
async def test_gemini_vlm_fn_returns_vlm_scene():
    """Verify make_gemini_vlm_fn returns a VlmScene with correct detections."""
    with patch(
        "halo.services.target_perception_service.gemini_vlm_fn._call_gemini_sync",
        return_value=_MOCK_RESPONSE,
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
    # Bbox should be normalised (640x480 image, no resize needed)
    assert det.bbox[0] == pytest.approx(100 / 640)
    assert det.bbox[1] == pytest.approx(200 / 480)


@pytest.mark.asyncio
async def test_gemini_vlm_fn_none_image():
    """None image returns empty VlmScene."""
    vlm_fn = make_gemini_vlm_fn(api_key="test-key")
    scene = await vlm_fn("arm0", None)
    assert scene == VlmScene(scene="", detections=[])


@pytest.mark.asyncio
async def test_gemini_vlm_fn_error_handling():
    """API errors return empty VlmScene, not raise."""
    with patch(
        "halo.services.target_perception_service.gemini_vlm_fn._call_gemini_sync",
        side_effect=RuntimeError("API quota exceeded"),
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        scene = await vlm_fn("arm0", img)

    assert scene == VlmScene(scene="", detections=[])


@pytest.mark.asyncio
async def test_gemini_vlm_fn_known_handles():
    """Known handles are appended to the prompt."""
    call_args = {}

    def fake_call(api_key, model, prompt, pil_image):
        call_args["prompt"] = prompt
        return {"scene": "table", "detections": []}

    with patch(
        "halo.services.target_perception_service.gemini_vlm_fn._call_gemini_sync",
        side_effect=fake_call,
    ):
        vlm_fn = make_gemini_vlm_fn(api_key="test-key")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        await vlm_fn("arm0", img, known_handles=["red_cube_01", "green_cube_01"])

    assert "red_cube_01" in call_args["prompt"]
    assert "green_cube_01" in call_args["prompt"]
