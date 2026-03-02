"""Tests for halo.services.target_perception_service.tracker_fn.

Covers: bbox conversion, image type coercion (_to_bgr), TargetInfo construction,
tracker discovery/fallback, get_tracker_name(), and the full factory + update cycle.

All bbox/centroid coordinates are normalised to 0..1.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.tracker_fn import (
    _bbox_norm_to_pixel_xywh,
    _create_tracker,
    _target_info_from_pixel_bbox,
    _to_bgr,
    get_tracker_name,
    make_tracker_factory_fn,
)
from halo.services.target_perception_service.vlm_parser import VlmDetection

# ---------------------------------------------------------------------------
# _bbox_norm_to_pixel_xywh
# ---------------------------------------------------------------------------


def test_bbox_norm_to_pixel_xywh_basic():
    # (0.1, 0.2, 0.6, 0.7) on 200x100 image → (20, 20, 100, 50)
    assert _bbox_norm_to_pixel_xywh((0.1, 0.2, 0.6, 0.7), 200, 100) == (20, 20, 100, 50)


def test_bbox_norm_to_pixel_xywh_full_frame():
    # Full frame → (0, 0, 640, 480)
    assert _bbox_norm_to_pixel_xywh((0.0, 0.0, 1.0, 1.0), 640, 480) == (0, 0, 640, 480)


# ---------------------------------------------------------------------------
# _to_bgr
# ---------------------------------------------------------------------------


def test_to_bgr_numpy_passthrough():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _to_bgr(arr)
    assert result is arr


def test_to_bgr_pil_image():
    from PIL import Image

    pil = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
    result = _to_bgr(pil)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)


def test_to_bgr_bytes():
    from io import BytesIO

    from PIL import Image

    pil = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    raw = buf.getvalue()

    result = _to_bgr(raw)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 50
    assert result.shape[1] == 50


def test_to_bgr_bytearray():
    from io import BytesIO

    from PIL import Image

    pil = Image.fromarray(np.zeros((30, 30, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    raw = bytearray(buf.getvalue())

    result = _to_bgr(raw)
    assert isinstance(result, np.ndarray)


def test_to_bgr_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported image type"):
        _to_bgr("not an image")


# ---------------------------------------------------------------------------
# _target_info_from_pixel_bbox
# ---------------------------------------------------------------------------


def test_target_info_from_pixel_bbox_centroid():
    info = _target_info_from_pixel_bbox("cube-1", (100, 200, 50, 80), 400, 400)
    # centre = (100+25)/400, (200+40)/400 = 0.3125, 0.6
    assert info.center_px == (0.3125, 0.6)
    assert info.handle == "cube-1"
    assert info.hint_valid is True
    assert info.confidence == 0.8


def test_target_info_from_pixel_bbox_custom_confidence():
    info = _target_info_from_pixel_bbox("obj", (0, 0, 10, 10), 100, 100, confidence=0.95)
    assert info.confidence == 0.95


def test_target_info_from_pixel_bbox_zeroed_3d():
    info = _target_info_from_pixel_bbox("obj", (0, 0, 10, 10), 100, 100)
    assert info.delta_xyz_ee == (0.0, 0.0, 0.0)
    assert info.distance_m == 0.0


def test_target_info_from_pixel_bbox_normalised_bbox():
    info = _target_info_from_pixel_bbox("cube-1", (100, 200, 50, 80), 400, 400)
    # bbox_xywh = (100/400, 200/400, 50/400, 80/400) = (0.25, 0.5, 0.125, 0.2)
    assert info.bbox_xywh == (0.25, 0.5, 0.125, 0.2)


# ---------------------------------------------------------------------------
# _create_tracker — fallback chain
# ---------------------------------------------------------------------------


def test_create_tracker_returns_tracker():
    """Should return a valid cv2.Tracker when at least one backend works."""
    tracker = _create_tracker()
    assert hasattr(tracker, "init")
    assert hasattr(tracker, "update")


def test_create_tracker_skips_cv2_error():
    """When all factories raise cv2.error, RuntimeError is raised."""
    with patch(
        "halo.services.target_perception_service.tracker_fn._TRACKER_FACTORIES",
        [("FakeTracker", "FakeTracker_create")],
    ):
        fake_fn = MagicMock(side_effect=cv2.error("fail"))
        with patch("halo.services.target_perception_service.tracker_fn.cv2") as mock_cv2:
            mock_cv2.error = cv2.error
            mock_cv2.FakeTracker_create = fake_fn
            with pytest.raises(RuntimeError, match="No suitable OpenCV tracker"):
                _create_tracker()


def test_create_tracker_none_available():
    """When no factory exists at all, RuntimeError is raised."""
    with patch(
        "halo.services.target_perception_service.tracker_fn._TRACKER_FACTORIES",
        [("Missing", "Missing_create")],
    ):
        with pytest.raises(RuntimeError, match="No suitable OpenCV tracker"):
            _create_tracker()


# ---------------------------------------------------------------------------
# get_tracker_name
# ---------------------------------------------------------------------------


def test_get_tracker_name_returns_string():
    name = get_tracker_name()
    # Should find at least TrackerMIL in this environment
    assert isinstance(name, str)
    assert name != "none"


def test_get_tracker_name_none_when_nothing_available():
    with patch(
        "halo.services.target_perception_service.tracker_fn._TRACKER_FACTORIES",
        [("Missing", "Missing_create")],
    ):
        assert get_tracker_name() == "none"


def test_get_tracker_name_skips_cv2_error():
    with patch(
        "halo.services.target_perception_service.tracker_fn._TRACKER_FACTORIES",
        [("Bad", "Bad_create")],
    ):
        fake_fn = MagicMock(side_effect=cv2.error("fail"))
        with patch("halo.services.target_perception_service.tracker_fn.cv2") as mock_cv2:
            mock_cv2.error = cv2.error
            mock_cv2.Bad_create = fake_fn
            assert get_tracker_name() == "none"


# ---------------------------------------------------------------------------
# make_tracker_factory_fn — full init + update cycle
# ---------------------------------------------------------------------------


def _make_frame(w: int = 200, h: int = 200) -> CapturedFrame:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return CapturedFrame(image=img, ts_ms=0, arm_id="arm0")


def _make_detection(handle: str = "cube-1") -> VlmDetection:
    """Detection with normalised 0..1 bbox (on a 200x200 frame → centre quarter)."""
    return VlmDetection(
        handle=handle,
        label="cube",
        bbox=(0.25, 0.25, 0.5, 0.5),
        centroid=(0.375, 0.375),
        is_graspable=True,
    )


async def test_factory_init_returns_hint_and_update():
    factory = make_tracker_factory_fn()
    frame = _make_frame()
    detection = _make_detection()

    init_hint, update_fn = await factory(frame, detection)

    assert init_hint.handle == "cube-1"
    assert init_hint.confidence == 0.9  # init confidence
    assert init_hint.hint_valid is True
    # On 200x200: pixel xywh=(50,50,50,50), centre=(75/200,75/200)=(0.375,0.375)
    assert init_hint.center_px == (0.375, 0.375)
    assert init_hint.delta_xyz_ee == (0.0, 0.0, 0.0)
    assert init_hint.distance_m == 0.0
    assert callable(update_fn)


async def test_factory_update_returns_target_info():
    factory = make_tracker_factory_fn()
    frame = _make_frame()
    detection = _make_detection()

    _, update_fn = await factory(frame, detection)

    # Feed a similar frame — tracker should still track
    next_frame = _make_frame()
    result = await update_fn(next_frame)

    # TrackerMIL may or may not track on random noise, but the function should
    # return either a TargetInfo or None without error
    if result is not None:
        assert result.handle == "cube-1"
        assert result.confidence == 0.8  # update confidence
        assert result.delta_xyz_ee == (0.0, 0.0, 0.0)
        assert result.distance_m == 0.0
        assert result.center_px is not None
        # All coords should be normalised 0..1
        assert 0.0 <= result.center_px[0] <= 1.0
        assert 0.0 <= result.center_px[1] <= 1.0


async def test_factory_update_with_pil_image():
    """The update function should handle PIL images via _to_bgr."""
    from PIL import Image

    factory = make_tracker_factory_fn()
    frame = _make_frame()
    detection = _make_detection()

    _, update_fn = await factory(frame, detection)

    pil_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    pil_frame = CapturedFrame(image=pil_img, ts_ms=1, arm_id="arm0")

    # Should not raise
    result = await update_fn(pil_frame)
    assert result is None or result.handle == "cube-1"


async def test_factory_init_with_pil_image():
    """The factory should accept PIL images for the init frame."""
    from PIL import Image

    factory = make_tracker_factory_fn()
    pil_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    frame = CapturedFrame(image=pil_img, ts_ms=0, arm_id="arm0")
    detection = _make_detection()

    init_hint, update_fn = await factory(frame, detection)
    assert init_hint.handle == "cube-1"
    assert callable(update_fn)
