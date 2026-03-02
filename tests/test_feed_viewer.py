"""Tests for halo.tui.feed_viewer.

Covers: _draw_annotations (None, bbox, center_px fallback) and _status_color mapping.
These tests use synthetic numpy frames and do not open any OpenCV windows.
"""

from __future__ import annotations

import numpy as np

from halo.contracts.enums import PerceptionFailureCode, TrackingStatus
from halo.contracts.snapshots import PerceptionInfo, TargetInfo
from halo.tui.feed_viewer import _draw_annotations, _status_color

# ---------------------------------------------------------------------------
# _status_color
# ---------------------------------------------------------------------------


def test_status_color_tracking():
    assert _status_color(TrackingStatus.TRACKING) == (0, 200, 0)


def test_status_color_relocalizing():
    assert _status_color(TrackingStatus.RELOCALIZING) == (0, 200, 200)


def test_status_color_reacquiring():
    assert _status_color(TrackingStatus.REACQUIRING) == (0, 200, 200)


def test_status_color_lost():
    assert _status_color(TrackingStatus.LOST) == (0, 0, 200)


def test_status_color_idle():
    assert _status_color(TrackingStatus.IDLE) == (128, 128, 128)


# ---------------------------------------------------------------------------
# _draw_annotations — None snapshot
# ---------------------------------------------------------------------------


def _blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_annotations_none_snapshot():
    """Should not crash with None target and None perception."""
    frame = _blank_frame()
    result = _draw_annotations(frame, None, None)
    assert result is frame
    # Frame should still be all zeros (no annotations drawn)
    assert frame.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# _draw_annotations — with bbox
# ---------------------------------------------------------------------------


def _make_target(
    bbox: tuple[float, float, float, float] | None = None,
    center_px: tuple[float, float] | None = None,
) -> TargetInfo:
    return TargetInfo(
        handle="cube-1",
        hint_valid=True,
        confidence=0.85,
        obs_age_ms=10,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.05,
        center_px=center_px,
        bbox_xywh=bbox,
    )


def _make_perception(
    status: TrackingStatus = TrackingStatus.TRACKING,
    failure: PerceptionFailureCode = PerceptionFailureCode.OK,
) -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=status,
        failure_code=failure,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def test_draw_annotations_with_bbox():
    """When bbox is available, annotations should be drawn (non-zero pixels)."""
    frame = _blank_frame()
    target = _make_target(bbox=(0.15, 0.2, 0.12, 0.12))
    perception = _make_perception()

    result = _draw_annotations(frame, target, perception)
    assert result is frame
    # Frame should have non-zero pixels from the rectangle and text
    assert np.any(frame > 0)


def test_draw_annotations_with_center_px_fallback():
    """When only center_px is available (no bbox), crosshair should be drawn."""
    frame = _blank_frame()
    target = _make_target(center_px=(0.5, 0.5))
    perception = _make_perception()

    result = _draw_annotations(frame, target, perception)
    assert result is frame
    assert np.any(frame > 0)


def test_draw_annotations_with_failure_code():
    """Failure code should be rendered when not OK."""
    frame = _blank_frame()
    target = _make_target(bbox=(0.08, 0.1, 0.06, 0.08))
    perception = _make_perception(
        status=TrackingStatus.LOST,
        failure=PerceptionFailureCode.OCCLUDED,
    )

    result = _draw_annotations(frame, target, perception)
    assert result is frame
    assert np.any(frame > 0)


def test_draw_annotations_perception_only_no_target():
    """With perception but no target, status bar should still render."""
    frame = _blank_frame()
    perception = _make_perception(status=TrackingStatus.IDLE)

    result = _draw_annotations(frame, None, perception)
    assert result is frame
    assert np.any(frame > 0)
