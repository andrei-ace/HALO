"""OpenCV tracker factory for TargetPerceptionService.

Uses the best available OpenCV tracker (TrackerVit > TrackerNano >
TrackerDaSiamRPN > TrackerMIL) to follow a target initialised from a VLM
detection bounding box.  Returns a ``TrackerFactoryFn``-compatible callable.

Since these are 2D trackers and we have no depth sensor in v0,
``delta_xyz_ee`` and ``distance_m`` are zeroed.  Only ``center_px``
carries real data (bbox centroid in pixels).  Real 3D will come from
ZED X depth fusion in the hardware phase.
"""

from __future__ import annotations

import cv2
import numpy as np

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.vlm_parser import VlmDetection


def _bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) to OpenCV (x, y, w, h) integer tuple."""
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _to_bgr(image: object) -> np.ndarray:
    """Ensure *image* is a numpy BGR HWC array (what OpenCV expects)."""
    if isinstance(image, np.ndarray):
        return image
    # PIL or bytes → numpy BGR
    import io

    from PIL import Image

    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _target_info_from_bbox(
    handle: str,
    bbox_xywh: tuple[int, int, int, int],
    confidence: float = 0.8,
) -> TargetInfo:
    """Build a TargetInfo from a 2D bounding box (no depth).

    Only ``center_px`` carries real data (bbox centroid in pixels).
    ``delta_xyz_ee`` and ``distance_m`` are zeroed — real 3D comes
    from ZED X depth fusion in the hardware phase.
    """
    x, y, w, h = bbox_xywh
    cx = x + w / 2
    cy = y + h / 2

    return TargetInfo(
        handle=handle,
        hint_valid=True,
        confidence=confidence,
        obs_age_ms=0,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.0,
        center_px=(cx, cy),
        bbox_xywh=bbox_xywh,
    )


# Tracker names → ``cv2`` factory function names, ordered by preference.
# Availability varies by OpenCV build; we try each at runtime via getattr.
_TRACKER_FACTORIES: list[tuple[str, str]] = [
    ("TrackerVit", "TrackerVit_create"),
    ("TrackerNano", "TrackerNano_create"),
    ("TrackerDaSiamRPN", "TrackerDaSiamRPN_create"),
    ("TrackerMIL", "TrackerMIL_create"),
]


def _create_tracker() -> cv2.Tracker:
    """Create the best available OpenCV tracker."""
    for _, attr in _TRACKER_FACTORIES:
        fn = getattr(cv2, attr, None)
        if fn is not None:
            try:
                return fn()
            except cv2.error:
                continue
    raise RuntimeError("No suitable OpenCV tracker available")


def get_tracker_name() -> str:
    """Return the name of the best available OpenCV tracker without allocating one."""
    for name, attr in _TRACKER_FACTORIES:
        fn = getattr(cv2, attr, None)
        if fn is not None:
            try:
                fn()
                return name
            except cv2.error:
                continue
    return "none"


def make_tracker_factory_fn():
    """Return a ``TrackerFactoryFn`` backed by the best available OpenCV tracker.

    Usage::

        factory = make_tracker_factory_fn()
        svc = TargetPerceptionService(..., tracker_factory_fn=factory)
    """

    async def factory(frame: CapturedFrame, detection: VlmDetection) -> tuple[TargetInfo, object]:
        bgr = _to_bgr(frame.image)

        x1, y1, x2, y2 = detection.bbox
        bbox_xywh = _bbox_xyxy_to_xywh(x1, y1, x2, y2)

        tracker = _create_tracker()
        tracker.init(bgr, bbox_xywh)

        init_hint = _target_info_from_bbox(detection.handle, bbox_xywh, confidence=0.9)

        async def update(f: CapturedFrame) -> TargetInfo | None:
            frame_bgr = _to_bgr(f.image)

            ok, new_bbox = tracker.update(frame_bgr)
            if not ok:
                return None

            bx, by, bw, bh = [int(v) for v in new_bbox]
            return _target_info_from_bbox(
                detection.handle,
                (bx, by, bw, bh),
                confidence=0.8,
            )

        return init_hint, update

    return factory
