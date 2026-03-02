"""OpenCV tracker factory for TargetPerceptionService.

Uses the best available OpenCV tracker (TrackerVit > TrackerNano >
TrackerDaSiamRPN > TrackerMIL) to follow a target initialised from a VLM
detection bounding box.  Returns a ``TrackerFactoryFn``-compatible callable.

All coordinates exposed to callers are **normalised to 0..1** (fraction of
frame width/height).  Pixel conversion happens only at the OpenCV boundary.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.vlm_parser import VlmDetection

__all__ = ["get_tracker_name", "make_tracker_factory_fn"]

# NanoTrack ONNX model paths (relative to repo root).
_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
_NANOTRACK_BACKBONE = _MODELS_DIR / "nanotrack_backbone_sim.onnx"
_NANOTRACK_HEAD = _MODELS_DIR / "nanotrack_head_sim.onnx"


def _bbox_norm_to_pixel_xywh(
    bbox_xyxy_norm: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalised (x1, y1, x2, y2) 0..1 → pixel (x, y, w, h)."""
    x1, y1, x2, y2 = bbox_xyxy_norm
    px1, py1 = int(x1 * img_w), int(y1 * img_h)
    px2, py2 = int(x2 * img_w), int(y2 * img_h)
    return px1, py1, px2 - px1, py2 - py1


def _target_info_from_pixel_bbox(
    handle: str,
    px_xywh: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    confidence: float = 0.8,
) -> TargetInfo:
    """Build a TargetInfo with normalised 0..1 coords from a pixel bbox."""
    x, y, w, h = px_xywh
    cx_norm = (x + w / 2) / img_w
    cy_norm = (y + h / 2) / img_h

    return TargetInfo(
        handle=handle,
        hint_valid=True,
        confidence=confidence,
        obs_age_ms=0,
        time_skew_ms=0,
        delta_xyz_ee=(0.0, 0.0, 0.0),
        distance_m=0.0,
        center_px=(cx_norm, cy_norm),
        bbox_xywh=(x / img_w, y / img_h, w / img_w, h / img_h),
    )


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


# Tracker names → ``cv2`` factory function names, ordered by preference.
# Availability varies by OpenCV build; we try each at runtime via getattr.
_TRACKER_FACTORIES: list[tuple[str, str]] = [
    ("TrackerVit", "TrackerVit_create"),
    ("TrackerNano", "TrackerNano_create"),
    ("TrackerDaSiamRPN", "TrackerDaSiamRPN_create"),
    ("TrackerMIL", "TrackerMIL_create"),
]


def _create_nano_tracker() -> cv2.Tracker | None:
    """Try to create a TrackerNano with the bundled ONNX models.

    Returns ``None`` if the models are missing or TrackerNano_Params is
    unavailable (e.g. plain ``opencv-python`` without contrib).
    """
    if not (_NANOTRACK_BACKBONE.exists() and _NANOTRACK_HEAD.exists()):
        return None
    params_cls = getattr(cv2, "TrackerNano_Params", None)
    create_fn = getattr(cv2, "TrackerNano_create", None)
    if params_cls is None or create_fn is None:
        return None
    try:
        params = params_cls()
        params.backbone = str(_NANOTRACK_BACKBONE)
        params.neckhead = str(_NANOTRACK_HEAD)
        return create_fn(params)
    except cv2.error:
        return None


def _create_tracker() -> cv2.Tracker:
    """Create the best available OpenCV tracker."""
    for name, attr in _TRACKER_FACTORIES:
        # TrackerNano needs model paths — use dedicated helper.
        if name == "TrackerNano":
            tracker = _create_nano_tracker()
            if tracker is not None:
                return tracker
            continue
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
        if name == "TrackerNano":
            if _create_nano_tracker() is not None:
                return name
            continue
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

    Bbox / centroid coordinates in the returned ``TargetInfo`` are normalised
    to 0..1 (fraction of frame dimensions).

    Usage::

        factory = make_tracker_factory_fn()
        svc = TargetPerceptionService(..., tracker_factory_fn=factory)
    """

    async def factory(frame: CapturedFrame, detection: VlmDetection) -> tuple[TargetInfo, object]:
        bgr = _to_bgr(frame.image)
        img_h, img_w = bgr.shape[:2]

        # Denormalise 0..1 bbox → pixel xywh for OpenCV tracker init
        px_xywh = _bbox_norm_to_pixel_xywh(detection.bbox, img_w, img_h)

        tracker = _create_tracker()
        tracker.init(bgr, px_xywh)

        init_hint = _target_info_from_pixel_bbox(detection.handle, px_xywh, img_w, img_h, confidence=0.9)

        async def update(f: CapturedFrame) -> TargetInfo | None:
            frame_bgr = _to_bgr(f.image)
            fh, fw = frame_bgr.shape[:2]

            ok, new_bbox = tracker.update(frame_bgr)
            if not ok:
                return None

            bx, by, bw, bh = [int(v) for v in new_bbox]
            return _target_info_from_pixel_bbox(
                detection.handle,
                (bx, by, bw, bh),
                fw,
                fh,
                confidence=0.8,
            )

        return init_hint, update

    return factory
