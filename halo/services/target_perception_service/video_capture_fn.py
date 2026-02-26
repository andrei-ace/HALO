"""Factory for a ``CaptureFn`` backed by a looping video file.

Usage::

    from halo.services.target_perception_service.video_capture_fn import make_video_capture_fn

    capture_fn = make_video_capture_fn("data/video.mp4")
    frame = await capture_fn("arm0")   # CapturedFrame with numpy image
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from halo.services.target_perception_service.frame_buffer import CapturedFrame

_DEFAULT_VIDEO = Path(__file__).parents[3] / "data" / "video.mp4"


def make_video_capture_fn(video_path: str | Path = _DEFAULT_VIDEO) -> object:
    """Return a ``CaptureFn`` that reads frames from *video_path*, looping forever.

    Each call decodes the next frame and returns it as a ``CapturedFrame`` with a
    ``numpy.ndarray`` (BGR, HWC) in the ``image`` field.  When the video reaches
    the end it seeks back to the start automatically.

    The ``cv2.VideoCapture`` is opened lazily on first call so the factory itself
    is cheap and does not hold file handles until actually used.
    """
    video_path = Path(video_path)
    cap: cv2.VideoCapture | None = None

    def _ensure_open() -> cv2.VideoCapture:
        nonlocal cap
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
        return cap

    def _read_frame() -> np.ndarray:
        vc = _ensure_open()
        ok, frame = vc.read()
        if not ok:
            # End of video — loop
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = vc.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame from video: {video_path}")
        return frame

    async def capture_fn(arm_id: str) -> CapturedFrame:
        return CapturedFrame(
            image=_read_frame(),
            ts_ms=int(time.monotonic() * 1000),
            arm_id=arm_id,
        )

    def release() -> None:
        nonlocal cap
        if cap is not None:
            cap.release()
            cap = None

    # Attach release() so callers can clean up the VideoCapture handle.
    capture_fn.release = release  # type: ignore[attr-defined]

    return capture_fn
