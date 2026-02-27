"""Shared video source that reads frames at native FPS in a background thread.

Simulates a real-time camera: one reader, multiple consumers.
Both the perception service (sequential frames via ``CaptureFn``) and the
feed viewer (latest frame for display) read from this single source.

In production, replace this with a real camera source exposing the same
interface (``make_capture_fn`` + ``latest_frame``).

Usage::

    from halo.services.target_perception_service.video_source import VideoSource

    source = VideoSource("data/video.mp4")
    source.start()

    capture_fn = source.make_capture_fn("arm0")   # for perception service
    frame = source.latest_frame                    # for feed viewer

    source.stop()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from halo.services.target_perception_service.frame_buffer import CapturedFrame

_DEFAULT_VIDEO = Path(__file__).parents[3] / "data" / "video.mp4"


class VideoSource:
    """Background-thread video reader that simulates a real-time camera.

    Reads frames from a video file at its native FPS.  Consumers access
    frames through two channels:

    - ``make_capture_fn(arm_id)`` — returns a ``CaptureFn`` that yields
      sequential frames (one per call, FIFO).  The tracker needs every
      frame in order.
    - ``latest_frame`` — the most recent frame as a numpy array (BGR HWC).
      The feed viewer uses this for display.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    max_queue_size : int
        Maximum frames buffered for the sequential consumer.  If the
        consumer falls behind, oldest frames are silently dropped (like
        a real camera that overwrites its ring buffer).
    queue_stride : int
        Only enqueue every Nth frame for the tracker (default 3).  At
        30 FPS this gives the tracker ~10 frames/s while the feed viewer
        still sees every frame for smooth display.
    """

    def __init__(
        self,
        video_path: str | Path = _DEFAULT_VIDEO,
        max_queue_size: int = 30,
        queue_stride: int = 3,
    ) -> None:
        self._video_path = Path(video_path)
        self._max_queue_size = max_queue_size
        self._queue_stride = max(1, queue_stride)

        # Producer-consumer queue with condition variable
        self._cond = threading.Condition()
        self._frame_queue: deque[np.ndarray] = deque(maxlen=max_queue_size)

        # Latest frame for the viewer (read under _cond)
        self._latest_frame: np.ndarray | None = None

        # Detected FPS (set when background thread opens the file)
        self._fps: float = 30.0

        # Thread control
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="video-source")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background reader thread and release resources."""
        self._stop.set()
        # Wake any waiters so they can exit
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- public properties ----------------------------------------------------

    @property
    def fps(self) -> float:
        """Native FPS of the video source."""
        return self._fps

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recent frame (BGR HWC numpy array), or None before first frame."""
        with self._cond:
            return self._latest_frame

    # -- capture_fn factory ---------------------------------------------------

    def make_capture_fn(self, arm_id: str = "arm0") -> object:
        """Return a ``CaptureFn`` that reads sequential frames from this source.

        Each call pops the next frame from the internal FIFO.  If the queue
        is empty, waits up to ~100 ms for the producer thread to deliver one.
        """
        source = self

        async def capture_fn(arm_id_: str) -> CapturedFrame:
            frame = source._pop_frame_blocking(timeout=0.1)
            if frame is None:
                raise RuntimeError("VideoSource: no frame available")
            return CapturedFrame(
                image=frame,
                ts_ms=int(time.monotonic() * 1000),
                arm_id=arm_id_,
            )

        def release() -> None:
            pass  # source lifecycle managed externally

        capture_fn.release = release  # type: ignore[attr-defined]
        return capture_fn

    # -- internals ------------------------------------------------------------

    def _pop_frame_blocking(self, timeout: float = 0.1) -> np.ndarray | None:
        """Pop the next frame, waiting up to *timeout* seconds if empty."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._frame_queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._cond.wait(timeout=remaining)
            return self._frame_queue.popleft()

    def _read_loop(self) -> None:
        """Background thread: read frames at native FPS, push to queue + latest."""
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        self._fps = fps
        frame_interval = 1.0 / fps

        frame_count = 0
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()

                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    ok, frame = cap.read()
                    if not ok:
                        break

                frame_count += 1

                with self._cond:
                    # Always update latest frame (feed viewer stays smooth)
                    self._latest_frame = frame
                    # Only enqueue every Nth frame for the tracker
                    if frame_count % self._queue_stride == 0:
                        self._frame_queue.append(frame)
                        self._cond.notify_all()

                # Sleep to maintain real-time rate
                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    self._stop.wait(timeout=sleep_time)
        finally:
            cap.release()
