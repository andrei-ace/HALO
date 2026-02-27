"""OpenCV camera feed viewer for debugging tracker bounding boxes.

Runs an OpenCV ``imshow`` window in a **subprocess** so that the Cocoa/AppKit
GUI backend has its own main thread (required on macOS).  A small pusher
thread in the parent process sends ``TargetInfo`` / ``PerceptionInfo``
snapshots to the child via a ``multiprocessing.Queue``.

Toggle with the **F** key in the TUI (live mode only).

Requires the ``viewer`` optional dependency::

    uv sync --extra viewer
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from pathlib import Path
from queue import Empty

import cv2
import numpy as np

from halo.contracts.enums import PerceptionFailureCode, TrackingStatus
from halo.contracts.snapshots import PerceptionInfo, TargetInfo

_DEFAULT_VIDEO = Path(__file__).parents[2] / "data" / "video.mp4"

_WINDOW_NAME = "HALO Feed"
_TARGET_FPS = 30

# Status → BGR colour
_STATUS_COLORS: dict[TrackingStatus, tuple[int, int, int]] = {
    TrackingStatus.TRACKING: (0, 200, 0),  # green
    TrackingStatus.RELOCALIZING: (0, 200, 200),  # yellow
    TrackingStatus.REACQUIRING: (0, 200, 200),  # yellow
    TrackingStatus.LOST: (0, 0, 200),  # red
    TrackingStatus.IDLE: (128, 128, 128),  # grey
}


def _status_color(status: TrackingStatus) -> tuple[int, int, int]:
    """Return BGR colour for a tracking status."""
    return _STATUS_COLORS.get(status, (128, 128, 128))


def _draw_annotations(
    frame: np.ndarray,
    target: TargetInfo | None,
    perception: PerceptionInfo | None,
) -> np.ndarray:
    """Draw tracker annotations onto *frame* (mutates in-place, returns it).

    - Bounding box rectangle (colour-coded by tracking status)
    - Handle label + confidence above the bbox
    - Tracking status bar at top-left
    - Failure code if not OK
    - Fallback crosshair at ``center_px`` when no bbox is available
    """
    if perception is not None:
        status = perception.tracking_status
        color = _status_color(status)

        # Status bar top-left
        label = status.value
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Failure code if not OK
        if perception.failure_code != PerceptionFailureCode.OK:
            fail_text = perception.failure_code.value
            cv2.putText(frame, fail_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    else:
        color = (128, 128, 128)

    if target is not None:
        bbox = target.bbox_xywh
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label above bbox
            conf_pct = int(target.confidence * 100)
            text = f"{target.handle} {conf_pct}%"
            text_y = max(y - 8, 15)
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif target.center_px is not None:
            # Fallback crosshair
            cx, cy = int(target.center_px[0]), int(target.center_px[1])
            size = 15
            cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)

            conf_pct = int(target.confidence * 100)
            text = f"{target.handle} {conf_pct}%"
            cv2.putText(frame, text, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


# ── Subprocess entry point ──────────────────────────────────────────


def _viewer_main(
    video_path: str,
    queue: mp.Queue,
    stop_event: mp.Event,
    ready_event: mp.Event,
) -> None:
    """Entry point for the viewer subprocess — OpenCV runs on the main thread."""
    import cv2  # noqa: F811 — fresh import in child process

    try:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error:
        return  # ready_event never set → parent detects failure

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    ready_event.set()

    target: TargetInfo | None = None
    perception: PerceptionInfo | None = None
    frame_delay_ms = max(1, int(1000 / _TARGET_FPS))

    try:
        while not stop_event.is_set():
            # Drain queue — keep only the latest snapshot pair
            while True:
                try:
                    target, perception = queue.get_nowait()
                except Empty:
                    break

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            _draw_annotations(frame, target, perception)
            cv2.imshow(_WINDOW_NAME, frame)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Public API ──────────────────────────────────────────────────────


class FeedViewer:
    """OpenCV camera feed window running in a child process.

    Uses ``multiprocessing.Process`` (spawn) so that the Cocoa GUI backend
    gets its own main thread — required on macOS.

    Parameters
    ----------
    store : RuntimeStateStore
        The runtime state store to read target/perception info from.
    arm_id : str
        Which arm's state to display.
    video_path : str | Path
        Video file to display (independent capture from perception service).
    """

    def __init__(
        self,
        store: object,
        arm_id: str = "arm0",
        video_path: str | Path = _DEFAULT_VIDEO,
    ) -> None:
        self._store = store
        self._arm_id = arm_id
        self._video_path = Path(video_path)
        self._process: mp.Process | None = None
        self._queue: mp.Queue | None = None
        self._stop_event: mp.Event | None = None
        self._pusher_stop = threading.Event()
        self._pusher_thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the viewer subprocess. Returns False if OpenCV GUI is unavailable."""
        if self._process is not None and self._process.is_alive():
            return True

        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        ready_event = ctx.Event()
        self._queue = ctx.Queue(maxsize=4)

        self._process = ctx.Process(
            target=_viewer_main,
            args=(str(self._video_path), self._queue, self._stop_event, ready_event),
            daemon=True,
        )
        self._process.start()

        # Wait for the child to confirm window creation
        if not ready_event.wait(timeout=5.0):
            self._process.terminate()
            self._process.join(timeout=2)
            self._process = None
            return False

        if not self._process.is_alive():
            self._process = None
            return False

        # Start a pusher thread that sends state snapshots to the child
        self._pusher_stop.clear()
        self._pusher_thread = threading.Thread(target=self._push_state, daemon=True, name="feed-pusher")
        self._pusher_thread.start()
        return True

    def stop(self) -> None:
        """Signal the child process to stop, join everything, clean up."""
        self._pusher_stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        if self._pusher_thread is not None:
            self._pusher_thread.join(timeout=2.0)
            self._pusher_thread = None
        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _push_state(self) -> None:
        """Periodically send the latest target/perception to the child process."""
        while not self._pusher_stop.is_set():
            if self._process is None or not self._process.is_alive():
                break

            target: TargetInfo | None = self._store._target.get(self._arm_id)  # type: ignore[union-attr]
            perception: PerceptionInfo | None = self._store._perception.get(self._arm_id)  # type: ignore[union-attr]

            # Drain stale items then push the latest
            if self._queue is not None:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        break
                try:
                    self._queue.put_nowait((target, perception))
                except Exception:
                    pass

            self._pusher_stop.wait(timeout=1.0 / _TARGET_FPS)
