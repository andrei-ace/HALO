"""OpenCV camera feed viewer for debugging tracker bounding boxes.

Runs an OpenCV ``imshow`` window in a **subprocess** so that the Cocoa/AppKit
GUI backend has its own main thread (required on macOS).  A small pusher
thread in the parent process sends frames + annotations to the child via a
``multiprocessing.Connection`` pipe.

When a ``VideoSource`` is provided, the viewer displays frames from the
shared source (same frames the tracker processes) — ensuring annotations
align perfectly.  Without a video source, falls back to playing an
independent video file (legacy/standalone mode).

Toggle with the **F** key in the TUI (live mode only).

Requires the ``viewer`` optional dependency::

    uv sync --extra viewer

Note: Uses only ``Pipe`` (no ``Queue``, ``Event``, or ``Lock``) to avoid
POSIX semaphore resource-tracker issues on Python 3.14+.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from halo.contracts.enums import PerceptionFailureCode, TrackingStatus
from halo.contracts.snapshots import PerceptionInfo, TargetInfo

_DEFAULT_VIDEO = Path(__file__).parents[2] / "data" / "video.mp4"

_WINDOW_NAME = "HALO Feed"


def _ensure_valid_stderr() -> None:
    """Ensure sys.stderr has a real fd (>= 0).

    Textual replaces stderr with a virtual stream whose ``fileno()``
    returns -1.  When ``multiprocessing`` later spawns the resource-
    tracker process it passes ``sys.stderr.fileno()`` into
    ``fds_to_keep``, and a negative fd triggers
    ``ValueError: bad value(s) in fds_to_keep``.

    Call this before any ``multiprocessing.Process.start()``.
    """
    try:
        if sys.stderr.fileno() >= 0:
            return
    except Exception:
        pass
    # Try the original stderr saved by the interpreter.
    try:
        if sys.__stderr__ is not None and sys.__stderr__.fileno() >= 0:
            sys.stderr = sys.__stderr__
            return
    except Exception:
        pass
    # Last resort: point stderr at /dev/null so the fd is valid.
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


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
        fh, fw = frame.shape[:2]
        bbox = target.bbox_xywh
        if bbox is not None:
            # Denormalise 0..1 → pixel coords
            bx = int(bbox[0] * fw)
            by = int(bbox[1] * fh)
            bw = int(bbox[2] * fw)
            bh = int(bbox[3] * fh)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

            # Label above bbox
            conf_pct = int(target.confidence * 100)
            text = f"{target.handle} {conf_pct}%"
            text_y = max(by - 8, 15)
            cv2.putText(frame, text, (bx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif target.center_px is not None:
            # Fallback crosshair — denormalise 0..1 → pixel
            cx, cy = int(target.center_px[0] * fw), int(target.center_px[1] * fh)
            size = 15
            cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)

            conf_pct = int(target.confidence * 100)
            text = f"{target.handle} {conf_pct}%"
            cv2.putText(frame, text, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


# ── Subprocess entry point ──────────────────────────────────────────


def _viewer_main(
    video_path: str | None,
    data_conn: multiprocessing.connection.Connection,
    stop_conn: multiprocessing.connection.Connection,
    ready_conn: multiprocessing.connection.Connection,
) -> None:
    """Entry point for the viewer subprocess — OpenCV runs on the main thread.

    When *video_path* is ``None`` the subprocess displays frames received from
    the parent (shared ``VideoSource`` mode).  Otherwise it opens its own
    ``cv2.VideoCapture`` on the file (legacy/standalone mode).

    All IPC uses ``Pipe`` connections only (no ``Queue``/``Event``/``Lock``)
    to avoid POSIX semaphore resource-tracker issues on Python 3.14+.
    """
    import cv2  # noqa: F811 — fresh import in child process

    from halo.tui.fsm_overlay import render_fsm_overlay

    try:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error:
        return  # ready signal never sent → parent detects failure

    cap: cv2.VideoCapture | None = None
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

    # Signal the parent that the window is ready
    ready_conn.send(True)
    ready_conn.close()

    target: TargetInfo | None = None
    perception: PerceptionInfo | None = None
    fsm_dict: dict | None = None
    frame: np.ndarray | None = None
    frame_delay_ms = max(1, int(1000 / _TARGET_FPS))

    def _stop_requested() -> bool:
        return stop_conn.poll(0)

    try:
        while not _stop_requested():
            # Drain pipe — keep only the latest message
            while data_conn.poll(0):
                try:
                    msg = data_conn.recv()
                except EOFError:
                    break
                if isinstance(msg, tuple) and len(msg) == 4:
                    new_frame, target, perception, fsm_dict = msg
                    if new_frame is not None:
                        frame = new_frame
                elif isinstance(msg, tuple) and len(msg) == 3:
                    frame, target, perception = msg
                    fsm_dict = None
                elif isinstance(msg, tuple) and len(msg) == 2:
                    target, perception = msg

            if cap is not None:
                # Legacy mode: read from own VideoCapture
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break

            if frame is None:
                cv2.waitKey(frame_delay_ms)
                continue

            display = frame.copy()
            _draw_annotations(display, target, perception)

            # Render FSM overlay below the camera frame
            fw = display.shape[1]
            fsm_canvas = render_fsm_overlay(fsm_dict, fw)
            display = np.vstack([display, fsm_canvas])

            cv2.imshow(_WINDOW_NAME, display)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
    finally:
        if cap is not None:
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
        Video file to display (legacy/standalone mode, ignored when
        *video_source* is provided).
    video_source : VideoSource | None
        Shared video source.  When provided, frames are sent through the
        pipe and the subprocess does not open its own ``VideoCapture``.
    skill_runner_svc : object | None
        SkillRunnerService instance.  When provided, the pusher thread
        calls ``get_view_model()`` each frame and the subprocess renders
        the FSM graph below the camera feed.
    """

    def __init__(
        self,
        store: object,
        arm_id: str = "arm0",
        video_path: str | Path = _DEFAULT_VIDEO,
        video_source: object | None = None,
        skill_runner_svc: object | None = None,
    ) -> None:
        self._store = store
        self._arm_id = arm_id
        self._video_path = Path(video_path)
        self._video_source = video_source
        self._skill_runner_svc = skill_runner_svc
        self._prev_skill_run_id: str | None = None
        self._prev_skill_dict: dict | None = None  # last 3 transitions of previous skill
        self._last_skill_dict: dict | None = None  # latest serialized dict (for snapshotting on change)
        self._cached_vm_key: tuple | None = None  # (run_id, current_node, outcome, n_transitions)
        self._cached_vm_dict: dict | None = None  # cached serialized output
        self._process: mp.Process | None = None
        self._data_conn: multiprocessing.connection.Connection | None = None
        self._stop_conn: multiprocessing.connection.Connection | None = None
        self._pusher_stop = threading.Event()
        self._pusher_thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the viewer subprocess. Returns False if OpenCV GUI is unavailable."""
        if self._process is not None and self._process.is_alive():
            return True

        _ensure_valid_stderr()
        ctx = mp.get_context("spawn")
        # Use only Pipe — no Queue/Event/Lock — to avoid POSIX semaphore
        # resource-tracker issues on Python 3.14+.
        data_parent, data_child = ctx.Pipe()
        stop_parent, stop_child = ctx.Pipe()
        ready_parent, ready_child = ctx.Pipe()
        self._data_conn = data_parent
        self._stop_conn = stop_parent

        video_arg = None if self._video_source is not None else str(self._video_path)
        self._process = ctx.Process(
            target=_viewer_main,
            args=(video_arg, data_child, stop_child, ready_child),
        )
        self._process.start()
        # Close child ends in the parent — only the child process uses them.
        data_child.close()
        stop_child.close()
        ready_child.close()

        # Wait for the child to confirm window creation
        if not ready_parent.poll(timeout=5.0):
            self._process.terminate()
            self._process.join(timeout=2)
            self._process = None
            ready_parent.close()
            return False
        ready_parent.close()

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
        if self._stop_conn is not None:
            try:
                self._stop_conn.send(True)
            except (BrokenPipeError, OSError):
                pass
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

    def _serialize_view_model(self) -> dict | None:
        """Call get_view_model() on the skill runner and convert to a plain dict.

        When the active skill changes, the outgoing skill's last 3
        transitions are stored in ``_prev_skill_dict`` and sent alongside
        the new skill so the subprocess can render a small recap.
        """
        if self._skill_runner_svc is None:
            return None
        try:
            vm = self._skill_runner_svc.get_view_model()  # type: ignore[union-attr]
        except Exception:
            return None
        if vm is None:
            # Invalidate cache — runner went idle
            self._cached_vm_key = None
            self._cached_vm_dict = None
            # Runner is idle — freeze last skill as prev if we haven't already
            if self._last_skill_dict is not None and self._prev_skill_run_id is not None:
                prev = dict(self._last_skill_dict)
                prev.pop("prev_skill", None)
                self._prev_skill_dict = prev
                self._last_skill_dict = None
                self._prev_skill_run_id = None
            # Return None (idle placeholder) but attach prev recap if available
            if self._prev_skill_dict is not None:
                return {"prev_skill": self._prev_skill_dict}
            return None

        run_id = vm.skill_run_id

        # Fast path: return cached dict if FSM state hasn't changed
        vm_key = (run_id, vm.current_node, vm.outcome, len(vm.transition_history))
        if vm_key == self._cached_vm_key and self._cached_vm_dict is not None:
            return self._cached_vm_dict

        current = {
            "skill_name": vm.skill_name,
            "variant": vm.variant,
            "target_handle": vm.target_handle,
            "current_node": vm.current_node,
            "outcome": vm.outcome,
            "failure_code": vm.failure_code,
            "nodes": [
                {"name": n.name, "phase_id": n.phase_id, "status": n.status, "elapsed_ms": n.elapsed_ms}
                for n in vm.nodes
            ],
            "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in vm.edges],
            "transition_history": [
                {"from_node": t.from_node, "to_node": t.to_node, "trigger": t.trigger} for t in vm.transition_history
            ],
        }

        # Track skill transitions: when run_id changes, freeze prev with full history
        if self._prev_skill_run_id is not None and run_id != self._prev_skill_run_id:
            if self._last_skill_dict is not None:
                prev = dict(self._last_skill_dict)
                prev.pop("prev_skill", None)  # strip nested chain
                self._prev_skill_dict = prev

        self._prev_skill_run_id = run_id
        self._last_skill_dict = current

        # Attach previous skill summary so subprocess can show it
        if self._prev_skill_dict is not None:
            current["prev_skill"] = self._prev_skill_dict

        self._cached_vm_key = vm_key
        self._cached_vm_dict = current
        return current

    def _push_state(self) -> None:
        """Periodically send the latest state (and frame) to the child process."""
        while not self._pusher_stop.is_set():
            if self._process is None or not self._process.is_alive():
                break

            target: TargetInfo | None = self._store._target.get(self._arm_id)  # type: ignore[union-attr]
            perception: PerceptionInfo | None = self._store._perception.get(self._arm_id)  # type: ignore[union-attr]
            fsm_dict = self._serialize_view_model()

            if self._data_conn is not None:
                try:
                    if self._video_source is not None:
                        frame = self._video_source.latest_frame  # type: ignore[union-attr]
                        if frame is not None:
                            self._data_conn.send((frame, target, perception, fsm_dict))
                        # Skip send if no frame yet (source still starting)
                    else:
                        self._data_conn.send((None, target, perception, fsm_dict))
                except (BrokenPipeError, OSError):
                    break

            self._pusher_stop.wait(timeout=1.0 / _TARGET_FPS)
