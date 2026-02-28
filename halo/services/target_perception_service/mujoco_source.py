"""Video source backed by MuJoCo/robosuite scene camera.

A **subprocess** creates the robosuite env and renders the scene camera in
a loop on its own main thread (required on macOS for OpenGL).  Frames are
sent to the parent process via a ``multiprocessing.Pipe``.  A reader thread
in the parent receives them and populates a frame queue + ``latest_frame``.

Interface is identical to ``VideoSource``: ``start()``, ``stop()``,
``make_capture_fn()``, ``latest_frame``.

Usage::

    from halo.services.target_perception_service.mujoco_source import MuJocoVideoSource

    source = MuJocoVideoSource()
    source.start()                     # blocks until first frame arrives

    capture_fn = source.make_capture_fn("arm0")
    frame = source.latest_frame

    source.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import os
import sys
import threading
import time
from collections import deque

import numpy as np

from halo.services.target_perception_service.frame_buffer import CapturedFrame

_CAMERA_NAME = "agentview"
_CAMERA_H, _CAMERA_W = 480, 640


def _suppress_robosuite_warnings() -> None:
    """Inject a fake ``macros_private`` module so robosuite skips its startup warnings.

    Must be called **before** ``import robosuite``.  Suppresses: missing
    macros_private, optional robosuite_models/mink, and controller-component
    mismatches for Panda (all harmless for our single-arm Lift env).
    """
    import types

    if "robosuite.macros_private" not in sys.modules:
        fake = types.ModuleType("robosuite.macros_private")
        fake.CONSOLE_LOGGING_LEVEL = "ERROR"  # type: ignore[attr-defined]
        sys.modules["robosuite.macros_private"] = fake


# ── Subprocess entry point ──────────────────────────────────────────────


def _renderer_main(
    fps: float,
    seed: int | None,
    frame_conn: multiprocessing.connection.Connection,
    stop_conn: multiprocessing.connection.Connection,
    ready_conn: multiprocessing.connection.Connection,
) -> None:
    """Subprocess: create robosuite env, render scene camera, send frames via pipe."""
    import cv2

    try:
        _suppress_robosuite_warnings()
        import robosuite as suite
    except ImportError:
        return  # ready never sent → parent detects failure

    try:
        controller_config = suite.load_composite_controller_config(controller="BASIC")
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=[_CAMERA_NAME],
            camera_heights=[_CAMERA_H],
            camera_widths=[_CAMERA_W],
            control_freq=20,
            horizon=500,
        )
    except Exception:
        return

    if seed is not None:
        np.random.seed(seed)
    env.reset()

    # Signal ready before sending any frames (pipe buffer is small on macOS;
    # sending the frame first can block if the parent hasn't started reading)
    try:
        ready_conn.send(True)
    except (BrokenPipeError, OSError):
        env.close()
        return
    ready_conn.close()

    frame_interval = 1.0 / fps

    try:
        while not stop_conn.poll(0):
            t0 = time.monotonic()

            rgb = env.sim.render(camera_name=_CAMERA_NAME, width=_CAMERA_W, height=_CAMERA_H)[::-1]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Extract joint state (qpos/qvel) to send alongside the frame
            qpos = env.sim.data.qpos.copy()
            qvel = env.sim.data.qvel.copy()

            try:
                frame_conn.send((bgr, qpos, qvel))
            except (BrokenPipeError, OSError):
                break

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()


# ── Parent-side source ──────────────────────────────────────────────────


def _ensure_valid_stderr() -> None:
    """Ensure sys.stderr has a real fd (Textual replaces it with fd=-1)."""
    try:
        if sys.stderr.fileno() >= 0:
            return
    except Exception:
        pass
    try:
        if sys.__stderr__ is not None and sys.__stderr__.fileno() >= 0:
            sys.stderr = sys.__stderr__
            return
    except Exception:
        pass
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


class MuJocoVideoSource:
    """Robosuite scene camera running in a subprocess, frames received via pipe.

    Parameters
    ----------
    fps : float
        Target render rate in the subprocess.
    max_queue_size : int
        Maximum frames buffered for the sequential consumer.
    seed : int | None
        Seed for env reset (reproducibility).
    """

    def __init__(
        self,
        fps: float = 10.0,
        max_queue_size: int = 30,
        seed: int | None = None,
    ) -> None:
        self._fps = fps
        self._max_queue_size = max_queue_size
        self._seed = seed

        self._cond = threading.Condition()
        self._frame_queue: deque[np.ndarray] = deque(maxlen=max_queue_size)
        self._latest_frame: np.ndarray | None = None
        self._latest_qpos: np.ndarray | None = None
        self._latest_qvel: np.ndarray | None = None

        self._process: mp.Process | None = None
        self._frame_conn: multiprocessing.connection.Connection | None = None
        self._stop_conn: multiprocessing.connection.Connection | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()

    def start(self, timeout: float = 60.0) -> None:
        """Spawn the renderer subprocess and wait for the first frame.

        Raises RuntimeError if the subprocess fails to start within *timeout*.
        """
        if self._process is not None and self._process.is_alive():
            return

        _ensure_valid_stderr()
        ctx = mp.get_context("spawn")

        # frame: child writes, parent reads
        frame_recv, frame_send = ctx.Pipe(duplex=False)
        # stop: parent writes, child reads
        stop_recv, stop_send = ctx.Pipe(duplex=False)
        # ready: child writes, parent reads
        ready_recv, ready_send = ctx.Pipe(duplex=False)

        self._frame_conn = frame_recv
        self._stop_conn = stop_send

        self._process = ctx.Process(
            target=_renderer_main,
            args=(self._fps, self._seed, frame_send, stop_recv, ready_send),
        )
        self._process.start()
        # Close child ends in the parent
        frame_send.close()
        stop_recv.close()
        ready_send.close()

        # Start reader thread first so it can drain frames from the pipe
        # (pipe buffer is small on macOS — must read to prevent subprocess blocking)
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True, name="mujoco-reader")
        self._reader_thread.start()

        # Wait for the subprocess to signal ready (env created successfully)
        if not ready_recv.poll(timeout=timeout):
            self._reader_stop.set()
            self._stop_conn.send(True)
            self._process.terminate()
            self._process.join(timeout=2)
            self._process = None
            ready_recv.close()
            raise RuntimeError(f"MuJocoVideoSource did not start within {timeout}s")
        ready_recv.close()

        if not self._process.is_alive():
            self._reader_stop.set()
            self._process = None
            raise RuntimeError("MuJocoVideoSource subprocess exited unexpectedly")

        # Wait for the first frame to arrive in the queue
        deadline = time.monotonic() + 5.0
        with self._cond:
            while self._latest_frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

    def stop(self) -> None:
        """Stop the subprocess and reader thread."""
        self._reader_stop.set()

        if self._stop_conn is not None:
            try:
                self._stop_conn.send(True)
            except (BrokenPipeError, OSError):
                pass

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None

        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

        if self._frame_conn is not None:
            try:
                self._frame_conn.close()
            except OSError:
                pass
            self._frame_conn = None
        if self._stop_conn is not None:
            try:
                self._stop_conn.close()
            except OSError:
                pass
            self._stop_conn = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recent frame (BGR HWC numpy array), or None before first frame."""
        with self._cond:
            return self._latest_frame

    @property
    def latest_qpos(self) -> np.ndarray | None:
        """Most recent joint positions from sim, or None if unavailable."""
        with self._cond:
            return self._latest_qpos

    @property
    def latest_qvel(self) -> np.ndarray | None:
        """Most recent joint velocities from sim, or None if unavailable."""
        with self._cond:
            return self._latest_qvel

    def make_capture_fn(self, arm_id: str = "arm0") -> object:
        """Return a CaptureFn that reads sequential frames from the renderer."""
        source = self

        async def capture_fn(arm_id_: str) -> CapturedFrame:
            frame = source._pop_frame(timeout=0.5)
            if frame is None:
                raise RuntimeError("MuJocoVideoSource: no frame available")
            return CapturedFrame(
                image=frame,
                ts_ms=int(time.monotonic() * 1000),
                arm_id=arm_id_,
            )

        def release() -> None:
            pass

        capture_fn.release = release  # type: ignore[attr-defined]
        return capture_fn

    def _pop_frame(self, timeout: float = 0.5) -> np.ndarray | None:
        """Pop the next frame from the queue, waiting up to *timeout* seconds."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._frame_queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._reader_stop.is_set():
                    return None
                self._cond.wait(timeout=remaining)
            return self._frame_queue.popleft()

    def _read_frames(self) -> None:
        """Reader thread: receive frames from subprocess pipe, push to queue."""
        conn = self._frame_conn
        while not self._reader_stop.is_set():
            try:
                if not conn.poll(timeout=0.1):
                    continue
                data = conn.recv()
            except (EOFError, OSError):
                break

            # Unpack (frame, qpos, qvel) tuple or plain frame for backward compat
            if isinstance(data, tuple):
                frame, qpos, qvel = data
            else:
                frame = data
                qpos = None
                qvel = None

            with self._cond:
                self._latest_frame = frame
                self._latest_qpos = qpos
                self._latest_qvel = qvel
                self._frame_queue.append(frame)
                self._cond.notify_all()
