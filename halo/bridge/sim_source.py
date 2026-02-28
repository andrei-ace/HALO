"""Video source backed by ZMQ telemetry from the MuJoCo sim server.

Drop-in replacement for ``MuJocoVideoSource`` — same public interface
(``start()``, ``stop()``, ``make_capture_fn()``, ``latest_frame``,
``latest_qpos``, ``latest_qvel``) but backed by Ch1 SUB telemetry
from ``SimClient`` instead of a subprocess + pipe.

Usage::

    from halo.bridge.config import SimBridgeConfig
    from halo.bridge.sim_source import SimSource

    source = SimSource()
    source.start()
    frame = source.latest_frame     # BGR HWC numpy array
    qpos = source.latest_qpos       # (13,) float64
    capture_fn = source.make_capture_fn("arm0")
    source.stop()
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from halo.bridge.config import SimBridgeConfig
from halo.bridge.sim_client import SimClient
from halo.services.target_perception_service.frame_buffer import CapturedFrame


class SimSource:
    """SO-101 scene camera via ZMQ telemetry from SimServer.

    Wraps a ``SimClient`` to provide the same interface as ``MuJocoVideoSource``.

    Parameters
    ----------
    config : SimBridgeConfig | None
        Bridge config. Defaults to ``SimBridgeConfig()``.
    max_queue_size : int
        Maximum frames buffered for sequential consumer (``make_capture_fn``).
    managed : bool | None
        If True, spawn the sim server. If None, uses ``config.managed``.
    """

    def __init__(
        self,
        config: SimBridgeConfig | None = None,
        max_queue_size: int = 30,
        managed: bool | None = None,
    ) -> None:
        self._config = config or SimBridgeConfig()
        if managed is not None:
            self._config.managed = managed

        self._client = SimClient(self._config)
        self._max_queue_size = max_queue_size

        self._cond = threading.Condition()
        self._frame_queue: deque[np.ndarray] = deque(maxlen=max_queue_size)
        self._latest_frame: np.ndarray | None = None
        self._latest_qpos: np.ndarray | None = None
        self._latest_qvel: np.ndarray | None = None

        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()

    def start(self, timeout: float = 60.0) -> None:
        """Connect to the sim server and start receiving telemetry.

        Blocks until the first frame arrives.

        Raises:
            TimeoutError: If no telemetry within timeout.
        """
        self._client.start(timeout=timeout)

        # Start polling thread to convert telemetry → frame queue
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_telemetry, daemon=True, name="sim-source-poll")
        self._poll_thread.start()

        # Wait for first frame
        deadline = time.monotonic() + 5.0
        with self._cond:
            while self._latest_frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

    def stop(self) -> None:
        """Disconnect from sim server and stop polling."""
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        self._client.stop()

    @property
    def client(self) -> SimClient:
        """Underlying SimClient for direct command access."""
        return self._client

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recent frame (BGR HWC numpy array), or None before first frame."""
        with self._cond:
            return self._latest_frame

    @property
    def latest_qpos(self) -> np.ndarray | None:
        """Most recent joint positions from sim (13,), or None."""
        with self._cond:
            return self._latest_qpos

    @property
    def latest_qvel(self) -> np.ndarray | None:
        """Most recent joint velocities from sim (12,), or None."""
        with self._cond:
            return self._latest_qvel

    def make_capture_fn(self, arm_id: str = "arm0") -> object:
        """Return a CaptureFn that reads sequential frames from telemetry."""
        source = self

        async def capture_fn(arm_id_: str) -> CapturedFrame:
            frame = source._pop_frame(timeout=0.5)
            if frame is None:
                raise RuntimeError("SimSource: no frame available")
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
                if remaining <= 0 or self._poll_stop.is_set():
                    return None
                self._cond.wait(timeout=remaining)
            return self._frame_queue.popleft()

    def _poll_telemetry(self) -> None:
        """Poll SimClient for new telemetry and update frame queue."""
        last_ts = -1
        while not self._poll_stop.is_set():
            telemetry = self._client.latest_telemetry
            if telemetry is not None and telemetry.get("ts_ms", -1) != last_ts:
                last_ts = telemetry["ts_ms"]
                # Convert RGB → BGR for consistency with MuJocoVideoSource
                rgb_scene = telemetry["rgb_scene"]
                bgr = cv2.cvtColor(rgb_scene, cv2.COLOR_RGB2BGR)

                with self._cond:
                    self._latest_frame = bgr
                    self._latest_qpos = telemetry["qpos"]
                    self._latest_qvel = telemetry["qvel"]
                    self._frame_queue.append(bgr)
                    self._cond.notify_all()
            else:
                time.sleep(0.01)  # 100 Hz polling
