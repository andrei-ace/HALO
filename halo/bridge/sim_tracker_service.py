"""Ch4 REP handler — receives VLM/tracker queries from the sim server.

Runs as a background thread in the HALO process. The sim server sends
queries on Ch4 (REQ) when it needs VLM detection or tracker updates during
episode generation. This service binds a REP socket and dispatches queries
to the VLM and tracker functions.

Queries:
    vlm_detect(rgb_scene)        → vlm_result(scene, detections) | no_detect
    tracker_init(rgb_scene, det) → tracker_init_ok(bbox_xywh)
    tracker_update(rgb_scene)    → tracked(ok, bbox_xywh)

Usage::

    from halo.bridge.sim_tracker_service import SimTrackerService

    service = SimTrackerService(
        query_url="tcp://127.0.0.1:5563",
        vlm_fn=vlm_fn,
        tracker_factory_fn=tracker_factory_fn,
    )
    service.start()
    # ... run episodes ...
    service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import msgpack
import numpy as np
import zmq
from mujoco_sim.server.protocol import (
    QUERY_TRACKER_INIT,
    QUERY_TRACKER_UPDATE,
    QUERY_VLM_DETECT,
    RESP_NO_DETECT,
    RESP_TRACKED,
    RESP_TRACKER_INIT_OK,
    RESP_VLM_RESULT,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SimTrackerService:
    """Ch4 REP service — handles VLM/tracker queries from the sim server.

    Binds a ZMQ REP socket and dispatches incoming queries to the provided
    VLM and tracker callables. Runs its own asyncio event loop in a
    background thread.

    Parameters
    ----------
    query_url : str
        ZMQ bind URL for the REP socket (e.g. ``tcp://127.0.0.1:5563``).
    vlm_fn : callable
        Async VLM function: ``(arm_id, image, known_handles) -> VlmScene``.
    tracker_factory_fn : callable
        Async tracker factory: ``(frame, detection) -> (hint, update_fn)``.
    """

    def __init__(
        self,
        query_url: str,
        vlm_fn: Callable[..., Awaitable[Any]],
        tracker_factory_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        self._query_url = query_url
        self._vlm_fn = vlm_fn
        self._tracker_factory_fn = tracker_factory_fn

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tracker_update_fn: Callable[..., Awaitable[Any]] | None = None

    def start(self) -> None:
        """Start the handler thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sim-tracker-svc")
        self._thread.start()
        logger.info("SimTrackerService started on %s", self._query_url)

    def stop(self) -> None:
        """Stop the handler thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("SimTrackerService stopped")

    def _run(self) -> None:
        """Thread entry: create event loop, bind REP socket, dispatch queries."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        ctx = zmq.Context()
        rep = ctx.socket(zmq.REP)
        rep.bind(self._query_url)
        rep.setsockopt(zmq.RCVTIMEO, 500)  # 500ms poll interval

        logger.info("Ch4 REP (tracker service) bound to %s", self._query_url)

        try:
            while not self._stop_event.is_set():
                try:
                    raw = rep.recv()
                except zmq.Again:
                    continue

                msg = msgpack.unpackb(raw, raw=False)
                reply = loop.run_until_complete(self._dispatch(msg))
                rep.send(msgpack.packb(reply, use_bin_type=True))
        finally:
            rep.close(linger=100)
            ctx.term()
            loop.close()

    async def _dispatch(self, msg: dict) -> dict:
        """Dispatch a query to the appropriate handler."""
        query_type = msg.get("type")

        if query_type == QUERY_VLM_DETECT:
            return await self._handle_vlm_detect(msg)

        if query_type == QUERY_TRACKER_INIT:
            return await self._handle_tracker_init(msg)

        if query_type == QUERY_TRACKER_UPDATE:
            return await self._handle_tracker_update(msg)

        return {"type": "error", "message": f"Unknown query type: {query_type}"}

    async def _handle_vlm_detect(self, msg: dict) -> dict:
        """Handle vlm_detect query: run VLM on the scene frame."""
        from halo.services.target_perception_service.frame_buffer import CapturedFrame

        rgb_bytes = msg["rgb_scene"]
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).copy()
        # Reshape — assume scene resolution from shape hint or standard 480x640
        h = msg.get("height", 480)
        w = msg.get("width", 640)
        rgb = rgb.reshape(h, w, 3)

        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        logger.info("VLM detect query — calling VLM...")
        try:
            scene = await self._vlm_fn("sim", bgr, [])
            if not scene.detections:
                logger.info("VLM returned no detections")
                return {"type": RESP_NO_DETECT}

            detection = scene.detections[0]

            # Init tracker on first detection
            frame = CapturedFrame(image=bgr, ts_ms=int(time.monotonic() * 1000), arm_id="sim")
            hint, self._tracker_update_fn = await self._tracker_factory_fn(frame, detection)

            logger.info("VLM detected: %s bbox_xywh=%s", detection.handle, hint.bbox_xywh)
            return {
                "type": RESP_VLM_RESULT,
                "handle": detection.handle,
                "label": detection.label,
                "bbox_xywh": list(hint.bbox_xywh) if hint.bbox_xywh else None,
                "confidence": hint.confidence,
            }
        except Exception as exc:
            logger.warning("VLM detect error: %s", exc)
            return {"type": RESP_NO_DETECT}

    async def _handle_tracker_init(self, msg: dict) -> dict:
        """Handle tracker_init query: init tracker on a frame with a detection."""
        from halo.services.target_perception_service.frame_buffer import CapturedFrame
        from halo.services.target_perception_service.vlm_parser import VlmDetection

        rgb_bytes = msg["rgb_scene"]
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).copy()
        h = msg.get("height", 480)
        w = msg.get("width", 640)
        rgb = rgb.reshape(h, w, 3)

        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        det_data = msg["detection"]
        detection = VlmDetection(
            handle=det_data["handle"],
            label=det_data.get("label", ""),
            bbox=tuple(det_data["bbox"]),
            centroid=tuple(det_data["centroid"]),
            is_graspable=det_data.get("is_graspable", False),
        )

        frame = CapturedFrame(image=bgr, ts_ms=int(time.monotonic() * 1000), arm_id="sim")
        try:
            hint, self._tracker_update_fn = await self._tracker_factory_fn(frame, detection)
            return {
                "type": RESP_TRACKER_INIT_OK,
                "bbox_xywh": list(hint.bbox_xywh) if hint.bbox_xywh else None,
            }
        except Exception as exc:
            logger.warning("Tracker init error: %s", exc)
            return {"type": RESP_NO_DETECT}

    async def _handle_tracker_update(self, msg: dict) -> dict:
        """Handle tracker_update query: feed a frame to the active tracker."""
        from halo.services.target_perception_service.frame_buffer import CapturedFrame

        if self._tracker_update_fn is None:
            return {"type": RESP_TRACKED, "ok": False, "bbox_xywh": None}

        rgb_bytes = msg["rgb_scene"]
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).copy()
        h = msg.get("height", 480)
        w = msg.get("width", 640)
        rgb = rgb.reshape(h, w, 3)

        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        frame = CapturedFrame(image=bgr, ts_ms=int(time.monotonic() * 1000), arm_id="sim")
        try:
            hint = await self._tracker_update_fn(frame)
            if hint is None:
                return {"type": RESP_TRACKED, "ok": False, "bbox_xywh": None}
            return {
                "type": RESP_TRACKED,
                "ok": True,
                "bbox_xywh": list(hint.bbox_xywh) if hint.bbox_xywh else None,
            }
        except Exception:
            return {"type": RESP_TRACKED, "ok": False, "bbox_xywh": None}
