"""Remote cognitive backend — thin HTTP client to remote GCP cognitive service."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable

import httpx
import numpy as np

from halo.cognitive.compactor import CompactionResult, MessageRecord
from halo.cognitive.config import BackendType, RemoteCloudConfig
from halo.contracts.commands import CommandEnvelope
from halo.contracts.serde import (
    command_envelope_from_dict,
    message_record_from_dict,
    snapshot_to_dict,
    vlm_scene_from_dict,
)
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

logger = logging.getLogger(__name__)


def _encode_jpeg(image: object, quality: int = 85) -> bytes:
    """Encode a numpy BGR image (or raw bytes passthrough) to JPEG bytes."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, np.ndarray):
        import cv2

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            msg = "JPEG encoding failed"
            raise RuntimeError(msg)
        return buf.tobytes()
    msg = f"Unsupported image type: {type(image)}"
    raise TypeError(msg)


class RemoteCognitiveBackend:
    """Brain + eyes backed by remote GCP cognitive service (HTTP client).

    Uses a ``last_msg_id`` sync protocol instead of warm-up/nonce:
    each ``decide()`` sends the UUID of the last conversation message.
    The server detects sync state inline — if out of sync, returns
    ``status: "need_history"`` and the client resends full msg_history.
    """

    def __init__(
        self,
        config: RemoteCloudConfig | None = None,
        arm_id: str = "arm0",
        run_logger: RunLogger | None = None,
    ) -> None:
        cfg = config or RemoteCloudConfig()
        self._config = cfg
        self._arm_id = arm_id
        self._run_logger = run_logger
        api_key = cfg.api_key or os.environ.get("HALO_CLOUD_API_KEY", "")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=cfg.service_url,
            timeout=httpx.Timeout(cfg.request_timeout_s),
            headers=headers,
        )
        self._last_reasoning = ""
        self._last_token_usage: dict[str, int] = {}
        self._session_id: str = uuid.uuid4().hex
        self._on_compaction: Callable[[CompactionResult], Awaitable[None]] | None = None

        # last_msg_id sync protocol
        self._last_msg_id: str | None = None
        self._msg_history: list[dict] | None = None
        self._pending_handoff: str | None = None

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        body: dict = {
            "snapshot": snapshot_to_dict(snap),
            "operator_cmd": operator_cmd,
            "session_id": self._session_id,
            "last_msg_id": self._last_msg_id,
        }
        if self._pending_handoff:
            body["handoff_context"] = self._pending_handoff
        if epoch is not None:
            body["epoch"] = epoch

        resp = await self._client.post("/decide", json=body)
        resp.raise_for_status()
        data = resp.json()

        # Handle need_history: resend with full msg_history
        if data.get("status") == "need_history":
            logger.info("Cloud needs history resync — resending msg_history")
            body["msg_history"] = self._msg_history or []
            resp = await self._client.post("/decide", json=body)
            resp.raise_for_status()
            data = resp.json()

        self._last_reasoning = data.get("reasoning", "")
        self._last_token_usage = data.get("token_usage") or {}
        self._pending_handoff = None  # consumed by the server

        # Update local sync state from response
        resp_history = data.get("msg_history")
        if resp_history is not None:
            self._msg_history = resp_history
            if resp_history:
                self._last_msg_id = resp_history[-1].get("msg_id")
            else:
                self._last_msg_id = None

        # Notify Switchboard of compaction
        if data.get("compacted") and self._on_compaction is not None:
            c = data.get("compaction", {})
            result = CompactionResult(
                summary=c.get("summary", ""),
                up_to_msg_id=c.get("up_to_msg_id", ""),
                compacted_count=c.get("compacted_count", 0),
                retained_count=c.get("retained_count", 0),
                ts_ms=int(time.time() * 1000),
            )
            try:
                await self._on_compaction(result)
            except Exception:
                logger.debug("Compaction callback failed", exc_info=True)

        return [command_envelope_from_dict(c) for c in data["commands"]]

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        t0 = time.monotonic()
        jpeg_bytes = _encode_jpeg(image)
        metadata = json.dumps(
            {
                "arm_id": arm_id,
                "known_handles": known_handles or [],
                "target_handle": target_handle,
                "session_id": self._session_id,
            }
        )
        resp = await self._client.post(
            "/vlm/scene",
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            data={"metadata": metadata},
        )
        resp.raise_for_status()
        resp_data = resp.json()
        vlm_token_usage = resp_data.get("token_usage") or {}
        scene = dataclasses.replace(vlm_scene_from_dict(resp_data), token_usage=vlm_token_usage)

        if self._run_logger is not None:
            det_dicts = [{"handle": d.handle, "bbox": d.bbox} for d in scene.detections]
            self._run_logger.log_vlm_inference(
                arm_id=arm_id,
                target_handle=target_handle or "",
                model="remote",
                raw_response={},
                target_info=None,
                inference_ms=int((time.monotonic() - t0) * 1000),
                image=image,
                detections=det_dicts,
                token_usage=vlm_token_usage,
            )

        return scene

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def set_on_compaction(self, callback: Callable[[CompactionResult], Awaitable[None]] | None) -> None:
        self._on_compaction = callback

    @property
    def model_name(self) -> str:
        return "remote"

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    @property
    def msg_history_records(self) -> list[MessageRecord]:
        """Convert internal msg_history dicts to MessageRecord list for cross-backend mirroring."""
        if not self._msg_history:
            return []
        return [message_record_from_dict(d) for d in self._msg_history]

    @property
    def last_token_usage(self) -> dict[str, int]:
        return self._last_token_usage

    def reset_loop_state(self) -> None:
        """Clear reasoning only; session state (last_msg_id, msg_history) survives switches."""
        self._last_reasoning = ""

    def reset_session(self, handoff_context: str | None = None) -> None:
        """Prepare for the next cloud decide() after a backend switch.

        Called by Switchboard when this backend becomes the *new* active after
        a switch.  Preserves ``_last_msg_id`` and ``_msg_history`` so the
        sync protocol can detect server-side session loss and resend history
        (``need_history`` flow) instead of silently creating a blank session.

        If *handoff_context* is provided, it is sent with the first ``decide()``
        so the cloud agent starts with prior context.
        """
        self._pending_handoff = handoff_context

    async def aclose(self) -> None:
        await self._client.aclose()
