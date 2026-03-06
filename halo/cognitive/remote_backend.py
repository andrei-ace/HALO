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

from halo.cognitive.compactor import CompactionResult
from halo.cognitive.config import BackendReadiness, BackendType, RemoteCloudConfig
from halo.cognitive.context_store import CognitiveState, ContextEntry
from halo.contracts.commands import CommandEnvelope
from halo.contracts.serde import (
    cognitive_state_to_dict,
    command_envelope_from_dict,
    context_entry_to_dict,
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

    NOTE: This backend uses HTTP request-response only. The Gemini Live API
    bidirectional streaming path (audio, voice commands) is only available
    via CloudCognitiveBackend running in-process. WebSocket streaming to
    Cloud Run is a future milestone.
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
        self._readiness = BackendReadiness.COLD
        self._caught_up_cursor = -1
        self._last_nonce: str | None = None
        self._session_id: str = uuid.uuid4().hex
        self._on_compaction: Callable[[CompactionResult], Awaitable[None]] | None = None

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        body: dict = {"snapshot": snapshot_to_dict(snap), "operator_cmd": operator_cmd, "session_id": self._session_id}
        if epoch is not None:
            body["epoch"] = epoch
        resp = await self._client.post("/decide", json=body)
        resp.raise_for_status()
        data = resp.json()
        self._last_reasoning = data.get("reasoning", "")
        self._last_token_usage = data.get("token_usage") or {}

        # Notify Switchboard of compaction (it publishes SESSION_COMPACTED
        # to EventBus, syncs to inactive backend, and the TUI logs it)
        if data.get("compacted") and self._on_compaction is not None:
            c = data.get("compaction", {})
            result = CompactionResult(
                summary=c.get("summary", ""),
                up_to_msg_id=c.get("up_to_msg_id", ""),
                compacted_count=c.get("compacted_count", 0),
                retained_count=c.get("retained_count", 0),
                ts_ms=int(time.monotonic() * 1000),
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
            if resp.status_code != 200:
                return False
            data = resp.json()
            nonce = data.get("nonce")
            if nonce and self._last_nonce and nonce != self._last_nonce:
                logger.warning("Cloud instance restarted (nonce changed: %s → %s)", self._last_nonce, nonce)
                self._readiness = BackendReadiness.COLD
                self._caught_up_cursor = -1
            if nonce:
                self._last_nonce = nonce
            return True
        except Exception:
            return False

    def set_on_compaction(self, callback: Callable[[CompactionResult], Awaitable[None]] | None) -> None:
        self._on_compaction = callback

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    @property
    def last_token_usage(self) -> dict[str, int]:
        return self._last_token_usage

    def reset_loop_state(self) -> None:
        """Clear local state — warm-up already establishes fresh context
        on the remote side. Reset readiness so the next failback goes through
        full COLD warm-up instead of reusing stale session history."""
        self._last_reasoning = ""
        self._readiness = BackendReadiness.COLD
        self._caught_up_cursor = -1

    # -- WarmableBackend --

    async def warm_up(
        self,
        state: CognitiveState | None,
        journal_entries: list[ContextEntry],
    ) -> bool:
        """POST CognitiveState + journal to cloud service's /warm-up endpoint."""
        try:
            body: dict = {
                "arm_id": self._arm_id,
                "session_id": self._session_id,
                "state": cognitive_state_to_dict(state) if state else None,
                "journal": [context_entry_to_dict(e) for e in journal_entries],
            }
            resp = await self._client.post("/warm-up", json=body)
            resp.raise_for_status()
            data = resp.json()
            self._readiness = BackendReadiness(data.get("readiness", "cold"))
            self._caught_up_cursor = data.get("cursor", -1)
            return self._readiness == BackendReadiness.READY
        except Exception:
            logger.exception("warm_up() failed")
            self._readiness = BackendReadiness.FAILED
            return False

    @property
    def readiness(self) -> str:
        return self._readiness

    @property
    def caught_up_cursor(self) -> int:
        return self._caught_up_cursor

    async def aclose(self) -> None:
        await self._client.aclose()
