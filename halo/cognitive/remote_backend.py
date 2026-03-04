"""Remote cognitive backend — thin HTTP client to remote GCP cognitive service."""

from __future__ import annotations

import json
import logging
import os

import httpx
import numpy as np

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

    def __init__(self, config: RemoteCloudConfig | None = None, arm_id: str = "arm0") -> None:
        cfg = config or RemoteCloudConfig()
        self._config = cfg
        self._arm_id = arm_id
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
        self._readiness = BackendReadiness.COLD
        self._caught_up_cursor = -1
        self._last_nonce: str | None = None

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
        epoch: int | None = None,
    ) -> list[CommandEnvelope]:
        body: dict = {"snapshot": snapshot_to_dict(snap), "operator_cmd": operator_cmd}
        if epoch is not None:
            body["epoch"] = epoch
        resp = await self._client.post("/decide", json=body)
        resp.raise_for_status()
        data = resp.json()
        self._last_reasoning = data.get("reasoning", "")
        return [command_envelope_from_dict(c) for c in data["commands"]]

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        jpeg_bytes = _encode_jpeg(image)
        metadata = json.dumps(
            {
                "arm_id": arm_id,
                "known_handles": known_handles or [],
                "target_handle": target_handle,
            }
        )
        resp = await self._client.post(
            "/vlm/scene",
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            data={"metadata": metadata},
        )
        resp.raise_for_status()
        return vlm_scene_from_dict(resp.json())

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

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    def reset_loop_state(self) -> None:
        """Clear local state only. Does NOT reset the remote session —
        warm-up already establishes fresh context on the remote side.
        Use reset_remote() for explicit operator-initiated resets."""
        self._last_reasoning = ""

    async def reset_remote(self) -> None:
        """Explicitly reset the remote session (operator-initiated only)."""
        try:
            await self._client.post("/reset")
        except Exception:
            pass

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
