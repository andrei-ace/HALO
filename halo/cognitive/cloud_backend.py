"""Cloud cognitive backend — thin HTTP client to remote GCP cognitive service."""

from __future__ import annotations

import json
import os

import httpx
import numpy as np

from halo.cognitive.config import BackendType, CloudConfig
from halo.contracts.commands import CommandEnvelope
from halo.contracts.serde import (
    command_envelope_from_dict,
    snapshot_to_dict,
    vlm_scene_from_dict,
)
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.vlm_parser import VlmScene


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


class CloudCognitiveBackend:
    """Brain + eyes backed by remote GCP cognitive service."""

    def __init__(self, config: CloudConfig | None = None) -> None:
        cfg = config or CloudConfig()
        self._config = cfg
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

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]:
        body = {"snapshot": snapshot_to_dict(snap), "operator_cmd": operator_cmd}
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
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning

    def reset_loop_state(self) -> None:
        # Server-side session state is ephemeral (Cloud Run cold starts).
        # Reset is a best-effort POST; failures are silently ignored.
        try:
            import asyncio

            asyncio.get_running_loop().create_task(self._reset_remote())
        except RuntimeError:
            pass

    async def _reset_remote(self) -> None:
        try:
            await self._client.post("/reset")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()
