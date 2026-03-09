"""Remote cognitive backend — thin HTTP client to remote GCP cognitive service."""

from __future__ import annotations

import dataclasses
import json
import logging
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


def _build_id_token_credentials(audience: str, sa_key_file: str | None = None, sa_email: str | None = None):
    """Build google-auth ID-token credentials for *audience*.

    Supports three credential sources (tried in order):

    1. **Explicit SA key file** — ``sa_key_file`` path to a JSON key.
    2. **SA / metadata server** — ``fetch_id_token_credentials`` (works for
       ``GOOGLE_APPLICATION_CREDENTIALS`` pointing at a SA key, or on GCE/Cloud Run).
    3. **User ADC + SA impersonation** — ``sa_email`` is required so user
       credentials can impersonate the invoker SA to mint ID tokens.
       Pass ``--sa-email`` on the CLI or set it in ``RemoteCloudConfig``.

    The returned object supports ``refresh()`` and exposes ``.token``.
    """
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    creds = None

    # 1. Explicit key file
    if sa_key_file:
        from google.oauth2 import service_account

        creds = service_account.IDTokenCredentials.from_service_account_file(sa_key_file, target_audience=audience)

    # 2. SA key via env / metadata server
    if creds is None:
        try:
            from google.oauth2 import id_token

            creds = id_token.fetch_id_token_credentials(audience, request=AuthRequest())
            creds.refresh(AuthRequest())
        except (google.auth.exceptions.DefaultCredentialsError, ValueError):
            creds = None

    # 3. User ADC → impersonate SA → mint ID token
    if creds is None:
        source_creds, _ = google.auth.default()
        from google.auth import impersonated_credentials

        # If ADC is already a SA (e.g. impersonated via gcloud), use it directly
        source_sa = getattr(source_creds, "service_account_email", None)
        if source_sa:
            creds = impersonated_credentials.IDTokenCredentials(
                target_credentials=source_creds,
                target_audience=audience,
            )
        elif sa_email:
            # User credentials — impersonate the invoker SA to get an ID token
            source_creds.refresh(AuthRequest())
            impersonated = impersonated_credentials.Credentials(
                source_credentials=source_creds,
                target_principal=sa_email,
                target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds = impersonated_credentials.IDTokenCredentials(
                target_credentials=impersonated,
                target_audience=audience,
                include_email=True,
            )
        else:
            msg = (
                "User ADC cannot mint ID tokens directly. Provide one of:\n"
                "  --sa-email <invoker-sa@project.iam.gserviceaccount.com>  (impersonate invoker SA)\n"
                "  --sa-key-file <path>  (service account key JSON)\n"
                "  gcloud auth application-default login --impersonate-service-account=<SA_EMAIL>"
            )
            raise google.auth.exceptions.DefaultCredentialsError(msg)

    # Pre-refresh so the first request doesn't block
    try:
        creds.refresh(AuthRequest())
    except google.auth.exceptions.DefaultCredentialsError:
        logger.warning("Could not refresh ID token credentials — requests may fail")
    return creds


def make_id_token_fn(audience: str, sa_key_file: str | None = None, sa_email: str | None = None) -> Callable[[], str]:
    """Return a ``() -> str`` callable that yields a fresh GCP identity token.

    Tokens are cached by the underlying google-auth library and transparently
    refreshed when expired.  Suitable for passing as *auth_token_fn* to
    ``LiveAgentClient``.
    """
    creds = _build_id_token_credentials(audience, sa_key_file, sa_email=sa_email)

    def _get_token() -> str:
        from google.auth.transport.requests import Request as AuthRequest

        if not creds.valid:
            creds.refresh(AuthRequest())
        return creds.token

    return _get_token


class _CloudRunAuth(httpx.Auth):
    """Attach GCP identity tokens for Cloud Run IAM auth.

    Uses ``google.oauth2.id_token`` to fetch tokens scoped to the Cloud Run
    service URL.  Tokens are cached and auto-refreshed by the underlying
    google-auth library.  When *sa_key_file* is provided, explicit service
    account credentials are used; otherwise Application Default Credentials
    (ADC) are used.
    """

    def __init__(self, audience: str, sa_key_file: str | None = None, sa_email: str | None = None) -> None:
        self._audience = audience
        self._credentials = _build_id_token_credentials(audience, sa_key_file, sa_email=sa_email)

    def auth_flow(self, request):
        from google.auth.transport.requests import Request as AuthRequest

        if not self._credentials.valid:
            self._credentials.refresh(AuthRequest())

        request.headers["Authorization"] = f"Bearer {self._credentials.token}"
        yield request


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
        # Only use IAM auth for https:// URLs (Cloud Run); skip for local dev (http://)
        _needs_iam = cfg.use_iam_auth and cfg.service_url.startswith("https://")
        auth = (
            _CloudRunAuth(audience=cfg.service_url, sa_key_file=cfg.sa_key_file, sa_email=cfg.sa_email)
            if _needs_iam
            else None
        )
        self._client = httpx.AsyncClient(
            base_url=cfg.service_url,
            timeout=httpx.Timeout(cfg.request_timeout_s),
            auth=auth,
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
        vlm_timeout = httpx.Timeout(self._config.vlm_timeout_s)
        resp = await self._client.post(
            "/vlm/scene",
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            data={"metadata": metadata},
            timeout=vlm_timeout,
        )
        resp.raise_for_status()
        resp_data = resp.json()
        vlm_token_usage = resp_data.get("token_usage") or {}
        server_vlm_ms = resp_data.get("vlm_ms")
        scene = dataclasses.replace(vlm_scene_from_dict(resp_data), token_usage=vlm_token_usage)

        total_ms = int((time.monotonic() - t0) * 1000)
        if server_vlm_ms is not None:
            logger.info(
                "vlm_scene arm_id=%s total_ms=%d server_vlm_ms=%d overhead_ms=%d",
                arm_id,
                total_ms,
                server_vlm_ms,
                total_ms - server_vlm_ms,
            )

        if self._run_logger is not None:
            det_dicts = [{"handle": d.handle, "bbox": d.bbox} for d in scene.detections]
            self._run_logger.log_vlm_inference(
                arm_id=arm_id,
                target_handle=target_handle or "",
                model="remote",
                raw_response={},
                target_info=None,
                inference_ms=total_ms,
                image=image,
                detections=det_dicts,
                token_usage=vlm_token_usage,
                server_vlm_ms=server_vlm_ms,
            )

        return scene

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=10.0)
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
