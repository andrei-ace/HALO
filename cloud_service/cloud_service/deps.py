"""FastAPI lifespan and dependency injection for the cloud cognitive service."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, Header, HTTPException

from cloud_service.config import ServiceConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Singleton state populated during lifespan
_session_mgr = None
_config: ServiceConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _session_mgr, _config  # noqa: PLW0603

    _config = ServiceConfig.from_env()
    if not _config.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Required for cloud service. Get a key at https://aistudio.google.com/apikey"
        )
    logger.info(
        "Starting cognitive service (planner=%s, vlm=%s)",
        _config.planner_model,
        _config.vlm_model,
    )

    from halo.services.target_perception_service.gemini_vlm_fn import make_gemini_vlm_fn

    from cloud_service.session_manager import SessionManager

    def _vlm_factory():
        return make_gemini_vlm_fn(
            model=_config.vlm_model,
            api_key=_config.google_api_key or None,
        )

    firestore_store = None
    if _config.firestore_enabled:
        from cloud_service.firestore_store import FirestoreSessionStore

        firestore_store = FirestoreSessionStore(
            collection=_config.firestore_collection,
            ttl_hours=_config.firestore_ttl_hours,
        )
        logger.info("Firestore session persistence enabled (collection=%s)", _config.firestore_collection)
    else:
        logger.info("Firestore session persistence disabled (in-memory only)")

    _session_mgr = SessionManager(
        model_name=_config.planner_model,
        prompts_dir=_config.prompts_dir,
        vlm_fn_factory=_vlm_factory,
        compaction_interval=_config.compaction_interval,
        compaction_overlap=_config.compaction_overlap,
        firestore_store=firestore_store,
    )

    logger.info("Cognitive service ready")
    yield

    _session_mgr = None
    _config = None


def get_session_manager():

    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _session_mgr


def get_vlm_fn():
    mgr = get_session_manager()
    return mgr.vlm_fn


def get_config() -> ServiceConfig:
    if _config is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _config


async def verify_api_key(
    authorization: str | None = Header(None),
    config: ServiceConfig = Depends(get_config),
) -> None:
    """Validate the Authorization: Bearer <key> header."""
    if not config.cloud_api_key:
        return  # no key configured — allow all requests
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != config.cloud_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
