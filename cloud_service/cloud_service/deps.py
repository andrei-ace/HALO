"""FastAPI lifespan and dependency injection for the cloud cognitive service."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import HTTPException

from cloud_service.config import ServiceConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Singleton state populated during lifespan
_session_mgr = None
_config: ServiceConfig | None = None
_live_agent_mgr = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _session_mgr, _config, _live_agent_mgr  # noqa: PLW0603

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

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

    # Live Agent manager
    if _config.live_agent_enabled:
        from cloud_service.live_agent_manager import LiveAgentManager

        _live_agent_mgr = LiveAgentManager(
            model=_config.live_agent_model,
            voice=_config.live_agent_voice,
            prompts_dir=_config.live_agent_prompts_dir,
        )
        app.state.live_agent_manager = _live_agent_mgr
        logger.info("Live Agent enabled (model=%s, voice=%s)", _config.live_agent_model, _config.live_agent_voice)
    else:
        logger.info("Live Agent disabled")

    logger.info("Cognitive service ready")
    yield

    if _live_agent_mgr is not None:
        await _live_agent_mgr.remove_all()
        _live_agent_mgr = None
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


def get_live_agent_manager():
    if _live_agent_mgr is None:
        raise HTTPException(status_code=503, detail="Live agent not enabled")
    return _live_agent_mgr
