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
    from halo.services.planner_service.agent import PlannerAgent

logger = logging.getLogger(__name__)

# Singleton state populated during lifespan
_agent: PlannerAgent | None = None
_vlm_fn = None
_config: ServiceConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _agent, _vlm_fn, _config  # noqa: PLW0603

    _config = ServiceConfig.from_env()
    logger.info("Starting cloud cognitive service (planner=%s, vlm=%s)", _config.planner_model, _config.vlm_model)

    from halo.services.planner_service.agent import PlannerAgent
    from halo.services.target_perception_service.gemini_vlm_fn import make_gemini_vlm_fn

    _agent = PlannerAgent(
        model_name=_config.planner_model,
        base_url="",
        prompts_dir=_config.prompts_dir,
        backend="cloud",
    )
    _vlm_fn = make_gemini_vlm_fn(
        model=_config.vlm_model,
        api_key=_config.google_api_key or None,
    )

    logger.info("Cloud cognitive service ready")
    yield

    _agent = None
    _vlm_fn = None
    _config = None


def get_agent() -> PlannerAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _agent


def get_vlm_fn():
    if _vlm_fn is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _vlm_fn


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
