"""Cloud (Gemini) cognitive backend — wraps PlannerAgent + Gemini VLM."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from halo.cognitive.config import BackendType, CloudConfig
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.agent import PlannerAgent
from halo.services.target_perception_service.gemini_vlm_fn import make_gemini_vlm_fn
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


class CloudCognitiveBackend:
    """Brain + eyes backed by Gemini cloud API."""

    def __init__(
        self,
        config: CloudConfig | None = None,
        prompts_dir: Path | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        cfg = config or CloudConfig()
        self._config = cfg
        self._api_key = cfg.api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._agent = PlannerAgent(
            model_name=cfg.planner_model,
            base_url="",
            prompts_dir=prompts_dir or _DEFAULT_PROMPTS_DIR,
            backend="cloud",
        )
        self._vlm_fn = make_gemini_vlm_fn(
            model=cfg.vlm_model,
            api_key=self._api_key,
            run_logger=run_logger,
        )

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    async def decide(
        self,
        snap: PlannerSnapshot,
        operator_cmd: str | None = None,
    ) -> list[CommandEnvelope]:
        return await self._agent.decide(snap, operator_cmd=operator_cmd)

    async def vlm_scene(
        self,
        arm_id: str,
        image: object,
        known_handles: list[str] | None = None,
        target_handle: str | None = None,
    ) -> VlmScene:
        return await self._vlm_fn(arm_id, image, known_handles, target_handle=target_handle)

    async def health_check(self) -> bool:
        """Verify Gemini API is reachable by fetching model info."""
        try:
            return await asyncio.to_thread(self._ping_gemini)
        except Exception:
            return False

    def _ping_gemini(self) -> bool:
        try:
            from google import genai

            client = genai.Client(api_key=self._api_key)
            model_info = client.models.get(model=self._config.planner_model)
            return model_info is not None
        except Exception:
            return False

    @property
    def last_reasoning(self) -> str:
        return self._agent.last_reasoning

    def reset_loop_state(self) -> None:
        self._agent.reset_loop_state()
