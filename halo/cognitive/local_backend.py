"""Local (Ollama) cognitive backend — wraps PlannerAgent + Ollama VLM."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from halo.cognitive.config import BackendType, LocalConfig
from halo.contracts.commands import CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.planner_service.agent import PlannerAgent
from halo.services.target_perception_service.ollama_vlm_fn import make_ollama_vlm_fn
from halo.services.target_perception_service.vlm_parser import VlmScene

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


class LocalCognitiveBackend:
    """Brain + eyes backed by local Ollama."""

    def __init__(
        self,
        config: LocalConfig | None = None,
        prompts_dir: Path | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        cfg = config or LocalConfig()
        self._config = cfg
        self._agent = PlannerAgent(
            model_name=cfg.planner_model,
            base_url=cfg.base_url,
            prompts_dir=prompts_dir or _DEFAULT_PROMPTS_DIR,
            backend="local",
        )
        self._vlm_fn = make_ollama_vlm_fn(
            base_url=cfg.base_url,
            model=cfg.vlm_model,
            run_logger=run_logger,
        )

    @property
    def backend_type(self) -> str:
        return BackendType.LOCAL

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
        """Ping Ollama /api/tags endpoint."""
        try:
            result = await asyncio.to_thread(self._ping_ollama)
            return result
        except Exception:
            return False

    def _ping_ollama(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._config.base_url}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
            return bool(data.get("models"))
        except Exception:
            return False

    @property
    def last_reasoning(self) -> str:
        return self._agent.last_reasoning

    def reset_loop_state(self) -> None:
        self._agent.reset_loop_state()
