"""Backward-compat shim — use ``vlm_fn.make_vlm_fn(provider="ollama")`` directly."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from halo.services.target_perception_service.vlm_fn import make_vlm_fn

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

__all__ = ["make_ollama_vlm_fn"]

_DEFAULT_PROMPT = Path(__file__).parents[3] / "configs" / "perception" / "scene_analysis.md"


def make_ollama_vlm_fn(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5vl",
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    return make_vlm_fn(
        provider="ollama", model=model, base_url=base_url, prompt_path=prompt_path, run_logger=run_logger
    )
