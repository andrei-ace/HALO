"""Backward-compat shim — use ``vlm_fn.make_vlm_fn(provider="gemini")`` directly."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from halo.services.target_perception_service.vlm_fn import (
    _extract_json,
    _resize_image,
    _to_pil,
    make_vlm_fn,
)

if TYPE_CHECKING:
    from halo.tui.run_logger import RunLogger

# Re-export for tests
__all__ = ["_extract_json", "_resize_image", "_to_pil", "make_gemini_vlm_fn"]

_DEFAULT_PROMPT = Path(__file__).parents[3] / "configs" / "perception" / "scene_analysis.md"


def make_gemini_vlm_fn(
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    prompt_path: Path = _DEFAULT_PROMPT,
    run_logger: RunLogger | None = None,
):
    return make_vlm_fn(provider="gemini", model=model, api_key=api_key, prompt_path=prompt_path, run_logger=run_logger)
