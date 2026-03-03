"""Server-side configuration for the cloud cognitive service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


@dataclass(frozen=True)
class ServiceConfig:
    planner_model: str = "gemini-2.5-flash"
    vlm_model: str = "gemini-2.5-flash"
    google_api_key: str = ""
    cloud_api_key: str = ""  # key clients must present in Authorization header
    prompts_dir: Path = _DEFAULT_PROMPTS_DIR

    @classmethod
    def from_env(cls) -> ServiceConfig:
        return cls(
            planner_model=os.environ.get("HALO_PLANNER_MODEL", "gemini-2.5-flash"),
            vlm_model=os.environ.get("HALO_VLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            cloud_api_key=os.environ.get("HALO_CLOUD_API_KEY", ""),
            prompts_dir=Path(os.environ.get("HALO_PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR))),
        )
