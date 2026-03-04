"""Server-side configuration for the cloud cognitive service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


@dataclass(frozen=True)
class ServiceConfig:
    backend: str = "cloud"  # "cloud" (Gemini) or "local" (Ollama)
    planner_model: str = "gemini-2.5-flash"
    vlm_model: str = "gemini-2.5-flash"
    google_api_key: str = ""
    cloud_api_key: str = ""  # key clients must present in Authorization header
    ollama_base_url: str = "http://localhost:11434"
    prompts_dir: Path = _DEFAULT_PROMPTS_DIR

    @classmethod
    def from_env(cls) -> ServiceConfig:
        return cls(
            backend=os.environ.get("HALO_SERVICE_BACKEND", "cloud"),
            planner_model=os.environ.get("HALO_PLANNER_MODEL", "gemini-2.5-flash"),
            vlm_model=os.environ.get("HALO_VLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            cloud_api_key=os.environ.get("HALO_CLOUD_API_KEY", ""),
            ollama_base_url=os.environ.get("HALO_OLLAMA_URL", "http://localhost:11434"),
            prompts_dir=Path(os.environ.get("HALO_PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR))),
        )
