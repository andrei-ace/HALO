"""Server-side configuration for the cloud cognitive service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PROMPTS_DIR = Path(__file__).parents[2] / "configs" / "planner"


def _resolve_firestore_enabled() -> bool:
    """Auto-detect Firestore availability.

    Enabled when:
    - HALO_FIRESTORE_ENABLED is explicitly "true"/"1"/"yes", OR
    - FIRESTORE_EMULATOR_HOST is set (local emulator)

    Disabled otherwise (safe default for local dev without creds).
    """
    explicit = os.environ.get("HALO_FIRESTORE_ENABLED")
    if explicit is not None:
        return explicit.lower() in ("true", "1", "yes")
    return bool(os.environ.get("FIRESTORE_EMULATOR_HOST"))


@dataclass(frozen=True)
class ServiceConfig:
    planner_model: str = "gemini-3.1-flash-lite-preview"
    vlm_model: str = "gemini-3.1-flash-lite-preview"
    google_api_key: str = ""
    cloud_api_key: str = ""  # key clients must present in Authorization header
    prompts_dir: Path = _DEFAULT_PROMPTS_DIR
    compaction_interval: int = 20  # invocations between compaction runs
    compaction_overlap: int = 4  # recent invocations kept uncompacted
    firestore_enabled: bool = False  # HALO_FIRESTORE_ENABLED — auto-enabled when FIRESTORE_EMULATOR_HOST is set
    firestore_collection: str = "halo_sessions"  # HALO_FIRESTORE_COLLECTION
    firestore_ttl_hours: float = 1.0  # HALO_FIRESTORE_TTL_HOURS

    @classmethod
    def from_env(cls) -> ServiceConfig:
        return cls(
            planner_model=os.environ.get("HALO_PLANNER_MODEL", "gemini-3.1-flash-lite-preview"),
            vlm_model=os.environ.get("HALO_VLM_MODEL", "gemini-3.1-flash-lite-preview"),
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            cloud_api_key=os.environ.get("HALO_CLOUD_API_KEY", ""),
            prompts_dir=Path(os.environ.get("HALO_PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR))),
            compaction_interval=int(os.environ.get("HALO_COMPACTION_INTERVAL", "20")),
            compaction_overlap=int(os.environ.get("HALO_COMPACTION_OVERLAP", "4")),
            firestore_enabled=_resolve_firestore_enabled(),
            firestore_collection=os.environ.get("HALO_FIRESTORE_COLLECTION", "halo_sessions"),
            firestore_ttl_hours=float(os.environ.get("HALO_FIRESTORE_TTL_HOURS", "1.0")),
        )
