"""Configuration for cognitive backends (local / cloud)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BackendType(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True)
class LocalConfig:
    base_url: str = "http://localhost:11434"
    planner_model: str = "gpt-oss:20b"
    vlm_model: str = "qwen2.5vl:3b"


@dataclass(frozen=True)
class CloudConfig:
    planner_model: str = "gemini-2.5-flash"
    vlm_model: str = "gemini-2.5-flash"
    api_key: str | None = None  # reads GOOGLE_API_KEY env if None


@dataclass(frozen=True)
class CognitiveConfig:
    active: BackendType = BackendType.LOCAL
    local: LocalConfig = field(default_factory=LocalConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    enable_failover: bool = False
    health_check_interval_s: float = 10.0
