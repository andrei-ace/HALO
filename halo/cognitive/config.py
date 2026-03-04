"""Configuration for cognitive backends (local / cloud)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BackendType(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class BackendReadiness(StrEnum):
    COLD = "cold"
    WARMING = "warming"
    READY = "ready"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass(frozen=True)
class LocalConfig:
    base_url: str = "http://localhost:11434"
    planner_model: str = "gpt-oss:20b"
    vlm_model: str = "qwen2.5vl:3b"


@dataclass(frozen=True)
class CloudConfig:
    """Config for direct Gemini Live API backend (planner + VLM)."""

    planner_model: str = "gemini-2.5-flash"
    vlm_model: str = "gemini-2.5-flash"
    audio_enabled: bool = True
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    voice_name: str = "Kore"
    session_resumption: bool = True
    context_compression: bool = True
    response_modalities: tuple[str, ...] = ("AUDIO",)
    enable_transcription: bool = True


@dataclass(frozen=True)
class RemoteCloudConfig:
    """Config for remote HTTP client to Cloud Run cognitive service."""

    service_url: str = ""  # e.g. "https://halo-cognitive-xxx-uc.a.run.app"
    api_key: str | None = None  # reads HALO_CLOUD_API_KEY env if None
    request_timeout_s: float = 30.0
    planner_model: str = "gemini-2.5-flash"
    vlm_model: str = "gemini-2.5-flash"


@dataclass(frozen=True)
class CognitiveConfig:
    active: BackendType = BackendType.LOCAL
    local: LocalConfig = field(default_factory=LocalConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    enable_failover: bool = False
    health_check_interval_s: float = 5.0
    startup_cloud_wait_s: float = 10.0  # max seconds to wait for cloud at startup
