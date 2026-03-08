"""Configuration for cognitive backends (local / cloud)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BackendType(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class BackendReadiness(StrEnum):
    COLD = "cold"
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
    """Config for Gemini live-session features and cloud model defaults."""

    planner_model: str = "gemini-3.1-flash-lite-preview"
    vlm_model: str = "gemini-3.1-flash-lite-preview"
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
    planner_model: str = "gemini-3.1-flash-lite-preview"
    vlm_model: str = "gemini-3.1-flash-lite-preview"


@dataclass(frozen=True)
class CompactionConfig:
    """Configuration for ADK event compaction on cloud backend."""

    enabled: bool = True
    compaction_interval: int = 20  # events between compaction runs
    overlap_size: int = 4  # recent events kept uncompacted


@dataclass(frozen=True)
class LiveAgentConfig:
    """Configuration for the Live Agent conversational layer."""

    enabled: bool = False
    voice_name: str = "Kore"
    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"


@dataclass(frozen=True)
class CognitiveConfig:
    active: BackendType = BackendType.LOCAL
    local: LocalConfig = field(default_factory=LocalConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    enable_failover: bool = False
    health_check_interval_s: float = 5.0
    startup_cloud_wait_s: float = 10.0  # max seconds to wait for cloud at startup
