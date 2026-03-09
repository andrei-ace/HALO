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
    use_iam_auth: bool = True  # when True, auto-fetch GCP identity tokens
    sa_key_file: str | None = None  # optional path to service account key JSON
    sa_email: str | None = None  # optional SA email for impersonation (user ADC → SA → ID token)
    request_timeout_s: float = 30.0
    vlm_timeout_s: float = 12.0  # VLM-specific timeout (shorter than planner default)
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
    startup_cloud_wait_s: float = 30.0  # max seconds to wait for cloud at startup (Cloud Run cold start)
