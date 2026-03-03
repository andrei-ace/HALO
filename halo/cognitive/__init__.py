"""Cognitive backend abstraction — brain (planner) + eyes (VLM) switching."""

from halo.cognitive.backend import CognitiveBackend
from halo.cognitive.config import BackendType, CloudConfig, CognitiveConfig, LocalConfig

__all__ = [
    "BackendType",
    "CloudConfig",
    "CognitiveBackend",
    "CognitiveConfig",
    "LocalConfig",
]
