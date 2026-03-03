"""
Integration test configuration.

- Health-checks Ollama at session start; skips all @pytest.mark.integration
  tests if the server is unreachable or the required model is not loaded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

OLLAMA_URL = os.getenv("HALO_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("HALO_MODEL_NAME", "gpt-oss:20b")


def _ollama_skip_reason() -> str | None:
    """Return a human-readable skip reason if Ollama is not ready, else None."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return f"Ollama unreachable at {OLLAMA_URL}: {exc}"

    available = [m.get("name", "") for m in data.get("models", [])]
    model_lower = MODEL_NAME.lower()
    if not any(model_lower in name.lower() for name in available):
        return (
            f"Model '{MODEL_NAME}' not found in Ollama "
            f"(available: {available or ['<none>']}). "
            f"Run: ollama pull {MODEL_NAME}"
        )
    return None


def _gemini_skip_reason() -> str | None:
    """Return a skip reason if GOOGLE_API_KEY is not set, else None."""
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not set — skipping Gemini cloud integration tests"
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: LLM integration tests — require a running Ollama instance",
    )
    config.addinivalue_line(
        "markers",
        "cloud_integration: Gemini cloud integration tests — require GOOGLE_API_KEY",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    ollama_skip = _ollama_skip_reason()
    gemini_skip = _gemini_skip_reason()
    for item in items:
        if item.get_closest_marker("integration") and ollama_skip:
            item.add_marker(pytest.mark.skip(reason=ollama_skip))
        if item.get_closest_marker("cloud_integration") and gemini_skip:
            item.add_marker(pytest.mark.skip(reason=gemini_skip))
