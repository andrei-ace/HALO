"""
Integration test configuration.

- Enables LangChain debug mode so every LLM prompt/response is printed to
  stdout during the test run (use `pytest -s` to see it, already set in the
  Makefile target).
- Health-checks Ollama at session start; skips all @pytest.mark.integration
  tests if the server is unreachable or the required model is not loaded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest
from langchain_core.globals import set_debug

set_debug(True)

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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: LLM integration tests — require a running Ollama instance",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_reason = _ollama_skip_reason()
    if skip_reason is None:
        return
    skip = pytest.mark.skip(reason=skip_reason)
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
