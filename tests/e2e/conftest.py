"""E2E test fixtures — health-check Ollama for both planner and VLM models.

Tests auto-skip if either model is unavailable.

By default only qwen2.5vl:3b is tested. Pass ``--run-all-vlm-models`` to
also run with qwen2.5vl:7b for side-by-side comparison.
"""

from __future__ import annotations

import os

import pytest

# E2E tests require Ollama with both planner and VLM models
OLLAMA_URL = os.environ.get("HALO_OLLAMA_URL", "http://localhost:11434")
PLANNER_MODEL = os.environ.get("HALO_MODEL_NAME", "gpt-oss:20b")

# VLM models — 3b is default, 7b only with --run-all-vlm-models
VLM_DEFAULT = "qwen2.5vl:3b"
VLM_ALL = ["qwen2.5vl:3b", "qwen2.5vl:7b"]


def _resolve_ollama_model(base_url: str, model: str) -> str | None:
    """Resolve a model name to the full Ollama tag, or None if unavailable."""
    try:
        import json
        import urllib.request

        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            if model in models:
                return model
            for m in models:
                if m.startswith(model):
                    return m
            return None
    except Exception:
        return None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-all-vlm-models",
        action="store_true",
        default=False,
        help="Run e2e VLM tests with all models (3b + 7b) instead of just 3b",
    )


_resolved_planner = _resolve_ollama_model(OLLAMA_URL, PLANNER_MODEL)
_resolved_vlm_models = {m: _resolve_ollama_model(OLLAMA_URL, m) for m in VLM_ALL}
_default_vlm_available = _resolved_vlm_models.get(VLM_DEFAULT) is not None

skip_no_planner = pytest.mark.skipif(_resolved_planner is None, reason=f"Ollama model '{PLANNER_MODEL}' not available")
skip_no_vlm = pytest.mark.skipif(not _default_vlm_available, reason=f"Ollama model '{VLM_DEFAULT}' not available")
skip_no_ollama = pytest.mark.skipif(
    not (_resolved_planner and _default_vlm_available),
    reason=f"Ollama planner '{PLANNER_MODEL}' and/or VLM '{VLM_DEFAULT}' not available",
)


@pytest.fixture
def ollama_url() -> str:
    return OLLAMA_URL


@pytest.fixture
def planner_model() -> str:
    assert _resolved_planner is not None
    return _resolved_planner


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "vlm_model" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--run-all-vlm-models"):
        models = [m for m, resolved in _resolved_vlm_models.items() if resolved is not None]
    else:
        models = [VLM_DEFAULT] if _default_vlm_available else []
    metafunc.parametrize("vlm_model", models)
