from __future__ import annotations

import asyncio
import json
from pathlib import Path

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.vlm_parser import VlmScene, parse_vlm_response

_MOCK_DIR = Path(__file__).parents[3] / "docs" / "data" / "mock"


def make_mock_observe_fn(mock_dir: Path = _MOCK_DIR):
    """
    Return an observe_fn backed by docs/data/mock/observe_fn_result.json.

    The JSON is loaded once at call time. The function returns the stored
    TargetInfo when the requested handle matches, otherwise None (simulating
    a momentary tracker miss).
    """
    data = json.loads((mock_dir / "observe_fn_result.json").read_text())
    stored = TargetInfo(
        handle=data["handle"],
        hint_valid=data["hint_valid"],
        confidence=data["confidence"],
        obs_age_ms=data["obs_age_ms"],
        time_skew_ms=data["time_skew_ms"],
        delta_xyz_ee=tuple(data["delta_xyz_ee"]),
        distance_m=data["distance_m"],
    )

    async def observe_fn(arm_id: str, target_handle: str) -> TargetInfo | None:
        return stored if target_handle == stored.handle else None

    return observe_fn


def make_mock_vlm_fn(mock_dir: Path = _MOCK_DIR):
    """
    Return a vlm_fn backed by docs/data/mock/vlm_response.json.

    Simulates VLM latency using the response's latency_ms field.
    Returns a VlmScene with the full scene description and detections.
    """
    raw = json.loads((mock_dir / "vlm_response.json").read_text())
    latency_s = raw.get("latency_ms", 0) / 1000.0
    scene = parse_vlm_response(raw)

    async def vlm_fn(arm_id: str) -> VlmScene:
        await asyncio.sleep(latency_s)
        return scene

    return vlm_fn
