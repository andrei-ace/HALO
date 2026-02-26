from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene, parse_vlm_response

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


def make_mock_capture_fn():
    """Return a capture_fn that produces synthetic ``CapturedFrame`` instances.

    Each call returns a new frame with a monotonically increasing counter
    embedded in the opaque ``image`` field (as a string, e.g. ``"frame_1"``).
    """
    counter = 0

    async def capture_fn(arm_id: str) -> CapturedFrame:
        nonlocal counter
        counter += 1
        return CapturedFrame(
            image=f"frame_{counter}",
            ts_ms=int(time.monotonic() * 1000),
            arm_id=arm_id,
        )

    return capture_fn


def make_mock_tracker_factory_fn(
    init_hint: TargetInfo | None = None,
    update_hint: TargetInfo | None = None,
):
    """Return a ``TrackerFactoryFn`` that produces predictable ``TargetInfo``.

    *init_hint*: returned by the tracker initialisation step.  Defaults to a
    reasonable hint with ``distance_m=0.15``.

    *update_hint*: returned by each subsequent ``update_fn`` call.  Defaults
    to the same value as *init_hint*.
    """
    if init_hint is None:
        init_hint = TargetInfo(
            handle="",
            hint_valid=True,
            confidence=0.9,
            obs_age_ms=10,
            time_skew_ms=0,
            delta_xyz_ee=(0.02, -0.01, -0.15),
            distance_m=0.15,
        )

    async def factory(frame: CapturedFrame, detection: VlmDetection) -> tuple[TargetInfo, object]:
        seed = dataclasses.replace(init_hint, handle=detection.handle)
        effective_update = update_hint if update_hint is not None else seed

        async def update(f: CapturedFrame) -> TargetInfo | None:
            return dataclasses.replace(effective_update, handle=detection.handle)

        return seed, update

    return factory
