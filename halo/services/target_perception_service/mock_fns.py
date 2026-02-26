from __future__ import annotations

import dataclasses
import time

from halo.contracts.snapshots import TargetInfo
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.vlm_parser import VlmDetection


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
