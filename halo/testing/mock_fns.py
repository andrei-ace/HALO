"""Variable-latency mock factories for HALO integration tests.

Each factory wraps a simple callable with configurable async delay, enabling
realistic-but-fast integration testing without external deps (Ollama, robot).
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from halo.contracts.actions import Action, ActionChunk
from halo.contracts.commands import CommandEnvelope, CommandPayload
from halo.contracts.enums import CommandType, PhaseId
from halo.contracts.snapshots import PlannerSnapshot
from halo.services.target_perception_service.frame_buffer import CapturedFrame
from halo.services.target_perception_service.mock_fns import (
    make_mock_tracker_factory_fn as _base_tracker_factory_fn,
)
from halo.services.target_perception_service.vlm_parser import VlmDetection, VlmScene


@dataclass(frozen=True)
class LatencyProfile:
    """Configurable delay ranges (min_s, max_s) for each mock function."""

    decide_s: tuple[float, float] = (0.0, 0.0)
    vlm_s: tuple[float, float] = (0.0, 0.0)
    capture_s: tuple[float, float] = (0.0, 0.0)
    apply_s: tuple[float, float] = (0.0, 0.0)
    chunk_s: tuple[float, float] = (0.0, 0.0)
    tracker_init_s: tuple[float, float] = (0.0, 0.0)
    tracker_update_s: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def instant(cls) -> LatencyProfile:
        """Zero delay — for unit tests."""
        return cls()

    @classmethod
    def realistic(cls) -> LatencyProfile:
        """Realistic latencies — LLM 2-5s, VLM 1-3s, hardware ~ms."""
        return cls(
            decide_s=(2.0, 5.0),
            vlm_s=(1.0, 3.0),
            capture_s=(0.005, 0.015),
            apply_s=(0.0005, 0.002),
            chunk_s=(0.01, 0.05),
            tracker_init_s=(0.02, 0.05),
            tracker_update_s=(0.001, 0.005),
        )

    @classmethod
    def fast_integration(cls) -> LatencyProfile:
        """Fast but non-zero — CI-friendly integration tests."""
        return cls(
            decide_s=(0.05, 0.15),
            vlm_s=(0.03, 0.08),
            capture_s=(0.001, 0.003),
            apply_s=(0.0001, 0.0005),
            chunk_s=(0.005, 0.01),
            tracker_init_s=(0.005, 0.01),
            tracker_update_s=(0.0005, 0.001),
        )


async def _delay(range_s: tuple[float, float]) -> None:
    """Sleep for a random duration in the given range."""
    lo, hi = range_s
    if hi > 0:
        await asyncio.sleep(random.uniform(lo, hi))


# -- decide_fn -------------------------------------------------------------


def make_mock_decide_fn(
    latency: LatencyProfile = LatencyProfile.instant(),
    commands_fn: callable | None = None,
):
    """Return a ``decide_fn`` compatible with PlannerService.

    *commands_fn*: ``(snapshot) -> list[CommandEnvelope]``. Defaults to returning
    an empty list (no commands). Use a custom function to script planner behaviour.
    """

    async def decide_fn(snapshot: PlannerSnapshot) -> list[CommandEnvelope]:
        await _delay(latency.decide_s)
        if commands_fn is not None:
            return commands_fn(snapshot)
        return []

    return decide_fn


# -- vlm_fn ----------------------------------------------------------------


def make_mock_vlm_fn(
    latency: LatencyProfile = LatencyProfile.instant(),
    scene: VlmScene | None = None,
):
    """Return a ``vlm_fn`` compatible with TargetPerceptionService.

    *scene*: fixed VlmScene to return. Defaults to a scene with one ``red_cube``
    detection.
    """
    if scene is None:
        scene = VlmScene(
            scene="A red cube on a table.",
            detections=[
                VlmDetection(
                    handle="red_cube",
                    label="red cube",
                    bbox=(280.0, 200.0, 360.0, 280.0),
                    centroid=(320.0, 240.0),
                    is_graspable=True,
                ),
            ],
        )

    async def vlm_fn(arm_id: str, image: object, known_handles: list[str], target_handle: str | None = None):
        await _delay(latency.vlm_s)
        return scene

    return vlm_fn


# -- apply_fn --------------------------------------------------------------


def make_mock_apply_fn(
    latency: LatencyProfile = LatencyProfile.instant(),
    log: list | None = None,
):
    """Return an ``apply_fn`` compatible with ControlService.

    *log*: if provided, each applied (arm_id, action) is appended.
    """

    async def apply_fn(arm_id: str, action: Action) -> None:
        await _delay(latency.apply_s)
        if log is not None:
            log.append((arm_id, action))

    return apply_fn


# -- chunk_fn --------------------------------------------------------------


def make_mock_chunk_fn(
    latency: LatencyProfile = LatencyProfile.instant(),
    n_actions: int = 10,
):
    """Return a ``chunk_fn`` compatible with SkillRunnerService.

    Produces ActionChunks with *n_actions* small forward motions.
    """
    seq = 0

    async def chunk_fn(arm_id: str, phase: PhaseId, snapshot: object) -> ActionChunk | None:
        nonlocal seq
        await _delay(latency.chunk_s)
        seq += 1
        actions = tuple(Action(0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n_actions))
        return ActionChunk(
            chunk_id=f"mock-chunk-{seq}",
            arm_id=arm_id,
            phase_id=phase,
            actions=actions,
            ts_ms=int(time.monotonic() * 1000),
        )

    return chunk_fn


# -- capture_fn with latency -----------------------------------------------


def _synthetic_image() -> np.ndarray:
    """Return a small synthetic BGR numpy image (240x320x3).

    Produces a valid image that ``_image_to_b64`` in ``ollama_vlm_fn`` can
    convert to PNG — unlike the string placeholder in the base mock_fns
    which is only suitable for unit tests that never touch the real VLM.
    """
    return np.zeros((240, 320, 3), dtype=np.uint8)


def make_mock_capture_fn_with_latency(latency: LatencyProfile = LatencyProfile.instant()):
    """Return a ``capture_fn`` with configurable latency.

    Produces ``CapturedFrame`` instances whose ``image`` field is a real
    numpy BGR array so the frame is compatible with both mock and real
    VLM functions (the real Ollama adapter calls ``_image_to_b64``).
    """
    counter = 0

    async def capture_fn(arm_id: str) -> CapturedFrame:
        nonlocal counter
        await _delay(latency.capture_s)
        counter += 1
        return CapturedFrame(
            image=_synthetic_image(),
            ts_ms=int(time.monotonic() * 1000),
            arm_id=arm_id,
        )

    return capture_fn


# -- video capture_fn (loops a real video file) ----------------------------

# Default video path: data/video.mp4 at the repo root
_DEFAULT_VIDEO_PATH = Path(__file__).parents[2] / "data" / "video.mp4"


def make_video_capture_fn(video_path: Path = _DEFAULT_VIDEO_PATH, latency: LatencyProfile = LatencyProfile.instant()):
    """Return a ``capture_fn`` that reads frames from a video file in a loop.

    Each call returns the next frame (BGR numpy array).  When the video
    ends it rewinds to the start, so the capture never runs out of frames.

    Requires ``opencv-python`` (``cv2``).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    async def capture_fn(arm_id: str) -> CapturedFrame:
        await _delay(latency.capture_s)
        ret, frame = cap.read()
        if not ret:
            # Rewind and retry
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from {video_path}")
        return CapturedFrame(
            image=frame,
            ts_ms=int(time.monotonic() * 1000),
            arm_id=arm_id,
        )

    return capture_fn


# -- tracker_factory_fn with latency ---------------------------------------


def make_mock_tracker_factory_fn_with_latency(latency: LatencyProfile = LatencyProfile.instant()):
    """Return a ``tracker_factory_fn`` with configurable latency.

    Wraps the base mock tracker factory from the perception service.
    """
    base = _base_tracker_factory_fn()

    async def factory(frame, detection):
        await _delay(latency.tracker_init_s)
        seed, raw_update = await base(frame, detection)

        async def update_with_latency(f):
            await _delay(latency.tracker_update_s)
            return await raw_update(f)

        return seed, update_with_latency

    return factory


# -- scripted decide_fn helpers --------------------------------------------


def make_scripted_decide_fn(
    script: list[callable],
    latency: LatencyProfile = LatencyProfile.instant(),
):
    """Return a ``decide_fn`` that executes steps from a script list.

    Each entry in *script* is ``(snapshot) -> list[CommandEnvelope]``.
    After all steps are consumed, returns empty lists.
    """
    idx = 0

    async def decide_fn(snapshot: PlannerSnapshot) -> list[CommandEnvelope]:
        nonlocal idx
        await _delay(latency.decide_s)
        if idx < len(script):
            fn = script[idx]
            idx += 1
            return fn(snapshot)
        return []

    return decide_fn


def make_command(
    arm_id: str,
    cmd_type: CommandType,
    payload: CommandPayload,
    snapshot_id: str | None = None,
) -> CommandEnvelope:
    """Convenience helper to build a CommandEnvelope."""
    return CommandEnvelope(
        command_id=str(uuid.uuid4()),
        arm_id=arm_id,
        issued_at_ms=int(time.time() * 1000),
        type=cmd_type,
        payload=payload,
        precondition_snapshot_id=snapshot_id,
    )
