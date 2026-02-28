"""E2E test: MuJocoVideoSource → real VLM + real OpenCV tracker → bbox verification.

Full pipeline: MuJoCo scene camera → VLM describe → track red cube → settle →
VLM re-query → compare tracker bbox vs VLM bbox (IoU).

The Lift env has a single static red cube, so one test covers the tracking pipeline.

Requires: robosuite, Ollama with VLM model.
Auto-skips if either is unavailable.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.e2e.conftest import skip_no_vlm
from tests.e2e.utils import assert_bbox_overlap, iou

try:
    from halo.services.target_perception_service.mujoco_source import _suppress_robosuite_warnings

    _suppress_robosuite_warnings()
    import robosuite  # noqa: F401

    _has_robosuite = True
except ImportError:
    _has_robosuite = False

skip_no_robosuite = pytest.mark.skipif(not _has_robosuite, reason="robosuite not installed")


def _log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


@skip_no_robosuite
@skip_no_vlm
async def test_mujoco_tracker_vs_vlm_bbox(vlm_model: str, ollama_url: str):
    """MuJoCo scene → VLM describe → track cube → settle → VLM re-query → IoU check."""
    from halo.contracts.events import EventType
    from halo.services.target_perception_service.mujoco_source import MuJocoVideoSource
    from halo.services.target_perception_service.ollama_vlm_fn import make_ollama_vlm_fn
    from halo.services.target_perception_service.tracker_fn import make_tracker_factory_fn
    from halo.testing.runner import HeadlessRunner, RunnerConfig

    _log(f"VLM model: {vlm_model}")

    # ── Start MuJoCo source (blocks until first frame) ─────────────────
    _log("Creating MuJocoVideoSource...")
    source = MuJocoVideoSource(fps=10.0, seed=42)
    source.start(timeout=60.0)
    _log(f"MuJocoVideoSource ready (fps={source.fps})")

    assert source.latest_frame is not None, "Source should have at least one frame after start()"
    frame_shape = source.latest_frame.shape
    _log(f"Frame shape: {frame_shape}")
    assert len(frame_shape) == 3 and frame_shape[2] == 3, f"Expected HWC BGR frame, got shape {frame_shape}"

    vlm_fn = make_ollama_vlm_fn(base_url=ollama_url, model=vlm_model)
    capture_fn = source.make_capture_fn("arm0")
    tracker_factory = make_tracker_factory_fn()

    runner = HeadlessRunner(
        config=RunnerConfig(
            arm_id="arm0",
            max_duration_s=120.0,
            enable_planner=False,
            enable_skill_runner=False,
            enable_control=False,
            enable_perception=True,
        ),
        vlm_fn=vlm_fn,
        capture_fn=capture_fn,
        tracker_factory_fn=tracker_factory,
    )

    await runner.start()
    try:
        svc = runner.perception_svc
        assert svc is not None

        # ── Step 1: trigger VLM scene description ──────────────────────
        _log("STEP 1 — requesting VLM scene description from MuJoCo frame")
        t0 = time.monotonic()
        await svc.request_refresh(reason="e2e_mujoco_test")

        for _ in range(300):  # up to ~30s
            await svc.tick()
            await asyncio.sleep(0.1)
            if runner.recorder.events_of_type(EventType.SCENE_DESCRIBED):
                break

        scene_events = runner.recorder.events_of_type(EventType.SCENE_DESCRIBED)
        vlm_describe_s = time.monotonic() - t0
        assert len(scene_events) >= 1, (
            f"VLM did not return SCENE_DESCRIBED after {vlm_describe_s:.1f}s. Events: {runner.recorder.event_types()}"
        )

        scene_data = scene_events[0].event.data
        detections = scene_data.get("detections", [])
        handles = [d["handle"] for d in detections]
        _log(f"STEP 1 done ({vlm_describe_s:.1f}s) — scene: {scene_data.get('scene', '')!r}")
        for d in detections:
            _log(f"  {d['handle']}: label={d['label']!r}  bbox={d['bbox']}")

        assert len(detections) > 0, "VLM returned no detections from MuJoCo scene"

        # Find the cube (Lift env always has one red cube)
        cube_det = next((d for d in detections if "cube" in d["handle"].lower()), None)
        graspable_det = next((d for d in detections if d.get("is_graspable")), None)
        target_det = cube_det or graspable_det
        assert target_det is not None, f"No cube or graspable object detected. Handles: {handles}"
        cube_handle = target_det["handle"]
        _log(f"Selected target: {cube_handle}")

        # ── Step 2: issue track_object for the cube ────────────────────
        _log(f"STEP 2 — setting tracking target: {cube_handle}")
        runner.recorder.clear()
        t1 = time.monotonic()
        await svc.set_tracking_target(cube_handle)

        for _ in range(300):  # up to ~30s
            await svc.tick()
            await asyncio.sleep(0.1)
            if runner.recorder.events_of_type(EventType.TARGET_ACQUIRED):
                break

        acquired = runner.recorder.events_of_type(EventType.TARGET_ACQUIRED)
        tracker_acquire_s = time.monotonic() - t1
        assert len(acquired) >= 1, (
            f"Tracker did not acquire target after {tracker_acquire_s:.1f}s. Events: {runner.recorder.event_types()}"
        )
        _log(f"STEP 2 done ({tracker_acquire_s:.1f}s) — TARGET_ACQUIRED")

        # ── Step 3: let tracker settle (5s — static scene) ────────────
        _log("STEP 3 — letting tracker settle (5s)")
        for _ in range(50):
            await svc.tick()
            await asyncio.sleep(0.1)

        snap = await runner.runtime.get_latest_runtime_snapshot("arm0")
        assert snap.target is not None, "No target in snapshot after tracking"
        assert snap.target.hint_valid, "Target hint should be valid after settling"
        assert snap.target.bbox_xywh is not None, "Tracker bbox_xywh is None — real tracker required"
        assert snap.target.center_px is not None, "Tracker center_px is None"

        tracker_bbox = snap.target.bbox_xywh
        tracker_center = snap.target.center_px
        _log(
            f"STEP 3 done — tracker bbox (xywh): {tracker_bbox}  "
            f"center_px: ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})  "
            f"confidence: {snap.target.confidence:.2f}"
        )

        # ── Step 4: capture current frame and re-query VLM ─────────────
        _log("STEP 4 — capturing frame and re-querying VLM")
        t2 = time.monotonic()
        frame = await capture_fn("arm0")
        vlm_scene = await vlm_fn("arm0", frame.image, [cube_handle], target_handle=cube_handle)
        vlm_requery_s = time.monotonic() - t2
        _log(f"STEP 4 done ({vlm_requery_s:.1f}s) — VLM detections: {len(vlm_scene.detections)}")
        for d in vlm_scene.detections:
            _log(f"  {d.handle}: bbox={d.bbox}  centroid={d.centroid}")

        vlm_det_now = next((d for d in vlm_scene.detections if d.handle == cube_handle), None)
        if vlm_det_now is None:
            vlm_det_now = next((d for d in vlm_scene.detections if "cube" in d.handle.lower()), None)
        assert vlm_det_now is not None, (
            f"VLM re-query didn't find {cube_handle!r}. Got: {[d.handle for d in vlm_scene.detections]}"
        )
        vlm_bbox_now = vlm_det_now.bbox

        # ── Step 5: compare tracker bbox vs VLM bbox ───────────────────
        score = iou(vlm_bbox_now, tracker_bbox)

        _log("STEP 5 — bbox comparison")
        _log(f"  VLM bbox    (xyxy): {vlm_bbox_now}")
        _log(f"  Tracker bbox(xywh): {tracker_bbox}")
        _log(f"  Tracker center_px:  ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})")
        _log(f"  IoU: {score:.3f}")

        # Summary table
        def _row(text: str, w: int = 36) -> str:
            return f"  │ {text:<{w}} │"

        w = 36
        print(f"\n  ┌─ MuJoCo + {vlm_model} {'─' * max(0, w - 10 - len(vlm_model))}─┐")
        print(_row(f"VLM describe:    {vlm_describe_s:>6.1f}s", w))
        print(_row(f"Tracker acquire: {tracker_acquire_s:>6.1f}s", w))
        print(_row(f"VLM re-query:    {vlm_requery_s:>6.1f}s", w))
        print(_row(f"IoU:             {score:>6.3f}", w))
        print(_row(f"Detections:      {len(detections):>6d}", w))
        print(f"  └{'─' * (w + 2)}┘")

        # Static scene (cube doesn't move) → expect high IoU
        assert_bbox_overlap("mujoco_cube", vlm_bbox_now, tracker_bbox, tracker_center, min_iou=0.75)

    finally:
        await runner.stop()
        source.stop()


@skip_no_robosuite
async def test_mujoco_source_frame_basics():
    """MuJocoVideoSource produces valid BGR frames and capture_fn works."""
    from halo.services.target_perception_service.mujoco_source import MuJocoVideoSource

    source = MuJocoVideoSource(fps=10.0, seed=42)
    source.start(timeout=60.0)

    try:
        # latest_frame should be populated
        frame = source.latest_frame
        assert frame is not None, "No frame after start()"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected HWC BGR, got {frame.shape}"
        assert frame.shape[0] == 480 and frame.shape[1] == 640, f"Expected 480x640, got {frame.shape}"

        # capture_fn should return a CapturedFrame
        capture_fn = source.make_capture_fn("arm0")
        captured = await capture_fn("arm0")
        assert captured.image is not None
        assert captured.arm_id == "arm0"
        assert captured.ts_ms > 0
        assert captured.image.shape == (480, 640, 3)

        # A second capture should also work
        captured2 = await capture_fn("arm0")
        assert captured2.ts_ms >= captured.ts_ms

        # latest_frame should reflect the last capture_fn render (cached, not re-rendered)
        latest = source.latest_frame
        assert latest is not None
        assert latest.shape == (480, 640, 3)

    finally:
        source.stop()


@skip_no_robosuite
async def test_mujoco_source_seeded_reproducibility():
    """Two MuJocoVideoSource instances with the same seed produce identical first frames."""
    import numpy as np

    from halo.services.target_perception_service.mujoco_source import MuJocoVideoSource

    source1 = MuJocoVideoSource(fps=10.0, seed=42)
    source1.start(timeout=60.0)
    frame1 = source1.latest_frame
    source1.stop()

    source2 = MuJocoVideoSource(fps=10.0, seed=42)
    source2.start(timeout=60.0)
    frame2 = source2.latest_frame
    source2.stop()

    assert frame1 is not None and frame2 is not None
    assert np.array_equal(frame1, frame2), "Seeded sources should produce identical first frames"
