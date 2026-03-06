"""E2E test: real VLM + real OpenCV tracker + video from data/video.mp4.

Requires Ollama running with at least one VLM model.
Auto-skips models that aren't installed. Parametrized over qwen2.5vl:3b and :7b
so results can be compared side-by-side.

Scene: wooden table with a black cube and the robot arm (from Isaac Sim).
"""

from __future__ import annotations

import asyncio
import time

from tests.e2e.conftest import skip_no_vlm
from tests.e2e.utils import assert_bbox_overlap, iou


def _log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


@skip_no_vlm
async def test_tracker_vs_vlm_bbox(vlm_model: str, ollama_url: str):
    """VLM describes scene → track cube → re-query VLM → compare bboxes."""
    from halo.contracts.events import EventType
    from halo.services.target_perception_service.handle_match import find_detection_by_handle
    from halo.services.target_perception_service.ollama_vlm_fn import make_ollama_vlm_fn
    from halo.services.target_perception_service.tracker_fn import make_tracker_factory_fn
    from halo.testing.mock_fns import make_video_capture_fn
    from halo.testing.runner import HeadlessRunner, RunnerConfig

    _log(f"VLM model: {vlm_model}")

    vlm_fn = make_ollama_vlm_fn(base_url=ollama_url, model=vlm_model)
    capture_fn = make_video_capture_fn()
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

        # ── Step 1: trigger VLM scene description ────────────────────────
        _log("STEP 1 — requesting VLM scene description")
        t0 = time.monotonic()
        await svc.request_refresh(reason="e2e_test_init")

        for _ in range(200):  # up to ~20s at 100ms/tick
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

        cube_det = next((d for d in detections if "cube" in d["handle"].lower()), None)
        assert cube_det is not None, f"VLM didn't detect any cube. Handles: {handles}"
        cube_handle = cube_det["handle"]
        _log(f"Selected target: {cube_handle}")

        # ── Step 2: set tracking target for the cube ──────────────────────
        _log(f"STEP 2 — setting tracking target: {cube_handle}")
        runner.recorder.clear()
        t1 = time.monotonic()
        await svc.set_tracking_target(cube_handle)

        for _ in range(200):
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

        # ── Step 3: tracking observation window (3s) ─────────────────────
        _log("STEP 3 — tracking observation window (3s)")
        for _ in range(30):
            await svc.tick()
            await asyncio.sleep(0.1)

        snap = await runner.runtime.get_latest_runtime_snapshot("arm0")
        assert snap.target is not None, "No target in snapshot after tracking"
        assert snap.target.bbox_xywh is not None, "Tracker bbox_xywh is None — real tracker required"
        assert snap.target.center_px is not None, "Tracker center_px is None"

        tracker_bbox = snap.target.bbox_xywh
        tracker_center = snap.target.center_px
        _log(
            f"STEP 3 done — tracker bbox (xywh): {tracker_bbox}  "
            f"center_px: ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})  "
            f"confidence: {snap.target.confidence:.2f}"
        )

        # ── Step 4: capture current frame and re-query VLM ───────────────
        _log("STEP 4 — capturing frame and re-querying VLM")
        t2 = time.monotonic()
        frame = await capture_fn("arm0")
        vlm_scene = await vlm_fn("arm0", frame.image, [cube_handle], target_handle=cube_handle)
        vlm_requery_s = time.monotonic() - t2
        _log(f"STEP 4 done ({vlm_requery_s:.1f}s) — VLM detections: {len(vlm_scene.detections)}")
        for d in vlm_scene.detections:
            _log(f"  {d.handle}: bbox={d.bbox}  centroid={d.centroid}")

        vlm_det_now = find_detection_by_handle(
            cube_handle,
            vlm_scene.detections,
            reference_center_px=tracker_center,
        )
        assert vlm_det_now is not None, (
            f"VLM re-query didn't find {cube_handle!r}. Got: {[d.handle for d in vlm_scene.detections]}"
        )
        vlm_bbox_now = vlm_det_now.bbox

        # ── Step 5: compare tracker bbox vs VLM bbox ─────────────────────
        score = iou(vlm_bbox_now, tracker_bbox)

        _log("STEP 5 — results")
        _log(f"  VLM bbox    (xyxy): {vlm_bbox_now}")
        _log(f"  Tracker bbox(xywh): {tracker_bbox}")
        _log(f"  Tracker center_px:  ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})")
        _log(f"  IoU: {score:.3f}")

        # Summary table
        def _row(text: str, w: int = 36) -> str:
            return f"  │ {text:<{w}} │"

        w = 36
        print(f"\n  ┌─ {vlm_model} {'─' * (w - 2 - len(vlm_model))}─┐")
        print(_row(f"VLM describe:    {vlm_describe_s:>6.1f}s", w))
        print(_row(f"Tracker acquire: {tracker_acquire_s:>6.1f}s", w))
        print(_row(f"VLM re-query:    {vlm_requery_s:>6.1f}s", w))
        print(_row(f"IoU:             {score:>6.3f}", w))
        print(_row(f"Detections:      {len(detections):>6d}", w))
        print(f"  └{'─' * (w + 2)}┘")

        # IoU > 0.75 is strict (COCO AP75); assert + centroid proximity
        assert_bbox_overlap("cube", vlm_bbox_now, tracker_bbox, tracker_center, min_iou=0.75)

    finally:
        await runner.stop()
