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


def _iou(bbox_xyxy: tuple, bbox_xywh: tuple) -> float:
    """Compute IoU between a VLM bbox (x1,y1,x2,y2) and a tracker bbox (x,y,w,h)."""
    ax1, ay1, ax2, ay2 = bbox_xyxy
    bx, by, bw, bh = bbox_xywh
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


@skip_no_vlm
async def test_tracker_vs_vlm_bbox(vlm_model: str, ollama_url: str):
    """VLM describes scene → track cube → settle → re-query VLM → compare bboxes."""
    from halo.contracts.events import EventType
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

        # ── Step 2: issue track_object for the cube ──────────────────────
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

        # ── Step 3: let tracker settle for 5s ────────────────────────────
        _log("STEP 3 — letting tracker settle (5s)")
        for _ in range(50):
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

        vlm_det_now = next((d for d in vlm_scene.detections if d.handle == cube_handle), None)
        if vlm_det_now is None:
            vlm_det_now = next((d for d in vlm_scene.detections if "cube" in d.handle.lower()), None)
        assert vlm_det_now is not None, (
            f"VLM re-query didn't find {cube_handle!r}. Got: {[d.handle for d in vlm_scene.detections]}"
        )
        vlm_bbox_now = vlm_det_now.bbox

        # ── Step 5: compare tracker bbox vs VLM bbox ─────────────────────
        iou = _iou(vlm_bbox_now, tracker_bbox)

        _log("STEP 5 — results")
        _log(f"  VLM bbox    (xyxy): {vlm_bbox_now}")
        _log(f"  Tracker bbox(xywh): {tracker_bbox}")
        _log(f"  Tracker center_px:  ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})")
        _log(f"  IoU: {iou:.3f}")

        # Summary table
        def _row(text: str, w: int = 36) -> str:
            return f"  │ {text:<{w}} │"

        w = 36
        print(f"\n  ┌─ {vlm_model} {'─' * (w - 2 - len(vlm_model))}─┐")
        print(_row(f"VLM describe:    {vlm_describe_s:>6.1f}s", w))
        print(_row(f"Tracker acquire: {tracker_acquire_s:>6.1f}s", w))
        print(_row(f"VLM re-query:    {vlm_requery_s:>6.1f}s", w))
        print(_row(f"IoU:             {iou:>6.3f}", w))
        print(_row(f"Detections:      {len(detections):>6d}", w))
        print(f"  └{'─' * (w + 2)}┘")

        # IoU > 0.5 is standard "good match" (PASCAL VOC); > 0.75 is strict (COCO AP75)
        assert iou > 0.75, f"Tracker bbox has low overlap with VLM detection (IoU={iou:.3f})"

        # Tracker centroid should be near the VLM bbox
        vx1, vy1, vx2, vy2 = vlm_bbox_now
        cx, cy = tracker_center
        margin_x = (vx2 - vx1) * 0.5
        margin_y = (vy2 - vy1) * 0.5
        assert vx1 - margin_x <= cx <= vx2 + margin_x, (
            f"Tracker centroid x={cx:.1f} outside VLM bbox [{vx1:.1f}, {vx2:.1f}] ± margin"
        )
        assert vy1 - margin_y <= cy <= vy2 + margin_y, (
            f"Tracker centroid y={cy:.1f} outside VLM bbox [{vy1:.1f}, {vy2:.1f}] ± margin"
        )

    finally:
        await runner.stop()
