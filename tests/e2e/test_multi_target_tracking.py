"""E2E test: track two objects sequentially — cube then robot arm.

Uses the same 5-step pattern from test_tracker_vs_vlm_bbox for each target:
  1. VLM describes scene (once, shared)
  2. Track cube → re-query VLM → compare bboxes
  3. Switch to robot → track → re-query VLM → compare bboxes

No second SCENE_DESCRIBED — the robot handle comes from the initial scene.

Scene: wooden table with a black cube and the robot arm (from Isaac Sim).
"""

from __future__ import annotations

import asyncio
import time

from tests.e2e.conftest import skip_no_vlm
from tests.e2e.utils import assert_bbox_overlap


def _log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


async def _track_and_verify(
    label: str,
    handle: str,
    svc,
    runner,
    capture_fn,
    vlm_fn,
    known_handles: list[str],
    *,
    min_iou: float = 0.5,
) -> dict:
    """Track a target, re-query VLM, compare bboxes. Returns metrics dict."""
    from halo.contracts.events import EventType
    from halo.services.target_perception_service.handle_match import find_detection_by_handle

    # ── Track target ───────────────────────────────────────────────
    _log(f"  [{label}] setting tracking target: {handle}")
    runner.recorder.clear()
    t0 = time.monotonic()
    await svc.set_tracking_target(handle)

    for _ in range(200):  # up to ~20s
        await svc.tick()
        await asyncio.sleep(0.1)
        if runner.recorder.events_of_type(EventType.TARGET_ACQUIRED):
            break

    acquired = runner.recorder.events_of_type(EventType.TARGET_ACQUIRED)
    acquire_s = time.monotonic() - t0
    assert len(acquired) >= 1, (
        f"[{label}] tracker did not acquire {handle!r} after {acquire_s:.1f}s. Events: {runner.recorder.event_types()}"
    )
    _log(f"  [{label}] TARGET_ACQUIRED ({acquire_s:.1f}s)")

    # ── Tracking observation window (3s) ─────────────────────────────
    _log(f"  [{label}] tracking observation window (3s)")
    for _ in range(30):
        await svc.tick()
        await asyncio.sleep(0.1)

    snap = await runner.runtime.get_latest_runtime_snapshot("arm0")
    assert snap.target is not None, f"[{label}] no target in snapshot after tracking"
    assert snap.target.bbox_xywh is not None, f"[{label}] tracker bbox_xywh is None"
    assert snap.target.center_px is not None, f"[{label}] tracker center_px is None"

    tracker_bbox = snap.target.bbox_xywh
    tracker_center = snap.target.center_px
    _log(
        f"  [{label}] tracker bbox(xywh): {tracker_bbox}  "
        f"center: ({tracker_center[0]:.1f}, {tracker_center[1]:.1f})  "
        f"conf: {snap.target.confidence:.2f}"
    )

    # ── Re-query VLM on current frame ──────────────────────────────
    _log(f"  [{label}] re-querying VLM")
    t1 = time.monotonic()
    frame = await capture_fn("arm0")
    vlm_scene = await vlm_fn("arm0", frame.image, known_handles, target_handle=handle)
    requery_s = time.monotonic() - t1
    _log(f"  [{label}] VLM done ({requery_s:.1f}s) — {len(vlm_scene.detections)} detections")
    for d in vlm_scene.detections:
        _log(f"    {d.handle}: bbox={d.bbox}  centroid={d.centroid}")

    vlm_det = find_detection_by_handle(
        handle,
        vlm_scene.detections,
        reference_center_px=tracker_center,
    )
    assert vlm_det is not None, (
        f"[{label}] VLM re-query didn't find {handle!r}. Got: {[d.handle for d in vlm_scene.detections]}"
    )

    # ── Compare ────────────────────────────────────────────────────
    iou = assert_bbox_overlap(label, vlm_det.bbox, tracker_bbox, tracker_center, min_iou=min_iou)

    _log(f"  [{label}] VLM bbox (xyxy): {vlm_det.bbox}")
    _log(f"  [{label}] Tracker bbox (xywh): {tracker_bbox}")
    _log(f"  [{label}] IoU: {iou:.3f}")

    return {
        "handle": handle,
        "acquire_s": acquire_s,
        "requery_s": requery_s,
        "iou": iou,
        "tracker_bbox": tracker_bbox,
        "vlm_bbox": vlm_det.bbox,
        "tracker_center": tracker_center,
    }


@skip_no_vlm
async def test_multi_target_tracking(vlm_model: str, ollama_url: str):
    """VLM describes scene → track cube → verify → switch to robot → verify."""
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
            max_duration_s=180.0,
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

        # ── Step 1: VLM scene description (shared for both targets) ────
        _log("STEP 1 — requesting VLM scene description")
        t0 = time.monotonic()
        await svc.request_refresh(reason="e2e_multi_target_init")

        for _ in range(200):
            await svc.tick()
            await asyncio.sleep(0.1)
            if runner.recorder.events_of_type(EventType.SCENE_DESCRIBED):
                break

        scene_events = runner.recorder.events_of_type(EventType.SCENE_DESCRIBED)
        describe_s = time.monotonic() - t0
        assert len(scene_events) >= 1, (
            f"VLM did not return SCENE_DESCRIBED after {describe_s:.1f}s. Events: {runner.recorder.event_types()}"
        )

        scene_data = scene_events[0].event.data
        detections = scene_data.get("detections", [])
        handles = [d["handle"] for d in detections]
        _log(f"STEP 1 done ({describe_s:.1f}s) — scene: {scene_data.get('scene', '')!r}")
        for d in detections:
            _log(f"  {d['handle']}: label={d['label']!r}  bbox={d['bbox']}")

        # Find both targets in the scene
        cube_det = next((d for d in detections if "cube" in d["handle"].lower()), None)
        robot_det = next(
            (d for d in detections if "robot" in d["handle"].lower() or "arm" in d["handle"].lower()), None
        )
        assert cube_det is not None, f"VLM didn't detect a cube. Handles: {handles}"
        assert robot_det is not None, f"VLM didn't detect the robot/arm. Handles: {handles}"

        cube_handle = cube_det["handle"]
        robot_handle = robot_det["handle"]
        _log(f"Targets: cube={cube_handle}, robot={robot_handle}")

        # ── Steps 2–4: track cube ──────────────────────────────────────
        _log("STEP 2-4 — tracking CUBE")
        cube_metrics = await _track_and_verify(
            label="CUBE",
            handle=cube_handle,
            svc=svc,
            runner=runner,
            capture_fn=capture_fn,
            vlm_fn=vlm_fn,
            known_handles=handles,
            min_iou=0.75,
        )

        # ── Steps 5–7: switch to robot ─────────────────────────────────
        _log("STEP 5-7 — switching to ROBOT (no new SCENE_DESCRIBED)")
        robot_metrics = await _track_and_verify(
            label="ROBOT",
            handle=robot_handle,
            svc=svc,
            runner=runner,
            capture_fn=capture_fn,
            vlm_fn=vlm_fn,
            known_handles=handles,
            min_iou=0.5,
        )

        # ── Summary ───────────────────────────────────────────────────
        def _row(text: str, w: int = 40) -> str:
            return f"  │ {text:<{w}} │"

        w = 40
        print(f"\n  ┌─ {vlm_model} {'─' * (w - 2 - len(vlm_model))}─┐")
        print(_row(f"VLM describe:    {describe_s:>6.1f}s", w))
        print(_row(f"Detections:      {len(detections):>6d}", w))
        print(f"  │{'─' * (w + 2)}│")
        print(_row("CUBE", w))
        print(_row(f"  acquire:       {cube_metrics['acquire_s']:>6.1f}s", w))
        print(_row(f"  VLM re-query:  {cube_metrics['requery_s']:>6.1f}s", w))
        print(_row(f"  IoU:           {cube_metrics['iou']:>6.3f}", w))
        print(f"  │{'─' * (w + 2)}│")
        print(_row("ROBOT", w))
        print(_row(f"  acquire:       {robot_metrics['acquire_s']:>6.1f}s", w))
        print(_row(f"  VLM re-query:  {robot_metrics['requery_s']:>6.1f}s", w))
        print(_row(f"  IoU:           {robot_metrics['iou']:>6.3f}", w))
        print(f"  └{'─' * (w + 2)}┘")

    finally:
        await runner.stop()
