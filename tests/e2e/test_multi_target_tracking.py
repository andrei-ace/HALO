"""E2E test: track two objects sequentially — cube then robot arm.

Uses the same 5-step pattern from test_tracker_vs_vlm_bbox for each target:
  1. VLM describes scene (once, shared)
  2. Track cube → settle → re-query VLM → compare bboxes
  3. Switch to robot → track → settle → re-query VLM → compare bboxes

No second SCENE_DESCRIBED — the robot handle comes from the initial scene.

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


def _assert_bbox_overlap(
    label: str,
    vlm_bbox_xyxy: tuple,
    tracker_bbox_xywh: tuple,
    tracker_center: tuple[float, float],
    *,
    min_iou: float = 0.5,
) -> float:
    """Assert IoU and centroid proximity, return IoU."""
    iou = _iou(vlm_bbox_xyxy, tracker_bbox_xywh)
    vx1, vy1, vx2, vy2 = vlm_bbox_xyxy
    cx, cy = tracker_center
    margin_x = (vx2 - vx1) * 0.5
    margin_y = (vy2 - vy1) * 0.5

    assert iou > min_iou, f"{label}: tracker bbox has low overlap with VLM detection (IoU={iou:.3f}, min={min_iou})"
    assert vx1 - margin_x <= cx <= vx2 + margin_x, (
        f"{label}: tracker centroid x={cx:.1f} outside VLM bbox [{vx1:.1f}, {vx2:.1f}] ± margin"
    )
    assert vy1 - margin_y <= cy <= vy2 + margin_y, (
        f"{label}: tracker centroid y={cy:.1f} outside VLM bbox [{vy1:.1f}, {vy2:.1f}] ± margin"
    )
    return iou


async def _track_and_verify(
    label: str,
    handle: str,
    svc,
    runner,
    capture_fn,
    vlm_fn,
    vlm_model: str,
    known_handles: list[str],
    *,
    min_iou: float = 0.5,
) -> dict:
    """Track a target, settle, re-query VLM, compare bboxes. Returns metrics dict."""
    from halo.contracts.events import EventType

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

    # ── Settle for 5s ──────────────────────────────────────────────
    _log(f"  [{label}] letting tracker settle (5s)")
    for _ in range(50):
        await svc.tick()
        await asyncio.sleep(0.1)

    snap = await runner.runtime.get_latest_runtime_snapshot("arm0")
    assert snap.target is not None, f"[{label}] no target in snapshot after tracking"
    assert snap.target.bbox_xywh is not None, f"[{label}] tracker bbox_xywh is None"
    assert snap.target.center_px is not None, f"[{label}] tracker center_px is None"

    tracker_bbox = snap.target.bbox_xywh
    tracker_center = snap.target.center_px
    _log(
        f"  [{label}] settled — bbox(xywh): {tracker_bbox}  "
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

    vlm_det = next((d for d in vlm_scene.detections if d.handle == handle), None)
    if vlm_det is None:
        # fuzzy: match on keyword from handle (e.g. "cube" or "robot")
        keyword = handle.split("_")[0] if "_" in handle else handle
        vlm_det = next((d for d in vlm_scene.detections if keyword in d.handle.lower()), None)
    assert vlm_det is not None, (
        f"[{label}] VLM re-query didn't find {handle!r}. Got: {[d.handle for d in vlm_scene.detections]}"
    )

    # ── Compare ────────────────────────────────────────────────────
    iou = _assert_bbox_overlap(label, vlm_det.bbox, tracker_bbox, tracker_center, min_iou=min_iou)

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
            vlm_model=vlm_model,
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
            vlm_model=vlm_model,
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
