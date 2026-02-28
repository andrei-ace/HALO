"""Shared helpers for e2e tests."""

from __future__ import annotations


def iou(bbox_xyxy: tuple, bbox_xywh: tuple) -> float:
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


def assert_bbox_overlap(
    label: str,
    vlm_bbox_xyxy: tuple,
    tracker_bbox_xywh: tuple,
    tracker_center: tuple[float, float],
    *,
    min_iou: float = 0.5,
) -> float:
    """Assert IoU and centroid proximity, return IoU."""
    score = iou(vlm_bbox_xyxy, tracker_bbox_xywh)
    vx1, vy1, vx2, vy2 = vlm_bbox_xyxy
    cx, cy = tracker_center
    margin_x = (vx2 - vx1) * 0.5
    margin_y = (vy2 - vy1) * 0.5

    assert score > min_iou, f"{label}: tracker bbox has low overlap with VLM detection (IoU={score:.3f}, min={min_iou})"
    assert vx1 - margin_x <= cx <= vx2 + margin_x, (
        f"{label}: tracker centroid x={cx:.1f} outside VLM bbox [{vx1:.1f}, {vx2:.1f}] ± margin"
    )
    assert vy1 - margin_y <= cy <= vy2 + margin_y, (
        f"{label}: tracker centroid y={cy:.1f} outside VLM bbox [{vy1:.1f}, {vy2:.1f}] ± margin"
    )
    return score
