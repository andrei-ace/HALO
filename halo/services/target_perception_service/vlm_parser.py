from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class VlmScene:
    scene: str
    detections: list[VlmDetection]


@dataclass(frozen=True)
class VlmDetection:
    handle: str
    label: str
    # Bounding box (x1, y1, x2, y2), normalised to 0..1
    bbox: tuple[float, float, float, float]
    # Centroid derived from bbox midpoint, normalised to 0..1
    centroid: tuple[float, float]
    is_graspable: bool


def parse_vlm_response(response: dict, *, img_w: int = 1, img_h: int = 1) -> VlmScene:
    """Parse a raw VLM JSON response into a VlmScene.

    When *img_w* and *img_h* are provided (the pixel dimensions of the image
    sent to the VLM), bbox and centroid coordinates are normalised to 0..1.
    Without them (default 1×1) coordinates pass through as-is.
    """
    scene = response.get("scene", "")
    detections: list[VlmDetection] = []
    for det in response.get("detections", []):
        x1, y1, x2, y2 = det["bounding_box"]
        detections.append(
            VlmDetection(
                handle=det["handle"],
                label=det["label"],
                bbox=(float(x1) / img_w, float(y1) / img_h, float(x2) / img_w, float(y2) / img_h),
                centroid=(
                    (x1 + x2) / 2 / img_w,
                    (y1 + y2) / 2 / img_h,
                ),
                is_graspable=det.get("is_graspable", True),
            )
        )
    return VlmScene(scene=scene, detections=detections)


def normalize_detection(det: VlmDetection, img_w: int, img_h: int) -> VlmDetection:
    """Normalise a detection's bbox/centroid from pixel coords to 0..1."""
    x1, y1, x2, y2 = det.bbox
    return dataclasses.replace(
        det,
        bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
        centroid=(det.centroid[0] / img_w, det.centroid[1] / img_h),
    )
