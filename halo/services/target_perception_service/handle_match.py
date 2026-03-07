from __future__ import annotations

import dataclasses
import math
import re

from halo.services.target_perception_service.vlm_parser import VlmDetection


def dedupe_detection_handles(detections: list[VlmDetection]) -> list[VlmDetection]:
    """Return a copy of *detections* with unique handles.

    The first instance of each handle is kept unchanged.
    Later duplicates are renamed to ``{handle}_dupN`` (starting at N=2),
    skipping any suffixes that are already present in the scene.
    """
    deduped: list[VlmDetection] = []
    used_handles: set[str] = set()
    duplicate_counts: dict[str, int] = {}

    for det in detections:
        handle = det.handle
        if handle not in used_handles:
            deduped.append(det)
            used_handles.add(handle)
            duplicate_counts.setdefault(handle, 1)
            continue

        suffix = duplicate_counts.get(handle, 1) + 1
        candidate = f"{handle}_dup{suffix}"
        while candidate in used_handles:
            suffix += 1
            candidate = f"{handle}_dup{suffix}"

        duplicate_counts[handle] = suffix
        used_handles.add(candidate)
        deduped.append(dataclasses.replace(det, handle=candidate))

    return deduped


_COLOR_PREFIX_RE = re.compile(
    r"^(red|green|blue|yellow|orange|purple|pink|black|white|grey|gray|brown|beige|tan|dark|light)_"
)


def _strip_color(name: str) -> str:
    """Strip a leading color adjective from *name* (e.g. ``beige_tray`` → ``tray``)."""
    return _COLOR_PREFIX_RE.sub("", name)


def find_detection_by_handle(
    target_handle: str,
    detections: list[VlmDetection],
    *,
    reference_center_px: tuple[float, float] | None = None,
) -> VlmDetection | None:
    """Find a detection for *target_handle* using service fuzzy-match semantics.

    1. Exact handle match.
    2. Fuzzy fallback: strip trailing ``_NN`` and match on prefix.
    3. Color-agnostic fallback: strip leading color adjective + trailing ``_NN``
       and match on base object type (e.g. ``beige_tray_01`` matches ``yellow_tray``).
    """
    exact = [d for d in detections if d.handle == target_handle]
    if exact:
        return _choose_candidate(exact, reference_center_px)

    prefix = re.sub(r"_\d+$", "", target_handle)
    candidates = [d for d in detections if re.sub(r"_\d+$", "", d.handle) == prefix]
    if candidates:
        return _choose_candidate(candidates, reference_center_px)

    # Color-agnostic: strip color adjective + numeric suffix, match on base type
    base = _strip_color(prefix)
    if base != prefix:
        candidates = [d for d in detections if _strip_color(re.sub(r"_\d+$", "", d.handle)) == base]
        return _choose_candidate(candidates, reference_center_px)

    return None


def _choose_candidate(
    detections: list[VlmDetection],
    reference_center_px: tuple[float, float] | None,
) -> VlmDetection | None:
    if not detections:
        return None
    if reference_center_px is None:
        return detections[0]

    cx, cy = reference_center_px
    return min(
        detections,
        key=lambda d: math.hypot(d.centroid[0] - cx, d.centroid[1] - cy),
    )
