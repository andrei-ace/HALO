"""Render FSM graph state as a BGR image using OpenCV drawing primitives.

Pure function: FSM dict in, numpy array out.  No side effects, no state.
Composited below the camera feed in the feed viewer subprocess.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Colours (BGR) ────────────────────────────────────────────────────

_BG = (30, 30, 30)

_NODE_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # status → (fill, text)
    "PENDING": ((50, 50, 50), (160, 160, 160)),
    "ACTIVE": ((0, 140, 0), (0, 0, 0)),
    "COMPLETED": ((30, 80, 30), (140, 140, 140)),
    "FAILED": ((0, 0, 160), (255, 255, 255)),
}

_ACTIVE_BORDER = (0, 255, 0)

_EDGE_DIM = (70, 70, 70)
_EDGE_TAKEN = (0, 180, 0)
_EDGE_RECOVERY = (0, 200, 200)
_EDGE_FAIL = (0, 0, 200)

_HEADER_TEXT = (220, 220, 220)

# ── Layout ───────────────────────────────────────────────────────────

_NODE_W = 82
_NODE_H = 28
_MARGIN_X = 14
_MARGIN_Y = 38
_SPACING_Y = 50
_HEADER_H = 24


def _is_recovery(name: str) -> bool:
    """Recovery nodes follow the RECOVER_* naming convention."""
    return name.startswith("RECOVER_")


def _topo_sort_main_path(
    nodes: list[dict],
    edges: list[dict],
) -> list[str]:
    """Topological sort of main-path (non-recovery, non-DONE) nodes."""
    main_names = {n["name"] for n in nodes if n["name"] != "DONE" and not _is_recovery(n["name"])}
    # Build in-degree map and adjacency from edges between main-path nodes
    in_degree = {name: 0 for name in main_names}
    adj: dict[str, list[str]] = {name: [] for name in main_names}
    for e in edges:
        s, t = e["source"], e["target"]
        if s in main_names and t in main_names:
            adj[s].append(t)
            in_degree[t] += 1
    # Kahn's algorithm
    queue = sorted(n for n in main_names if in_degree[n] == 0)
    result: list[str] = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for nb in sorted(adj[node]):
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    # Append any nodes not reached (e.g. cycles in a malformed graph)
    for name in sorted(main_names - set(result)):
        result.append(name)
    return result


def _compute_layout(
    nodes: list[dict],
    edges: list[dict],
    width: int,
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """Phase-id ordered layered layout.

    Returns
    -------
    (positions, layers)
        positions: {node_name: (center_x, center_y)}
        layers:    {node_name: column_index}
    """
    if not nodes:
        return {}, {}

    node_map = {n["name"]: n for n in nodes}

    main_path = _topo_sort_main_path(nodes, edges)

    # Assign layers: main path gets consecutive columns 0..N-1
    layers: dict[str, int] = {}
    for i, name in enumerate(main_path):
        layers[name] = i

    # Recovery nodes: place in the same column as the earliest main-path
    # node that has an edge *to* them.
    edge_sources: dict[str, list[str]] = {}
    for e in edges:
        edge_sources.setdefault(e["target"], []).append(e["source"])

    for n in nodes:
        name = n["name"]
        if name in layers or name == "DONE":
            continue
        if _is_recovery(name):
            # Find the earliest source column that feeds into this recovery node
            sources = edge_sources.get(name, [])
            source_layers = [layers[s] for s in sources if s in layers]
            if source_layers:
                layers[name] = min(source_layers) + 1
            else:
                layers[name] = len(main_path)
        else:
            # Unknown node — append after main path
            layers[name] = len(main_path)

    # DONE always in rightmost column
    max_layer = max(layers.values()) if layers else 0
    layers["DONE"] = max_layer + 1

    # Group by layer, split main vs recovery rows
    layer_groups: dict[int, list[str]] = {}
    for name, layer in layers.items():
        if name in node_map:
            layer_groups.setdefault(layer, []).append(name)

    # Compute positions — fit all columns within available width
    positions: dict[str, tuple[int, int]] = {}
    num_layers = max(layers.values()) + 1 if layers else 1
    avail_w = width - 2 * _MARGIN_X - _NODE_W
    spacing_x = avail_w // max(num_layers - 1, 1) if num_layers > 1 else 0

    for layer_idx, names in layer_groups.items():
        main = [n for n in names if not _is_recovery(n)]
        recovery = [n for n in names if _is_recovery(n)]
        cx = _MARGIN_X + _NODE_W // 2 + layer_idx * spacing_x
        # Main row
        for i, name in enumerate(main):
            cy = _HEADER_H + _MARGIN_Y + i * _SPACING_Y
            positions[name] = (cx, cy)
        # Recovery row below
        for i, name in enumerate(recovery):
            cy = _HEADER_H + _MARGIN_Y + (len(main) + i) * _SPACING_Y
            positions[name] = (cx, cy)

    return positions, layers


# ── Drawing ──────────────────────────────────────────────────────────


_LABEL_MAX_LEN = 12


def _short_label(name: str) -> str:
    """Return a compact display label for a node, auto-shortened to fit."""
    # Strip RECOVER_ prefix for recovery nodes
    display = name.removeprefix("RECOVER_")
    words = display.split("_")
    label = " ".join(words)
    if len(label) <= _LABEL_MAX_LEN:
        return label
    # Progressively cap word length until it fits
    for cap in range(5, 2, -1):
        label = " ".join(w[:cap] if len(w) > cap else w for w in words)
        if len(label) <= _LABEL_MAX_LEN:
            return label
    return label[:_LABEL_MAX_LEN]


def _draw_arrowhead(
    canvas: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    """Draw a small triangular arrowhead at *tip* pointing along *direction*."""
    dx, dy = direction
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    a = 7
    px, py = float(tip[0]), float(tip[1])
    left = (int(px - a * ux + a * 0.4 * uy), int(py - a * uy - a * 0.4 * ux))
    right = (int(px - a * ux - a * 0.4 * uy), int(py - a * uy + a * 0.4 * ux))
    cv2.fillPoly(canvas, [np.array([tip, left, right])], color, cv2.LINE_AA)


def _bezier_curve(p0: tuple, p1: tuple, p2: tuple, p3: tuple, n: int = 30) -> np.ndarray:
    """Compute *n* points on a cubic Bezier curve as an int32 array."""
    ts = np.linspace(0, 1, n).reshape(-1, 1)
    pts = (
        (1 - ts) ** 3 * np.array(p0)
        + 3 * (1 - ts) ** 2 * ts * np.array(p1)
        + 3 * (1 - ts) * ts**2 * np.array(p2)
        + ts**3 * np.array(p3)
    )
    return pts.astype(np.int32)


def _edge_color(
    src: str,
    tgt: str,
    label: str | None,
    taken: bool,
) -> tuple[tuple[int, int, int], int]:
    """Return (color, thickness) for an edge."""
    if not taken:
        return _EDGE_DIM, 1
    if _is_recovery(tgt) or _is_recovery(src):
        return _EDGE_RECOVERY, 2
    if tgt == "DONE" and label in ("timeout", "max_retries"):
        return _EDGE_FAIL, 2
    return _EDGE_TAKEN, 2


def _draw_edges(
    canvas: np.ndarray,
    edges: list[dict],
    positions: dict[str, tuple[int, int]],
    taken_set: set[tuple[str, str]],
    layers: dict[str, int],
) -> None:
    """Draw edges with smart routing.

    - Adjacent-layer forward edges: straight horizontal lines.
    - Skip edges (span ≥2 columns forward): arc above the main row.
    - Back edges (target column < source column): arc below.
    - Down/up edges (to/from recovery row): straight diagonal.
    """
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in positions or tgt not in positions:
            continue
        sx, sy = positions[src]
        tx, ty = positions[tgt]
        label = e.get("label")
        taken = (src, tgt) in taken_set
        color, thickness = _edge_color(src, tgt, label, taken)

        # Connection points: right edge of source, left edge of target
        p1 = (sx + _NODE_W // 2, sy)
        p2 = (tx - _NODE_W // 2, ty)

        src_layer = layers.get(src, 0)
        tgt_layer = layers.get(tgt, 0)
        span = tgt_layer - src_layer

        if sy != ty:
            # Different rows (e.g. main ↔ recovery) — straight diagonal
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
            _draw_arrowhead(canvas, p2, (p2[0] - p1[0], p2[1] - p1[1]), color)
        elif span == 1:
            # Adjacent columns — straight horizontal
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
            _draw_arrowhead(canvas, p2, (1, 0), color)
        elif span >= 2:
            # Skip edge — arc above the main row
            mid_x = (p1[0] + p2[0]) // 2
            arc_y = sy - _NODE_H - 8 - 4 * span  # higher for longer skips
            pts = _bezier_curve(p1, (mid_x, arc_y), (mid_x, arc_y), p2)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            # Arrowhead direction from second-to-last point
            d = (int(pts[-1][0] - pts[-3][0]), int(pts[-1][1] - pts[-3][1]))
            _draw_arrowhead(canvas, tuple(pts[-1]), d, color)
        else:
            # Back edge (span < 0) — arc below the main row
            mid_x = (p1[0] + p2[0]) // 2
            arc_y = sy + _NODE_H + 10 + 4 * abs(span)
            # Route: exit bottom of source, arc below, enter bottom of target
            p1_bot = (sx + _NODE_W // 2, sy + _NODE_H // 2)
            p2_bot = (tx - _NODE_W // 2, ty + _NODE_H // 2)
            pts = _bezier_curve(p1_bot, (p1_bot[0], arc_y), (p2_bot[0], arc_y), p2_bot)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            d = (int(pts[-1][0] - pts[-3][0]), int(pts[-1][1] - pts[-3][1]))
            _draw_arrowhead(canvas, tuple(pts[-1]), d, color)

        # Edge label near midpoint for skip/back edges
        if abs(span) >= 2 and label:
            lx = (p1[0] + p2[0]) // 2
            ly = (sy - _NODE_H - 6 * span) if span >= 2 else (sy + _NODE_H + 8)
            cv2.putText(canvas, label, (lx - 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)


def _draw_nodes(
    canvas: np.ndarray,
    nodes: list[dict],
    positions: dict[str, tuple[int, int]],
    current_node: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n in nodes:
        name = n["name"]
        if name not in positions:
            continue
        cx, cy = positions[name]
        status = n.get("status", "PENDING")
        fill, text_color = _NODE_COLORS.get(status, _NODE_COLORS["PENDING"])

        # Rounded rect
        x1, y1 = cx - _NODE_W // 2, cy - _NODE_H // 2
        x2, y2 = cx + _NODE_W // 2, cy + _NODE_H // 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)

        # Active border
        if name == current_node and status == "ACTIVE":
            cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), _ACTIVE_BORDER, 2, cv2.LINE_AA)

        # Label
        label = _short_label(name)
        elapsed_ms = n.get("elapsed_ms")
        if name == current_node and elapsed_ms is not None:
            secs = elapsed_ms / 1000.0
            label += f" {secs:.1f}s"

        # Auto-scale font to fit inside node
        font_scale = 0.33
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        if tw > _NODE_W - 6:
            font_scale = font_scale * (_NODE_W - 6) / tw
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(canvas, label, (tx, ty), font, font_scale, text_color, 1, cv2.LINE_AA)


def _draw_header(
    canvas: np.ndarray,
    skill_name: str | None,
    target_handle: str | None,
    outcome: str | None,
    failure_code: str | None,
    width: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if skill_name is None:
        text = "No active skill"
        cv2.putText(canvas, text, (10, 16), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        return

    text = skill_name
    if target_handle:
        text += f" -> {target_handle}"
    cv2.putText(canvas, text, (10, 16), font, 0.45, _HEADER_TEXT, 1, cv2.LINE_AA)

    # Outcome badge on the right
    if outcome and outcome != "IN_PROGRESS":
        badge_color = (0, 180, 0) if outcome == "SUCCESS" else (0, 0, 200)
        badge_text = outcome
        if failure_code:
            badge_text += f" ({failure_code})"
        (bw, bh), _ = cv2.getTextSize(badge_text, font, 0.4, 1)
        bx = width - bw - 12
        cv2.putText(canvas, badge_text, (bx, 16), font, 0.4, badge_color, 1, cv2.LINE_AA)


# ── Public API ───────────────────────────────────────────────────────

# Layout cache keyed by (skill_name, node_count, width)
_layout_cache: dict[tuple[str, int, int], tuple[dict[str, tuple[int, int]], dict[str, int]]] = {}

_DEFAULT_HEIGHT = 220


def render_fsm_overlay(fsm_dict: dict | None, width: int, height: int = _DEFAULT_HEIGHT) -> np.ndarray:
    """Render FSM graph as a BGR image.

    Parameters
    ----------
    fsm_dict : dict | None
        Serialized FsmViewModel, or None when no skill is active.
    width : int
        Canvas width in pixels (should match camera frame width).
    height : int
        Canvas height in pixels.

    Returns
    -------
    np.ndarray
        BGR image of shape (height, width, 3).
    """
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

    if fsm_dict is None:
        _draw_header(canvas, None, None, None, None, width)
        return canvas

    skill_name = fsm_dict.get("skill_name")
    target_handle = fsm_dict.get("target_handle")
    outcome = fsm_dict.get("outcome")
    failure_code = fsm_dict.get("failure_code")
    nodes = fsm_dict.get("nodes", [])
    edges = fsm_dict.get("edges", [])
    current_node = fsm_dict.get("current_node", "")
    transitions = fsm_dict.get("transition_history", [])

    _draw_header(canvas, skill_name, target_handle, outcome, failure_code, width)

    # Compute or retrieve cached layout
    variant = fsm_dict.get("variant", "")
    cache_key = (skill_name or "", variant, len(nodes), width)
    if cache_key in _layout_cache:
        positions, layer_map = _layout_cache[cache_key]
    else:
        positions, layer_map = _compute_layout(nodes, edges, width)
        _layout_cache[cache_key] = (positions, layer_map)
        # Limit cache size
        if len(_layout_cache) > 20:
            oldest = next(iter(_layout_cache))
            del _layout_cache[oldest]

    # Highlight all edges taken during the current skill run
    taken_set: set[tuple[str, str]] = set()
    for t in transitions:
        taken_set.add((t.get("from_node", ""), t.get("to_node", "")))

    _draw_edges(canvas, edges, positions, taken_set, layer_map)
    _draw_nodes(canvas, nodes, positions, current_node)

    # Previous skill recap (last ≤3 transitions) — small text at bottom-left
    prev = fsm_dict.get("prev_skill")
    if prev is not None:
        _draw_prev_skill(canvas, prev, height)

    return canvas


_MAX_PREV_RENDER_NODES = 3


def _draw_prev_skill(canvas: np.ndarray, prev: dict, canvas_h: int) -> None:
    """Draw a one-line recap of the previous skill at the bottom of the canvas.

    Full transition history is available in *prev*, but only the last
    ``_MAX_PREV_RENDER_NODES`` nodes are rendered.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = prev.get("skill_name", "?")
    outcome = prev.get("outcome", "?")
    transitions = prev.get("transition_history", [])

    # Collect unique ordered node names from the full history
    all_nodes: list[str] = []
    seen: set[str] = set()
    for t in transitions:
        for key in ("from_node", "to_node"):
            n = t.get(key, "")
            if n and n not in seen:
                all_nodes.append(n)
                seen.add(n)

    # Show only the last N nodes
    tail = all_nodes[-_MAX_PREV_RENDER_NODES:] if len(all_nodes) > _MAX_PREV_RENDER_NODES else all_nodes
    ellipsis = "... > " if len(all_nodes) > _MAX_PREV_RENDER_NODES else ""
    chain = ellipsis + " > ".join(tail) if tail else ""

    outcome_tag = "OK" if outcome == "SUCCESS" else outcome
    text = f"prev: {name} [{outcome_tag}]"
    if chain:
        text += f"  {chain}"

    color = (0, 120, 0) if outcome == "SUCCESS" else (0, 0, 140)
    y = canvas_h - 6
    cv2.putText(canvas, text, (10, y), font, 0.3, color, 1, cv2.LINE_AA)
