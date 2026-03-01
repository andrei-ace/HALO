"""Measure the TCP pinch-point offset from gripperframe to the jaw contact surface.

The SO-101 gripper has a fixed jaw (wrist_roll_follower_so101_v1 mesh on the
``gripper`` body) and a moving jaw (moving_jaw_so101_v1 child body).  IK targets
the ``gripperframe`` site.  This script computes the offset from gripperframe to
the actual jaw contact surface by finding mesh vertices that are close together
when the gripper is closed.

Usage::

    uv run python -m mujoco_sim.scripts.measure_pinch_offset
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_SCENE_XML = str(Path(__file__).resolve().parent.parent / "assets" / "so101" / "pick_scene.xml")


def _get_mesh_vertices_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    """Return mesh vertices transformed to world frame."""
    mesh_id = model.geom_dataid[geom_id]
    vs = model.mesh_vertadr[mesh_id]
    vc = model.mesh_vertnum[mesh_id]
    verts_local = model.mesh_vert[vs : vs + vc].copy()
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    pos = data.geom_xpos[geom_id]
    return (rot @ verts_local.T).T + pos


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data = mujoco.MjData(model)

    # Close the gripper
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[gripper_joint_id] = -0.17  # GRIPPER_CLOSE
    mujoco.mj_forward(model, data)

    # --- Resolve IDs ---
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    moving_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

    # Fixed jaw: last collision geom (group 3) on gripper body
    gripper_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == gripper_body_id]
    fixed_jaw_geom_id = [g for g in gripper_geoms if model.geom_group[g] == 3][-1]

    # Moving jaw: collision geom on moving_jaw body
    moving_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == moving_body_id]
    moving_jaw_geom_id = [g for g in moving_geoms if model.geom_group[g] == 3][-1]

    # --- Get gripperframe ---
    ee_pos = data.site_xpos[ee_site_id].copy()
    ee_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    # --- Get mesh vertices in world frame (subsample for speed) ---
    fixed_verts = _get_mesh_vertices_world(model, data, fixed_jaw_geom_id)[::5]
    moving_verts = _get_mesh_vertices_world(model, data, moving_jaw_geom_id)[::5]

    print("=== Gripperframe site ===")
    print(f"  World pos: [{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}]")

    # --- Method 1: geom-centroid midpoint (coarse, includes mounting structure) ---
    fixed_center = data.geom_xpos[fixed_jaw_geom_id].copy()
    moving_center = data.geom_xpos[moving_jaw_geom_id].copy()
    centroid_midpoint = (fixed_center + moving_center) / 2.0
    centroid_offset = ee_rot.T @ (centroid_midpoint - ee_pos)
    print("\n=== Method 1: geom-centroid midpoint (COARSE) ===")
    print(f"  Offset (local): [{centroid_offset[0]:.4f}, {centroid_offset[1]:.4f}, {centroid_offset[2]:.4f}]")
    print(f"  Magnitude: {np.linalg.norm(centroid_offset) * 1000:.1f} mm")
    print("  WARNING: includes jaw mounting structure, NOT just contact surface")

    # --- Method 2: vertex-proximity contact zone centroid (precise) ---
    print("\n=== Method 2: vertex-proximity contact zone ===")
    for threshold_mm in [3, 5, 10]:
        threshold = threshold_mm / 1000.0
        pairs = []
        for fv in fixed_verts:
            dists = np.linalg.norm(moving_verts - fv, axis=1)
            close = np.where(dists < threshold)[0]
            for j in close:
                pairs.append((fv + moving_verts[j]) / 2.0)

        if not pairs:
            print(f"  Threshold {threshold_mm}mm: no pairs found")
            continue

        contact_center = np.mean(pairs, axis=0)
        offset_local = ee_rot.T @ (contact_center - ee_pos)
        mag = np.linalg.norm(offset_local) * 1000
        print(
            f"  Threshold {threshold_mm}mm: {len(pairs)} pairs, "
            f"offset [{offset_local[0]:.4f}, {offset_local[1]:.4f}, {offset_local[2]:.4f}] "
            f"({mag:.1f}mm)"
        )

    # --- Recommended offset (3mm threshold) ---
    pairs_3mm = []
    for fv in fixed_verts:
        dists = np.linalg.norm(moving_verts - fv, axis=1)
        close = np.where(dists < 0.003)[0]
        for j in close:
            pairs_3mm.append((fv + moving_verts[j]) / 2.0)

    if pairs_3mm:
        contact = np.mean(pairs_3mm, axis=0)
        off = ee_rot.T @ (contact - ee_pos)
        print("\n=== Recommended TCP offset (3mm contact zone) ===")
        print(f"  _TCP_PINCH_OFFSET_LOCAL = np.array([{off[0]:.3f}, {off[1]:.3f}, {off[2]:.3f}])")
        print(f"  Magnitude: {np.linalg.norm(off) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
