"""Visualize IK-solved poses by setting joint configurations and rendering snapshots.

Runs the grasp planner + keyframe planner + waypoint generator (IK), then for
each solved waypoint directly sets qpos and renders a scene camera image.
No physics stepping — purely static FK snapshots.

Usage::

    uv run python -m mujoco_sim.scripts.visualize_ik_pose
    uv run python -m mujoco_sim.scripts.visualize_ik_pose --seed 7 --resolution 1080 1440
    uv run python -m mujoco_sim.scripts.visualize_ik_pose -v  # verbose IK logging
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mujoco_sim.config.env_config import EnvConfig
from mujoco_sim.constants import SO101_ARM_JOINT_NAMES
from mujoco_sim.env.so101_env import SO101Env
from mujoco_sim.scene_info import TCP_PINCH_OFFSET_LOCAL, SceneInfo
from mujoco_sim.teacher.grasp_planner import evaluate_grasps
from mujoco_sim.teacher.keyframe_planner import plan_pick_keyframes
from mujoco_sim.teacher.waypoint_generator import generate_joint_waypoints

logger = logging.getLogger(__name__)


def _resolve_ids(model: mujoco.MjModel) -> tuple[int, list[int], int]:
    """Resolve MuJoCo IDs for EE site, arm joints, and gripper joint.

    Returns:
        (ee_site_id, arm_joint_ids, gripper_joint_id)
    """
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in SO101_ARM_JOINT_NAMES]
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    return ee_site_id, arm_joint_ids, gripper_joint_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IK-solved poses as rendered snapshots")
    parser.add_argument("--seed", type=int, default=7, help="Cube placement seed (default: 7)")
    parser.add_argument("--output-dir", type=str, default="data/ik_poses", help="Base output directory")
    parser.add_argument(
        "--resolution", type=int, nargs=2, metavar=("H", "W"), default=[480, 640], help="Render resolution"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose IK debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    h, w = args.resolution
    config = EnvConfig(scene_resolution=(h, w))
    env = SO101Env(config)
    model = env.mujoco_model
    data = env.mujoco_data

    # Reset with seed to place cube
    obs = env.reset(seed=args.seed)
    cube_pos = obs["object_pose"][:3]
    cube_quat = obs["object_pose"][3:]
    home_joints = obs["joint_pos"][:6].copy()

    print(f"Seed: {args.seed}")
    print(f"Cube position: [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]")
    print(f"Resolution: {h}x{w}")

    ee_site_id, arm_joint_ids, gripper_joint_id = _resolve_ids(model)
    scene = SceneInfo.from_model(model)

    # Evaluate grasp candidates and select best
    best = evaluate_grasps(
        cube_pos=cube_pos,
        cube_quat=cube_quat,
        cube_half_sizes=scene.cube_half_sizes,
        model=model,
        data=data,
        ee_site_id=ee_site_id,
        arm_joint_ids=arm_joint_ids,
        seed_joints=home_joints[:5],
        table_z=scene.table_z,
        best_effort=True,
    )
    print(
        f"\nSelected grasp: face={best.grasp.face_label} yaw={best.grasp.yaw_variant}"
        f" tilt={best.grasp.tilt_deg:.0f}°"
        f" score={best.score:.3f} pos_err={best.ik_pos_err:.4f}m ori_err={best.ori_err_deg:.1f}°"
    )

    # Plan keyframes from selected grasp
    keyframes = plan_pick_keyframes(
        home_joints=home_joints,
        grasp_pose=best.grasp,
        ee_site_id=ee_site_id,
        model=model,
        data=data,
    )
    print(f"Planned {len(keyframes)} keyframes: {[kf.label for kf in keyframes]}")

    # Solve IK
    waypoints = generate_joint_waypoints(
        keyframes=keyframes,
        model=model,
        data=data,
        ee_site_id=ee_site_id,
        arm_joint_ids=arm_joint_ids,
        seed_joints=home_joints[:5],
    )
    print(f"Solved {len(waypoints)} IK waypoints")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render each waypoint
    print(f"\nRendering to {out_dir}/\n")
    print(
        f"{'Label':<15} {'Target pos':>30} {'Achieved pos':>30} {'Err (mm)':>10}"
        f" {'Pinch pos':>30} {'Pinch→Cube (mm)':>16}"
    )
    print("-" * 140)

    for kf, wp in zip(keyframes, waypoints):
        # Set arm joints
        for i, jid in enumerate(arm_joint_ids):
            data.qpos[jid] = wp.arm_joints[i]
        # Set gripper
        data.qpos[gripper_joint_id] = wp.gripper
        # FK (no physics)
        mujoco.mj_forward(model, data)

        # Read achieved EE position
        achieved_pos = data.site_xpos[ee_site_id].copy()
        achieved_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()
        pos_err = np.linalg.norm(kf.position - achieved_pos) * 1000  # mm

        # Compute jaw midpoint (pinch point) from achieved pose
        pinch_world = achieved_pos + achieved_rot @ TCP_PINCH_OFFSET_LOCAL
        pinch_to_cube = np.linalg.norm(pinch_world - cube_pos) * 1000  # mm

        # Render and save
        img = env.render()
        img_path = out_dir / f"{wp.label}.png"
        Image.fromarray(img).save(img_path)

        # Print diagnostics
        target_str = f"[{kf.position[0]:7.4f}, {kf.position[1]:7.4f}, {kf.position[2]:7.4f}]"
        achieved_str = f"[{achieved_pos[0]:7.4f}, {achieved_pos[1]:7.4f}, {achieved_pos[2]:7.4f}]"
        pinch_str = f"[{pinch_world[0]:7.4f}, {pinch_world[1]:7.4f}, {pinch_world[2]:7.4f}]"
        print(
            f"{wp.label:<15} {target_str:>30} {achieved_str:>30} {pos_err:>9.2f} {pinch_str:>30} {pinch_to_cube:>15.2f}"
        )

    env.close()
    print(f"\nDone. Images saved to {out_dir}/")


if __name__ == "__main__":
    main()
