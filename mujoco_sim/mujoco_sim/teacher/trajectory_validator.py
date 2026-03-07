"""FK-based trajectory clearance validation.

Samples each trajectory segment at 5 points and checks that the gripper
stays above the table surface with sufficient margin.
"""

from __future__ import annotations

import logging

import mujoco

from mujoco_sim.teacher.trajectory import TrajectoryPlan

logger = logging.getLogger(__name__)


def validate_trajectory_clearance(
    plan: TrajectoryPlan,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    arm_joint_ids: list[int],
    table_z: float,
    margin: float = 0.01,
) -> bool:
    """Return True if all sampled points have sufficient clearance above the table.

    For each segment, samples at t=0, t/4, t/2, 3t/4, t and runs FK to check
    that the EE Z position stays above ``table_z + margin``.

    Args:
        plan: Planned trajectory to validate.
        model: MuJoCo model.
        data: MuJoCo data (not mutated — used as template for qpos).
        ee_site_id: MuJoCo site id for gripperframe.
        arm_joint_ids: List of 5 arm joint IDs.
        table_z: Table surface height (Z).
        margin: Minimum clearance above table (metres).

    Returns:
        True if all sampled points clear the threshold.
    """
    threshold = table_z + margin
    qpos_idx = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    d_fk = mujoco.MjData(model)

    for seg in plan.segments:
        if seg.duration < 1e-9:
            continue
        sample_times = [0.0, seg.duration * 0.25, seg.duration * 0.5, seg.duration * 0.75, seg.duration]
        for t in sample_times:
            arm_joints, _gripper = seg.sample(t)

            d_fk.qpos[:] = data.qpos[:]
            for i, jid in enumerate(arm_joint_ids):
                d_fk.qpos[qpos_idx[i]] = arm_joints[i]
            mujoco.mj_forward(model, d_fk)

            ee_z = float(d_fk.site_xpos[ee_site_id][2])
            if ee_z < threshold:
                logger.info(
                    "Clearance violation in segment '%s' at t=%.3f: ee_z=%.4f < threshold=%.4f",
                    seg.label,
                    t,
                    ee_z,
                    threshold,
                )
                return False

    return True
