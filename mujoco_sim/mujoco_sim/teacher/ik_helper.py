"""Damped least-squares IK: desired EE position → 5 arm joint angles."""

from __future__ import annotations

import mujoco
import numpy as np


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    ee_site_id: int,
    arm_joint_ids: list[int],
    *,
    max_iters: int = 100,
    tol: float = 1e-3,
    damping: float = 1e-2,
) -> np.ndarray:
    """Solve position-only IK for the arm joints using damped least-squares.

    Operates on a *copy* of ``data`` to avoid mutating live sim state.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (will NOT be mutated — we work on a copy).
        target_pos: (3,) desired EE position in world frame.
        ee_site_id: MuJoCo site id for the end-effector (gripperframe).
        arm_joint_ids: List of 5 arm joint qpos indices (excludes gripper).
        max_iters: Maximum solver iterations.
        tol: Position error tolerance (metres).
        damping: Damping coefficient (λ² in (JᵀJ + λ²I)⁻¹Jᵀe).

    Returns:
        (5,) arm joint angles that place the EE near ``target_pos``.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, d)

    jac_pos = np.zeros((3, model.nv))

    for _ in range(max_iters):
        ee_pos = d.site_xpos[ee_site_id].copy()
        err = target_pos - ee_pos

        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(model, d, jac_pos, None, ee_site_id)

        # Extract columns for arm joints only (dof indices = joint ids for 1-DOF hinges)
        J = jac_pos[:, arm_joint_ids]  # (3, n_joints)

        # Damped least-squares: dq = Jᵀ(JJᵀ + λ²I)⁻¹ e
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        for i, jid in enumerate(arm_joint_ids):
            d.qpos[jid] += dq[i]
            # Clamp to joint limits
            lo, hi = model.jnt_range[jid]
            d.qpos[jid] = np.clip(d.qpos[jid], lo, hi)

        mujoco.mj_forward(model, d)

    return np.array([d.qpos[jid] for jid in arm_joint_ids], dtype=np.float64)
