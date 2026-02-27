"""Coordinate frame transforms for the HALO <-> sim bridge."""

from __future__ import annotations


def world_to_ee_frame(
    world_delta: list[float],
    ee_quat: list[float],
) -> list[float]:
    """Rotate a world-frame 3D vector into the end-effector frame.

    Parameters
    ----------
    world_delta : [dx, dy, dz] in world frame
    ee_quat : [w, x, y, z] unit quaternion (world → EE orientation)

    Returns
    -------
    [dx, dy, dz] in the EE frame.

    Uses the inverse (conjugate) quaternion to rotate the world-frame
    vector into the EE body frame:  v_ee = q* ⊗ v_world ⊗ q
    """
    w, x, y, z = ee_quat

    # Rotation matrix from quaternion (transposed = inverse rotation)
    # R^T rotates world → body
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y + w * z)
    r02 = 2.0 * (x * z - w * y)
    r10 = 2.0 * (x * y - w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z + w * x)
    r20 = 2.0 * (x * z + w * y)
    r21 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    vx, vy, vz = world_delta
    return [
        r00 * vx + r01 * vy + r02 * vz,
        r10 * vx + r11 * vy + r12 * vz,
        r20 * vx + r21 * vy + r22 * vz,
    ]
