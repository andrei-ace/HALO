"""Scene constants extracted from the MuJoCo model at runtime.

Provides a single source of truth for cube half-sizes, cube default position,
table surface height, TCP pinch offset, and MuJoCo entity names.  Values are
read from the compiled ``MjModel`` so they stay in sync with ``pick_scene.xml``
without manual duplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# MuJoCo entity names (match pick_scene.xml)
# ---------------------------------------------------------------------------

CUBE_GEOM_NAME = "cube_geom"
CUBE_BODY_NAME = "cube"
TABLE_BODY_NAME = "table"
TABLE_GEOM_NAME = "table_top"
EE_SITE_NAME = "gripperframe"

# ---------------------------------------------------------------------------
# TCP pinch-point offset — single canonical definition
# ---------------------------------------------------------------------------
# Offset from gripperframe site to jaw contact-surface centroid,
# expressed in gripperframe-local coordinates.
# Measured via mujoco_sim.scripts.measure_pinch_offset (vertex proximity method,
# centroid of jaw mesh vertices within 3 mm when gripper closed).
TCP_PINCH_OFFSET_LOCAL = np.array([-0.003, 0.0, 0.003])


# ---------------------------------------------------------------------------
# SceneInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneInfo:
    """Scene geometry constants read from a compiled MuJoCo model.

    Use ``SceneInfo.from_model(model)`` to construct.
    """

    cube_half_sizes: np.ndarray  # (3,) from model.geom_size[cube_geom_id]
    cube_default_pos: np.ndarray  # (3,) from model.body_pos[cube_body_id]
    table_z: float  # table body Z + table_top geom half-height

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> SceneInfo:
        """Extract scene constants from a compiled MuJoCo model.

        Args:
            model: Compiled MjModel (e.g. from ``pick_scene.xml``).

        Returns:
            Frozen SceneInfo with cube geometry, default position, and table height.
        """
        cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM_NAME)
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
        table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME)

        cube_half_sizes = model.geom_size[cube_geom_id].copy()
        cube_default_pos = model.body_pos[cube_body_id].copy()

        # Table surface = table body Z + table_top geom half-height (Z component of box size)
        table_z = float(model.body_pos[table_body_id][2] + model.geom_size[table_geom_id][2])

        return cls(
            cube_half_sizes=cube_half_sizes,
            cube_default_pos=cube_default_pos,
            table_z=table_z,
        )
