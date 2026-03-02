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

GREEN_CUBE_GEOM_NAME = "green_cube_geom"
GREEN_CUBE_BODY_NAME = "green_cube"
RED_CUBE_GEOM_NAME = "red_cube_geom"
RED_CUBE_BODY_NAME = "red_cube"
TABLE_BODY_NAME = "table"
TABLE_GEOM_NAME = "table_top"
EE_SITE_NAME = "gripperframe"

# ---------------------------------------------------------------------------
# TCP pinch-point offset — single canonical definition
# ---------------------------------------------------------------------------
# Offset from gripperframe site to jaw contact-surface centroid,
# expressed in gripperframe-local coordinates.
# Set to zero: the gripperframe site is close enough to the jaw contact
# centroid that compensating for the offset hurts more than it helps
# (IK error + approach-angle projection dominate the small 3-4 mm offset).
TCP_PINCH_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Grasp planner defaults — single canonical definitions
# ---------------------------------------------------------------------------

# Number of random grasp candidates to sample (split evenly across 4 side faces).
DEFAULT_GRASP_N_CANDIDATES = 64

# Maximum angular deviation from face normal for approach direction (degrees).
DEFAULT_GRASP_MAX_CONE_DEG = 5.0

# Tangential contact span on cube side faces (0.0 = center only, 1.0 = full face).
DEFAULT_CUBE_FACE_CONTACT_SPAN = 0.10

# Distance (metres) to offset the grasp contact point outward along the face
# normal.  Compensates for the jaw midpoint being ahead of the gripperframe
# site, preventing the jaws from overshooting past the cube.
DEFAULT_FACE_STANDOFF = 0.003  # 3 mm (matches jaw tip overshoot past gripperframe)

# Approximate depth of gripper body behind the contact point (for table collision filter).
DEFAULT_GRIPPER_DEPTH = 0.10  # 100 mm

# Safety margin above table for geometric feasibility filter.
DEFAULT_TABLE_MARGIN = 0.01  # 10 mm

# ---------------------------------------------------------------------------
# Teacher / trajectory defaults — single canonical definitions
# ---------------------------------------------------------------------------

# Pregrasp standoff: distance (m) along approach direction above grasp contact.
DEFAULT_PREGRASP_STANDOFF = 0.08

# Lift height (m) above grasp contact point.
DEFAULT_LIFT_HEIGHT = 0.08

# Maximum IK orientation error allowed (degrees). 5-DOF arm has ~17-34° typical.
DEFAULT_ORI_TOL_DEG = 55.0

# IK solver parameters
DEFAULT_IK_POS_WEIGHT = 1.0
DEFAULT_IK_ORI_WEIGHT = 0.1
DEFAULT_IK_MAX_ITERS = 200
DEFAULT_IK_TOL = 1e-3
DEFAULT_IK_POS_TOL = 0.03  # metres


# ---------------------------------------------------------------------------
# SceneInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneInfo:
    """Scene geometry constants read from a compiled MuJoCo model.

    Use ``SceneInfo.from_model(model)`` to construct.
    """

    green_cube_half_sizes: np.ndarray  # (3,) from model.geom_size[green_cube_geom_id]
    green_cube_default_pos: np.ndarray  # (3,) from model.body_pos[green_cube_body_id]
    red_cube_half_sizes: np.ndarray  # (3,) from model.geom_size[red_cube_geom_id]
    red_cube_default_pos: np.ndarray  # (3,) from model.body_pos[red_cube_body_id]
    table_z: float  # table body Z + table_top geom half-height

    def half_sizes_for_body(self, body_name: str) -> np.ndarray:
        """Look up cube half-sizes by body name.

        Raises:
            KeyError: If *body_name* is not a known cube body.
        """
        mapping = {
            GREEN_CUBE_BODY_NAME: self.green_cube_half_sizes,
            RED_CUBE_BODY_NAME: self.red_cube_half_sizes,
        }
        if body_name not in mapping:
            raise KeyError(f"Unknown cube body: {body_name!r}. Known: {list(mapping)}")
        return mapping[body_name]

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> SceneInfo:
        """Extract scene constants from a compiled MuJoCo model.

        Args:
            model: Compiled MjModel (e.g. from ``pick_scene.xml``).

        Returns:
            Frozen SceneInfo with cube geometry, default position, and table height.
        """
        green_cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GREEN_CUBE_GEOM_NAME)
        green_cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GREEN_CUBE_BODY_NAME)
        red_cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RED_CUBE_GEOM_NAME)
        red_cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RED_CUBE_BODY_NAME)
        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
        table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME)

        green_cube_half_sizes = model.geom_size[green_cube_geom_id].copy()
        green_cube_default_pos = model.body_pos[green_cube_body_id].copy()
        red_cube_half_sizes = model.geom_size[red_cube_geom_id].copy()
        red_cube_default_pos = model.body_pos[red_cube_body_id].copy()

        # Table surface = table body Z + table_top geom half-height (Z component of box size)
        table_z = float(model.body_pos[table_body_id][2] + model.geom_size[table_geom_id][2])

        return cls(
            green_cube_half_sizes=green_cube_half_sizes,
            green_cube_default_pos=green_cube_default_pos,
            red_cube_half_sizes=red_cube_half_sizes,
            red_cube_default_pos=red_cube_default_pos,
            table_z=table_z,
        )
