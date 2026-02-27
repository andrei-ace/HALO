"""Minimal spawn + wiggle test — verify Franka Panda loads in Isaac Lab.

Usage:
    python -m halo_sim.scripts.test_robot_spawn

Note: Requires Isaac Lab runtime.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def test_spawn() -> None:
    """Spawn Franka Panda, wiggle joints, verify no errors.

    Steps:
    1. Create Isaac Lab simulation context
    2. Spawn Franka Panda at origin
    3. Set joint targets to default positions
    4. Step physics 100 times
    5. Read joint positions
    6. Verify they match targets within tolerance
    """
    # Stub — requires Isaac Lab runtime
    #
    # from isaaclab.sim import SimulationContext
    # from isaaclab_assets import FRANKA_PANDA_CFG
    #
    # sim = SimulationContext(physics_dt=0.02)
    # robot = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    # robot.spawn(robot.prim_path, robot.spawn_cfg)
    # sim.reset()
    #
    # for _ in range(100):
    #     sim.step()
    #
    # joint_pos = robot.get_joint_pos()
    # logger.info("Joint positions: %s", joint_pos)
    # sim.close()

    logger.info("test_robot_spawn: stub (requires Isaac Lab runtime)")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_spawn()


if __name__ == "__main__":
    main()
