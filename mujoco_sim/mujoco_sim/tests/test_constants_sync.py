"""Local constants sync test — verifies mujoco_sim.constants matches expected HALO contract values.

This is the local counterpart of tests/test_mujoco_sim_contract_sync.py (which runs
from the HALO repo root and loads constants via importlib).
"""

from mujoco_sim.constants import (
    ACTION_DIM,
    ACTION_FIELDS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_DONE,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
    PHASE_PLAN_APPROACH,
    PHASE_RECOVER_ABORT,
    PHASE_RECOVER_REGRASP,
    PHASE_RECOVER_RETRY_APPROACH,
    PHASE_SELECT_GRASP,
    PHASE_VERIFY_GRASP,
    PHASE_VISUAL_ALIGN,
    WRIST_ACTIVE_PHASES,
)


def test_phase_id_values():
    """Phase constants match expected integer values from halo.contracts.enums.PhaseId."""
    expected = {
        "IDLE": (PHASE_IDLE, 0),
        "SELECT_GRASP": (PHASE_SELECT_GRASP, 1),
        "PLAN_APPROACH": (PHASE_PLAN_APPROACH, 2),
        "MOVE_PREGRASP": (PHASE_MOVE_PREGRASP, 3),
        "VISUAL_ALIGN": (PHASE_VISUAL_ALIGN, 4),
        "EXECUTE_APPROACH": (PHASE_EXECUTE_APPROACH, 5),
        "CLOSE_GRIPPER": (PHASE_CLOSE_GRIPPER, 6),
        "VERIFY_GRASP": (PHASE_VERIFY_GRASP, 7),
        "LIFT": (PHASE_LIFT, 8),
        "DONE": (PHASE_DONE, 9),
        "RECOVER_RETRY_APPROACH": (PHASE_RECOVER_RETRY_APPROACH, 50),
        "RECOVER_REGRASP": (PHASE_RECOVER_REGRASP, 51),
        "RECOVER_ABORT": (PHASE_RECOVER_ABORT, 52),
    }
    for name, (actual, expected_val) in expected.items():
        assert actual == expected_val, f"PHASE_{name}: got {actual}, expected {expected_val}"


def test_action_fields():
    """ACTION_FIELDS matches expected 7-DOF EE-delta field names."""
    expected = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_cmd"]
    assert ACTION_FIELDS == expected


def test_action_dim():
    """ACTION_DIM matches len(ACTION_FIELDS)."""
    assert ACTION_DIM == len(ACTION_FIELDS)
    assert ACTION_DIM == 7


def test_gripper_semantics():
    """GRIPPER_OPEN=-1.0, GRIPPER_CLOSE=1.0 (robosuite convention)."""
    assert GRIPPER_OPEN == -1.0
    assert GRIPPER_CLOSE == 1.0


def test_wrist_active_phases():
    """WRIST_ACTIVE_PHASES contains exactly the expected phases."""
    expected = frozenset(
        {PHASE_VISUAL_ALIGN, PHASE_EXECUTE_APPROACH, PHASE_CLOSE_GRIPPER, PHASE_VERIFY_GRASP, PHASE_LIFT}
    )
    assert WRIST_ACTIVE_PHASES == expected
