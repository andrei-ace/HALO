"""Local constants sync test — verifies mujoco_sim.constants matches expected HALO contract values.

This is the local counterpart of tests/test_mujoco_sim_contract_sync.py (which runs
from the HALO repo root and loads constants via importlib).
"""

from mujoco_sim.constants import (
    ACTION_DIM,
    ACTION_FIELDS,
    GRIPPER_CLOSE,
    GRIPPER_JOINT_NAME,
    GRIPPER_OPEN,
    PHASE_CLOSE_GRIPPER,
    PHASE_DESCEND_PLACE,
    PHASE_DONE,
    PHASE_EXECUTE_APPROACH,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_MOVE_PREGRASP,
    PHASE_OPEN,
    PHASE_PLAN_APPROACH,
    PHASE_RECOVER_ABORT,
    PHASE_RECOVER_REGRASP,
    PHASE_RECOVER_RETRY_APPROACH,
    PHASE_RETREAT,
    PHASE_SELECT_GRASP,
    PHASE_SELECT_PLACE,
    PHASE_TRANSIT_PREPLACE,
    PHASE_VERIFY_GRASP,
    PHASE_VISUAL_ALIGN,
    SO101_ARM_JOINT_NAMES,
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
        "TRANSIT_PREPLACE": (PHASE_TRANSIT_PREPLACE, 30),
        "DESCEND_PLACE": (PHASE_DESCEND_PLACE, 31),
        "OPEN": (PHASE_OPEN, 32),
        "RETREAT": (PHASE_RETREAT, 33),
        "SELECT_PLACE": (PHASE_SELECT_PLACE, 34),
        "RECOVER_RETRY_APPROACH": (PHASE_RECOVER_RETRY_APPROACH, 50),
        "RECOVER_REGRASP": (PHASE_RECOVER_REGRASP, 51),
        "RECOVER_ABORT": (PHASE_RECOVER_ABORT, 52),
    }
    for name, (actual, expected_val) in expected.items():
        assert actual == expected_val, f"PHASE_{name}: got {actual}, expected {expected_val}"


def test_action_fields():
    """ACTION_FIELDS matches expected 6D joint-position field names."""
    expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    assert ACTION_FIELDS == expected


def test_action_dim():
    """ACTION_DIM matches len(ACTION_FIELDS)."""
    assert ACTION_DIM == len(ACTION_FIELDS)
    assert ACTION_DIM == 6


def test_gripper_semantics():
    """GRIPPER_OPEN=1.75, GRIPPER_CLOSE=-0.17 (SO-101 joint angle, rad)."""
    assert GRIPPER_OPEN == 1.75
    assert GRIPPER_CLOSE == -0.17


def test_wrist_active_phases():
    """WRIST_ACTIVE_PHASES contains exactly the expected phases."""
    expected = frozenset(
        {
            PHASE_VISUAL_ALIGN,
            PHASE_EXECUTE_APPROACH,
            PHASE_CLOSE_GRIPPER,
            PHASE_VERIFY_GRASP,
            PHASE_LIFT,
            PHASE_DESCEND_PLACE,
            PHASE_OPEN,
            PHASE_RETREAT,
        }
    )
    assert WRIST_ACTIVE_PHASES == expected


def test_so101_arm_joint_names():
    """SO101_ARM_JOINT_NAMES has 5 arm joints, GRIPPER_JOINT_NAME is 'gripper'."""
    assert len(SO101_ARM_JOINT_NAMES) == 5
    assert SO101_ARM_JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    assert GRIPPER_JOINT_NAME == "gripper"
    # All action fields = arm joints + gripper
    assert list(SO101_ARM_JOINT_NAMES) + [GRIPPER_JOINT_NAME] == ACTION_FIELDS
