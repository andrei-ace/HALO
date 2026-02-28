"""Constants synced from halo.contracts. Verified by tests/test_sim_contract_sync.py."""

# PhaseId integer values (halo.contracts.enums.PhaseId)
PHASE_IDLE = 0
PHASE_SELECT_GRASP = 1
PHASE_PLAN_APPROACH = 2
PHASE_MOVE_PREGRASP = 3
PHASE_VISUAL_ALIGN = 4
PHASE_EXECUTE_APPROACH = 5
PHASE_CLOSE_GRIPPER = 6
PHASE_VERIFY_GRASP = 7
PHASE_LIFT = 8
PHASE_DONE = 9
PHASE_RECOVER_RETRY_APPROACH = 50
PHASE_RECOVER_REGRASP = 51
PHASE_RECOVER_ABORT = 52

# Wrist camera active phases
WRIST_ACTIVE_PHASES = frozenset(
    {
        PHASE_VISUAL_ALIGN,
        PHASE_EXECUTE_APPROACH,
        PHASE_CLOSE_GRIPPER,
        PHASE_VERIFY_GRASP,
        PHASE_LIFT,
    }
)

# Action space layout — 6D joint-position control (SO-101: 5 arm DOF + 1 gripper)
ACTION_FIELDS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ACTION_DIM = 6

# SO-101 joint names (5 DOF arm + 1 DOF gripper)
SO101_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME = "gripper"

# Gripper semantics (joint angle, not ±1 command)
GRIPPER_OPEN = -0.17  # fully open (joint range minimum, rad)
GRIPPER_CLOSE = 1.75  # fully closed (joint range maximum, rad)

# Control timing
CONTROL_RATE_HZ = 10
CHUNK_HORIZON_STEPS = 10
