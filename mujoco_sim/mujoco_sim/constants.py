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
    }
)

# Action space layout (halo.contracts.actions.Action field order)
ACTION_FIELDS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_cmd"]
ACTION_DIM = 7
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0

# Control timing
CONTROL_RATE_HZ = 10
CHUNK_HORIZON_STEPS = 10

# Panda arm joint names (7 DOF, excludes gripper fingers)
PANDA_JOINT_NAMES: tuple[str, ...] = (
    "J1_shoulder",
    "J2_shoulder",
    "J3_elbow",
    "J4_elbow",
    "J5_wrist",
    "J6_wrist",
    "J7_wrist",
)
