from enum import IntEnum, StrEnum


class PerceptionFailureCode(StrEnum):
    OK = "OK"
    OCCLUDED = "OCCLUDED"
    OUT_OF_VIEW = "OUT_OF_VIEW"
    DEPTH_INVALID = "DEPTH_INVALID"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    CALIB_INVALID = "CALIB_INVALID"
    TRACK_JUMP_REJECTED = "TRACK_JUMP_REJECTED"
    REACQUIRE_FAILED = "REACQUIRE_FAILED"


class TrackingStatus(StrEnum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RELOCALIZING = "RELOCALIZING"
    REACQUIRING = "REACQUIRING"


class SkillName(StrEnum):
    PICK = "PICK"
    PLACE = "PLACE"


class PhaseId(IntEnum):
    # -- PICK sub-phases --
    IDLE = 0
    SELECT_GRASP = 1
    PLAN_APPROACH = 2
    MOVE_PREGRASP = 3
    VISUAL_ALIGN = 4
    EXECUTE_APPROACH = 5
    CLOSE_GRIPPER = 6
    VERIFY_GRASP = 7
    LIFT = 8
    DONE = 9
    # -- PLACE sub-phases (reserved, not used in milestone 1) --
    TRANSIT_PREPLACE = 30
    DESCEND_PLACE = 31
    OPEN = 32
    RETREAT = 33
    # -- Recovery --
    RECOVER_RETRY_APPROACH = 50
    RECOVER_REGRASP = 51
    RECOVER_ABORT = 52


# Phases where the wrist camera provides real frames (vs. black/masked).
# Shared contract consumed by PickFSM, ControlService, and sim constants.
WRIST_ACTIVE_PHASES: frozenset[PhaseId] = frozenset(
    {
        PhaseId.VISUAL_ALIGN,
        PhaseId.EXECUTE_APPROACH,
        PhaseId.CLOSE_GRIPPER,
        PhaseId.VERIFY_GRASP,
        PhaseId.LIFT,
    }
)


class SkillOutcomeState(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNCERTAIN = "UNCERTAIN"


class SkillFailureCode(StrEnum):
    TIMEOUT = "TIMEOUT"
    NO_PROGRESS = "NO_PROGRESS"
    NO_GRASP = "NO_GRASP"
    DROP_DETECTED = "DROP_DETECTED"
    PLACE_MISS = "PLACE_MISS"
    UNSAFE_ABORT = "UNSAFE_ABORT"


class SafetyReflexReason(StrEnum):
    JOINT_LIMIT = "JOINT_LIMIT"
    WORKSPACE_LIMIT = "WORKSPACE_LIMIT"
    COLLISION_RISK = "COLLISION_RISK"
    OVERCURRENT = "OVERCURRENT"
    ESTOP = "ESTOP"


class SafetyState(StrEnum):
    OK = "OK"
    FAULT = "FAULT"


class ActStatus(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    BUFFER_LOW = "BUFFER_LOW"
    STALE = "STALE"


class CommandType(StrEnum):
    START_SKILL = "START_SKILL"
    ABORT_SKILL = "ABORT_SKILL"
    OVERRIDE_TARGET = "OVERRIDE_TARGET"
    DESCRIBE_SCENE = "DESCRIBE_SCENE"
    TRACK_OBJECT = "TRACK_OBJECT"


class CommandAckStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED_STALE = "REJECTED_STALE"
    REJECTED_WRONG_SKILL_RUN = "REJECTED_WRONG_SKILL_RUN"
    ALREADY_APPLIED = "ALREADY_APPLIED"
