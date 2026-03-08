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
    TRACK = "TRACK"
    PLACE = "PLACE"


class PlaceModifier(StrEnum):
    PLACE_FLOOR = "PLACE_FLOOR"
    PLACE_NEXT_TO = "PLACE_NEXT_TO"
    PLACE_IN_TRAY = "PLACE_IN_TRAY"


class PhaseId(IntEnum):
    # -- PICK sub-phases --
    IDLE = 0
    SELECT_GRASP = 1
    PLAN_APPROACH = 2
    MOVE_PREGRASP = 3
    VISUAL_ALIGN = 4
    EXECUTE_APPROACH = 5
    CLOSE_GRIPPER = 6
    LIFT = 7
    VERIFY_GRASP = 8
    DONE = 9
    # -- TRACK sub-phases --
    ACQUIRING = 20
    # -- PLACE sub-phases --
    TRANSIT_PREPLACE = 30
    DESCEND_PLACE = 31
    OPEN = 32
    RETREAT = 33
    SELECT_PLACE = 34
    # -- Recovery --
    RECOVER_RETRY_APPROACH = 50
    RECOVER_REGRASP = 51
    RECOVER_ABORT = 52
    # -- Transient --
    RETURNING = 60


# Phases where the wrist camera provides real frames (vs. black/masked).
# Shared contract consumed by FsmEngine handlers, ControlService, and sim constants.
WRIST_ACTIVE_PHASES: frozenset[PhaseId] = frozenset(
    {
        PhaseId.VISUAL_ALIGN,
        PhaseId.EXECUTE_APPROACH,
        PhaseId.CLOSE_GRIPPER,
        PhaseId.VERIFY_GRASP,
        PhaseId.LIFT,
        PhaseId.DESCEND_PLACE,
        PhaseId.OPEN,
        PhaseId.RETREAT,
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
    PERCEPTION_LOST = "PERCEPTION_LOST"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    UNSAFE_ABORT = "UNSAFE_ABORT"
    PLANNER_ABORT = "PLANNER_ABORT"


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


class CommandAckStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REJECTED_STALE = "REJECTED_STALE"
    REJECTED_WRONG_SKILL_RUN = "REJECTED_WRONG_SKILL_RUN"
    REJECTED_WRONG_EPOCH = "REJECTED_WRONG_EPOCH"
    ALREADY_APPLIED = "ALREADY_APPLIED"
