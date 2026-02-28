"""Contract sync verification — ensures mujoco_sim.constants stays in sync with halo.contracts.

Requires ``--extra sim`` (halo-mujoco-sim). Auto-skips if not installed.
"""

import pytest

from halo.contracts.actions import Action
from halo.contracts.enums import WRIST_ACTIVE_PHASES, PhaseId

try:
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
    )
    from mujoco_sim.constants import (
        WRIST_ACTIVE_PHASES as SIM_WRIST_ACTIVE_PHASES,
    )

    _has_mujoco_sim = True
except ImportError:
    _has_mujoco_sim = False

pytestmark = pytest.mark.skipif(not _has_mujoco_sim, reason="halo-mujoco-sim not installed (uv sync --extra sim)")


def test_phase_id_values_match():
    """Every PHASE_* in mujoco_sim constants matches PhaseId[name].value."""
    expected = {
        "PHASE_IDLE": (PHASE_IDLE, PhaseId.IDLE),
        "PHASE_SELECT_GRASP": (PHASE_SELECT_GRASP, PhaseId.SELECT_GRASP),
        "PHASE_PLAN_APPROACH": (PHASE_PLAN_APPROACH, PhaseId.PLAN_APPROACH),
        "PHASE_MOVE_PREGRASP": (PHASE_MOVE_PREGRASP, PhaseId.MOVE_PREGRASP),
        "PHASE_VISUAL_ALIGN": (PHASE_VISUAL_ALIGN, PhaseId.VISUAL_ALIGN),
        "PHASE_EXECUTE_APPROACH": (PHASE_EXECUTE_APPROACH, PhaseId.EXECUTE_APPROACH),
        "PHASE_CLOSE_GRIPPER": (PHASE_CLOSE_GRIPPER, PhaseId.CLOSE_GRIPPER),
        "PHASE_VERIFY_GRASP": (PHASE_VERIFY_GRASP, PhaseId.VERIFY_GRASP),
        "PHASE_LIFT": (PHASE_LIFT, PhaseId.LIFT),
        "PHASE_DONE": (PHASE_DONE, PhaseId.DONE),
        "PHASE_RECOVER_RETRY_APPROACH": (PHASE_RECOVER_RETRY_APPROACH, PhaseId.RECOVER_RETRY_APPROACH),
        "PHASE_RECOVER_REGRASP": (PHASE_RECOVER_REGRASP, PhaseId.RECOVER_REGRASP),
        "PHASE_RECOVER_ABORT": (PHASE_RECOVER_ABORT, PhaseId.RECOVER_ABORT),
    }
    for const_name, (sim_val, phase_id) in expected.items():
        assert sim_val == int(phase_id), f"{const_name}: mujoco_sim={sim_val} != halo={int(phase_id)}"


def test_action_fields_match():
    """ACTION_FIELDS matches Action.__dataclass_fields__ ordering."""
    action_fields = list(Action.__dataclass_fields__.keys())
    assert ACTION_FIELDS == action_fields


def test_action_dim_matches():
    """ACTION_DIM matches number of Action fields."""
    assert ACTION_DIM == len(Action.__dataclass_fields__)


def test_gripper_semantics_match():
    """GRIPPER_OPEN/CLOSE match Action contract (0.0=open, 1.0=close)."""
    assert GRIPPER_OPEN == 0.0
    assert GRIPPER_CLOSE == 1.0


def test_wrist_active_phases_match():
    """WRIST_ACTIVE_PHASES matches halo.contracts WRIST_ACTIVE_PHASES."""
    halo_phases = {int(p) for p in WRIST_ACTIVE_PHASES}
    assert SIM_WRIST_ACTIVE_PHASES == halo_phases
