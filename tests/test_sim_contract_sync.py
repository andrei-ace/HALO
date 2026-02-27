"""Contract sync verification — ensures sim/halo_sim/constants.py stays in sync with halo.contracts.

Loads sim constants via importlib.util (no Isaac Lab deps). Verifies all phase IDs,
action fields, gripper semantics, and wrist camera phases match.
"""

import importlib.util
from pathlib import Path

import pytest

from halo.contracts.actions import Action
from halo.contracts.enums import WRIST_ACTIVE_PHASES, PhaseId


@pytest.fixture(scope="module")
def sim_constants():
    """Load sim/halo_sim/constants.py without requiring sim dependencies."""
    constants_path = Path(__file__).parents[1] / "sim" / "halo_sim" / "constants.py"
    if not constants_path.exists():
        pytest.skip("sim/halo_sim/constants.py not found")
    spec = importlib.util.spec_from_file_location("halo_sim_constants", constants_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase_id_values_match(sim_constants):
    """Every PHASE_* in sim constants matches PhaseId[name].value."""
    expected = {
        "PHASE_IDLE": PhaseId.IDLE,
        "PHASE_SELECT_GRASP": PhaseId.SELECT_GRASP,
        "PHASE_PLAN_APPROACH": PhaseId.PLAN_APPROACH,
        "PHASE_MOVE_PREGRASP": PhaseId.MOVE_PREGRASP,
        "PHASE_VISUAL_ALIGN": PhaseId.VISUAL_ALIGN,
        "PHASE_EXECUTE_APPROACH": PhaseId.EXECUTE_APPROACH,
        "PHASE_CLOSE_GRIPPER": PhaseId.CLOSE_GRIPPER,
        "PHASE_VERIFY_GRASP": PhaseId.VERIFY_GRASP,
        "PHASE_LIFT": PhaseId.LIFT,
        "PHASE_DONE": PhaseId.DONE,
        "PHASE_RECOVER_RETRY_APPROACH": PhaseId.RECOVER_RETRY_APPROACH,
        "PHASE_RECOVER_REGRASP": PhaseId.RECOVER_REGRASP,
        "PHASE_RECOVER_ABORT": PhaseId.RECOVER_ABORT,
    }
    for const_name, phase_id in expected.items():
        sim_val = getattr(sim_constants, const_name)
        assert sim_val == int(phase_id), f"{const_name}: sim={sim_val} != halo={int(phase_id)}"


def test_action_fields_match(sim_constants):
    """ACTION_FIELDS matches Action.__dataclass_fields__ ordering."""
    action_fields = list(Action.__dataclass_fields__.keys())
    assert sim_constants.ACTION_FIELDS == action_fields


def test_action_dim_matches(sim_constants):
    """ACTION_DIM matches number of Action fields."""
    assert sim_constants.ACTION_DIM == len(Action.__dataclass_fields__)


def test_gripper_semantics_match(sim_constants):
    """GRIPPER_OPEN/CLOSE match Action contract (0.0=open, 1.0=close)."""
    assert sim_constants.GRIPPER_OPEN == 0.0
    assert sim_constants.GRIPPER_CLOSE == 1.0


def test_wrist_active_phases_match(sim_constants):
    """WRIST_ACTIVE_PHASES matches PickFSM._WRIST_ACTIVE_PHASES."""
    halo_phases = {int(p) for p in WRIST_ACTIVE_PHASES}
    sim_phases = sim_constants.WRIST_ACTIVE_PHASES
    assert sim_phases == halo_phases
