"""Unit tests for PlannerService tools (AgentContext + build_tools)
and loop-detection helper.

The ADK agent itself (PlannerAgent / make_decide_fn) requires a running
Ollama instance and is not unit-tested here.
"""

from __future__ import annotations

import time
import uuid

from halo.contracts.commands import CommandEnvelope, StartSkillPayload
from halo.contracts.enums import CommandType, SkillName
from halo.services.planner_service.agent import _command_key
from halo.services.planner_service.tools import AgentContext, build_tools


def _make_ctx(arm_id: str = "arm0", snapshot_id: str | None = "snap-1") -> AgentContext:
    return AgentContext(arm_id=arm_id, snapshot_id=snapshot_id)


def _tools_by_name(ctx: AgentContext) -> dict:
    return {fn.__name__: fn for fn in build_tools(ctx)}


# ---------------------------------------------------------------------------
# start_skill
# ---------------------------------------------------------------------------


def test_start_skill_tool_appends_command() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["start_skill"](skill_name="PICK", target_handle="cube-1")

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.START_SKILL
    assert cmd.payload.skill_name == "PICK"
    assert cmd.payload.target_handle == "cube-1"


# ---------------------------------------------------------------------------
# abort_skill
# ---------------------------------------------------------------------------


def test_abort_skill_tool_appends_command() -> None:
    ctx = _make_ctx()
    ctx.active_skill_run_id = "run-42"
    ctx.skill_outcome_state = "IN_PROGRESS"
    tools = _tools_by_name(ctx)
    tools["abort_skill"](skill_run_id="run-42", reason="timeout")

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.ABORT_SKILL
    assert cmd.payload.skill_run_id == "run-42"
    assert cmd.payload.reason == "timeout"


def test_abort_skill_rejected_when_not_in_progress() -> None:
    """abort_skill is rejected when outcome is not IN_PROGRESS."""
    ctx = _make_ctx()
    ctx.active_skill_run_id = "run-42"
    ctx.skill_outcome_state = "SUCCESS"
    tools = _tools_by_name(ctx)
    result = tools["abort_skill"](skill_run_id="run-42", reason="stale")

    assert "REJECTED" in result
    assert "SUCCESS" in result
    assert len(ctx.commands) == 0


def test_abort_skill_rejected_when_idle() -> None:
    """abort_skill is rejected when no skill is running."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    result = tools["abort_skill"](skill_run_id="run-42", reason="stale")

    assert "REJECTED" in result
    assert "idle" in result
    assert len(ctx.commands) == 0


def test_abort_skill_rejected_on_run_id_mismatch() -> None:
    """abort_skill is rejected when skill_run_id doesn't match the snapshot."""
    ctx = _make_ctx()
    ctx.active_skill_run_id = "run-42"
    ctx.skill_outcome_state = "IN_PROGRESS"
    tools = _tools_by_name(ctx)
    result = tools["abort_skill"](skill_run_id="run-WRONG", reason="mismatch")

    assert "REJECTED" in result
    assert "mismatch" in result
    assert len(ctx.commands) == 0


# ---------------------------------------------------------------------------
# describe_scene
# ---------------------------------------------------------------------------


def test_describe_scene_tool_appends_command() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["describe_scene"](reason="lost target")

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.DESCRIBE_SCENE
    assert cmd.payload.reason == "lost target"


# ---------------------------------------------------------------------------
# arm_id propagation
# ---------------------------------------------------------------------------


def test_tool_commands_have_correct_arm_id() -> None:
    ctx = _make_ctx(arm_id="arm0")
    ctx.active_skill_run_id = "run-1"
    ctx.skill_outcome_state = "IN_PROGRESS"
    tools = _tools_by_name(ctx)
    tools["start_skill"](skill_name="PICK", target_handle="cube-1")
    tools["abort_skill"](skill_run_id="run-1", reason="test")
    tools["describe_scene"](reason="")

    for cmd in ctx.commands:
        assert cmd.arm_id == "arm0"


# ---------------------------------------------------------------------------
# unique command IDs
# ---------------------------------------------------------------------------


def test_duplicate_tool_call_rejected() -> None:
    """Calling the same tool with the same args twice is rejected."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["start_skill"](skill_name="PICK", target_handle="cube-1")
    result2 = tools["start_skill"](skill_name="PICK", target_handle="cube-1")

    assert len(ctx.commands) == 1, "Only the first call should produce a command"
    assert "REJECTED" in result2


def test_start_skill_different_args_allowed() -> None:
    """start_skill(TRACK, X) then start_skill(PICK, X) both succeed — different args."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    r1 = tools["start_skill"](skill_name="TRACK", target_handle="cube-1")
    r2 = tools["start_skill"](skill_name="PICK", target_handle="cube-1")

    assert "REJECTED" not in r1
    assert "REJECTED" not in r2
    assert len(ctx.commands) == 2


def test_start_skill_same_args_rejected() -> None:
    """start_skill(TRACK, X) then start_skill(TRACK, X) again — second rejected."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    r1 = tools["start_skill"](skill_name="TRACK", target_handle="cube-1")
    r2 = tools["start_skill"](skill_name="TRACK", target_handle="cube-1")

    assert "REJECTED" not in r1
    assert "REJECTED" in r2
    assert len(ctx.commands) == 1


def test_different_tools_allowed_same_tick() -> None:
    """Different tools can each be called once in the same tick."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["describe_scene"](reason="check")
    tools["start_skill"](skill_name="PICK", target_handle="cube-1")

    assert len(ctx.commands) == 2


def test_global_call_cap() -> None:
    """After MAX_TOOL_CALLS (10) tool calls, further calls are rejected."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    # 2-object workflow: (TRACK+PICK+TRACK+PLACE)*2 = 8, + describe_scene = 9
    for obj_i in range(2):
        tools["start_skill"](skill_name="TRACK", target_handle=f"obj-{obj_i}")
        tools["start_skill"](skill_name="PICK", target_handle=f"obj-{obj_i}")
        tools["start_skill"](skill_name="TRACK", target_handle=f"tray-{obj_i}")
        tools["start_skill"](skill_name="PLACE", target_handle=f"tray-{obj_i}")
    tools["describe_scene"](reason="check")
    # One more unique call to reach 10
    tools["start_skill"](skill_name="TRACK", target_handle="obj-extra")
    assert len(ctx.commands) == 10
    assert ctx.total_calls == 10
    # 11th call should be hard-stopped
    result = tools["start_skill"](skill_name="TRACK", target_handle="obj-99")
    assert "HARD STOP" in result
    assert len(ctx.commands) == 10
    assert ctx.loop_detected is True


def test_start_skill_track_appends_command() -> None:
    """start_skill(TRACK, ...) creates a START_SKILL command."""
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["start_skill"](skill_name="TRACK", target_handle="mug-2")

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.START_SKILL
    assert cmd.payload.skill_name == SkillName.TRACK
    assert cmd.payload.target_handle == "mug-2"


# ---------------------------------------------------------------------------
# _command_key (loop detection)
# ---------------------------------------------------------------------------


def _cmd(cmd_type: CommandType = CommandType.START_SKILL, **payload_kw) -> CommandEnvelope:
    """Build a CommandEnvelope with unique id/ts but deterministic payload."""
    payload = StartSkillPayload(
        skill_name=payload_kw.get("skill_name", SkillName.PICK),
        target_handle=payload_kw.get("target_handle", "cube-1"),
    )
    return CommandEnvelope(
        command_id=str(uuid.uuid4()),
        arm_id="arm0",
        issued_at_ms=int(time.time() * 1000),
        type=cmd_type,
        payload=payload,
    )


def test_command_key_stable_across_ids() -> None:
    """Same type+payload → same key despite different command_id/ts."""
    a = _cmd()
    b = _cmd()
    assert a.command_id != b.command_id
    assert _command_key(a) == _command_key(b)


def test_command_key_differs_for_different_payloads() -> None:
    a = _cmd(target_handle="cube-1")
    b = _cmd(target_handle="cube-2")
    assert _command_key(a) != _command_key(b)


def test_command_key_differs_for_different_types() -> None:
    a = _cmd(CommandType.START_SKILL, skill_name=SkillName.PICK, target_handle="cube-1")
    b = _cmd(CommandType.START_SKILL, skill_name=SkillName.TRACK, target_handle="cube-1")
    assert _command_key(a) != _command_key(b)
