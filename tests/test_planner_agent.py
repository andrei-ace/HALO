"""Unit tests for PlannerService tools (AgentContext + build_tools).

The LangGraph agent itself (PlannerAgent / make_decide_fn) requires a running
Ollama instance and is not unit-tested here.
"""
from __future__ import annotations

from halo.contracts.enums import CommandType
from halo.services.planner_service.tools import AgentContext, build_tools


def _make_ctx(arm_id: str = "arm0", snapshot_id: str | None = "snap-1") -> AgentContext:
    return AgentContext(arm_id=arm_id, snapshot_id=snapshot_id)


def _tools_by_name(ctx: AgentContext) -> dict:
    return {t.name: t for t in build_tools(ctx)}


# ---------------------------------------------------------------------------
# start_skill
# ---------------------------------------------------------------------------


def test_start_skill_tool_appends_command() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["start_skill"].invoke({"skill_name": "PICK", "target_handle": "cube-1"})

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
    tools = _tools_by_name(ctx)
    tools["abort_skill"].invoke({"skill_run_id": "run-42", "reason": "timeout"})

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.ABORT_SKILL
    assert cmd.payload.skill_run_id == "run-42"
    assert cmd.payload.reason == "timeout"


# ---------------------------------------------------------------------------
# override_target
# ---------------------------------------------------------------------------


def test_override_target_tool_appends_command() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["override_target"].invoke({"skill_run_id": "run-7", "target_handle": "mug-2"})

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.OVERRIDE_TARGET
    assert cmd.payload.skill_run_id == "run-7"
    assert cmd.payload.target_handle == "mug-2"


# ---------------------------------------------------------------------------
# describe_scene
# ---------------------------------------------------------------------------


def test_describe_scene_tool_appends_command() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["describe_scene"].invoke({"reason": "lost target"})

    assert len(ctx.commands) == 1
    cmd = ctx.commands[0]
    assert cmd.type == CommandType.DESCRIBE_SCENE
    assert cmd.payload.reason == "lost target"


# ---------------------------------------------------------------------------
# arm_id propagation
# ---------------------------------------------------------------------------


def test_tool_commands_have_correct_arm_id() -> None:
    ctx = _make_ctx(arm_id="arm0")
    tools = _tools_by_name(ctx)
    tools["start_skill"].invoke({"skill_name": "PICK", "target_handle": "cube-1"})
    tools["abort_skill"].invoke({"skill_run_id": "run-1", "reason": "test"})
    tools["override_target"].invoke({"skill_run_id": "run-1", "target_handle": "box-2"})
    tools["describe_scene"].invoke({"reason": ""})

    for cmd in ctx.commands:
        assert cmd.arm_id == "arm0"


# ---------------------------------------------------------------------------
# unique command IDs
# ---------------------------------------------------------------------------


def test_tool_commands_have_unique_ids() -> None:
    ctx = _make_ctx()
    tools = _tools_by_name(ctx)
    tools["start_skill"].invoke({"skill_name": "PICK", "target_handle": "cube-1"})
    tools["start_skill"].invoke({"skill_name": "PICK", "target_handle": "cube-2"})

    assert len(ctx.commands) == 2
    ids = {cmd.command_id for cmd in ctx.commands}
    assert len(ids) == 2, "Both commands must have distinct UUIDs"
