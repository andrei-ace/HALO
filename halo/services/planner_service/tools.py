from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandEnvelope,
    DescribeScenePayload,
    StartSkillPayload,
)
from halo.contracts.enums import CommandType, SkillName

MAX_TOOL_CALLS = 10


@dataclass
class AgentContext:
    arm_id: str
    snapshot_id: str | None
    commands: list[CommandEnvelope] = field(default_factory=list)
    used_tools: set[tuple] = field(default_factory=set)
    epoch: int | None = None
    loop_detected: bool = False
    total_calls: int = 0


def build_tools(ctx: AgentContext) -> list:
    """Build tool list that close over ctx. ADK introspects name/signature/docstring."""

    def _once(name: str, *args: str) -> str | None:
        """Return an error string if this exact (name, args) was already called, else mark used.

        Deduplicates on the full (name, *args) tuple so the same tool with
        different arguments is allowed (e.g. start_skill(TRACK, X) then
        start_skill(PICK, X)).

        A global cap of 10 total tool calls per tick allows multi-object
        workflows (2 objects × TRACK+PICK+TRACK+PLACE = 8, plus spare for
        describe_scene).
        """
        ctx.total_calls += 1
        if ctx.total_calls > MAX_TOOL_CALLS:
            ctx.loop_detected = True
            return (
                f"HARD STOP: {MAX_TOOL_CALLS} tool calls reached this tick. "
                "Stop calling tools and respond with your reasoning."
            )
        key = (name, *args)
        if key in ctx.used_tools:
            return f"REJECTED: {name} already called with these arguments this tick. Wait for the next tick."
        ctx.used_tools.add(key)
        return None

    def start_skill(skill_name: str, target_handle: str, options: str = "") -> str:
        """Start a named skill on the arm.

        Args:
            skill_name: Skill to run. One of: PICK, TRACK, PLACE.
                For PLACE, target_handle is the reference object handle.
                Use options to specify the modifier (PLACE_FLOOR, PLACE_NEXT_TO, or PLACE_IN_TRAY).
            target_handle: Target object handle string (from perception).
                For PLACE_FLOOR: the held object handle.
                For PLACE_NEXT_TO: the reference object handle to place next to.
                For PLACE_IN_TRAY: the tray handle.
            options: Optional JSON string of key/value overrides for the skill.
                For PLACE: '{"modifier": "PLACE_FLOOR"}', '{"modifier": "PLACE_NEXT_TO"}',
                or '{"modifier": "PLACE_IN_TRAY"}'.
        """
        if err := _once("start_skill", skill_name, target_handle):
            return err
        try:
            skill = SkillName(skill_name)
        except Exception:
            allowed = ", ".join(s.value for s in SkillName)
            return f"REJECTED: invalid skill_name {skill_name!r}. Expected one of [{allowed}]."
        opts: dict = {}
        if options:
            import json

            try:
                opts = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                pass
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.START_SKILL,
            payload=StartSkillPayload(
                skill_name=skill,
                target_handle=target_handle,
                options=opts,
            ),
            precondition_snapshot_id=ctx.snapshot_id,
            epoch=ctx.epoch,
        )
        ctx.commands.append(cmd)
        return f"Queued START_SKILL {skill_name} target={target_handle}"

    def abort_skill(skill_run_id: str, reason: str) -> str:
        """Abort the currently running skill.

        Args:
            skill_run_id: ID of the skill run to abort (from snapshot.skill.skill_run_id).
            reason: Human-readable reason for aborting.
        """
        if err := _once("abort_skill", skill_run_id):
            return err
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.ABORT_SKILL,
            payload=AbortSkillPayload(
                skill_run_id=skill_run_id,
                reason=reason,
            ),
            precondition_snapshot_id=ctx.snapshot_id,
            epoch=ctx.epoch,
        )
        ctx.commands.append(cmd)
        return f"Queued ABORT_SKILL run_id={skill_run_id} reason={reason}"

    def describe_scene(reason: str = "") -> str:
        """Ask TargetPerceptionService to run VLM scene analysis.

        Triggers a full VLM pass that describes the scene and returns
        bounding boxes for all detected objects.  The result is delivered
        asynchronously via a SCENE_DESCRIBED event.

        Args:
            reason: Human-readable reason for requesting the scene description.
        """
        if err := _once("describe_scene"):
            return err
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.DESCRIBE_SCENE,
            payload=DescribeScenePayload(
                reason=reason,
            ),
            # No precondition: scene description is a stateless side-effect.
            # Pinning it to a snapshot_id causes REJECTED_STALE as soon as the
            # snapshot advances between decide() and submit().
            precondition_snapshot_id=None,
            epoch=ctx.epoch,
        )
        ctx.commands.append(cmd)
        return f"Queued DESCRIBE_SCENE reason={reason}"

    return [start_skill, abort_skill, describe_scene]
