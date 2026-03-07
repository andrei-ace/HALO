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


@dataclass
class AgentContext:
    arm_id: str
    snapshot_id: str | None
    commands: list[CommandEnvelope] = field(default_factory=list)
    used_tools: set[str] = field(default_factory=set)
    epoch: int | None = None
    call_counts: dict[str, int] = field(default_factory=dict)
    loop_detected: bool = False


def build_tools(ctx: AgentContext) -> list:
    """Build tool list that close over ctx. ADK introspects name/signature/docstring."""

    def _once(name: str) -> str | None:
        """Return an error string if tool was already called this tick, else mark it used.

        Also tracks per-tool call counts within a tick. If any tool is called
        5+ times (including rejected attempts), sets loop_detected=True and
        returns a hard stop message.
        """
        ctx.call_counts[name] = ctx.call_counts.get(name, 0) + 1
        if ctx.call_counts[name] >= 5:
            ctx.loop_detected = True
            return (
                f"LOOP DETECTED: {name} called {ctx.call_counts[name]} times. "
                "Stop calling tools and respond with your reasoning."
            )
        if name in ctx.used_tools:
            return f"REJECTED: {name} already called this tick. Wait for the next tick."
        ctx.used_tools.add(name)
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
        if err := _once("start_skill"):
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
        if err := _once("abort_skill"):
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
