from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandEnvelope,
    DescribeScenePayload,
    OverrideTargetPayload,
    StartSkillPayload,
    TrackObjectPayload,
)
from halo.contracts.enums import CommandType, SkillName


@dataclass
class AgentContext:
    arm_id: str
    snapshot_id: str | None
    commands: list[CommandEnvelope] = field(default_factory=list)
    used_tools: set[str] = field(default_factory=set)
    epoch: int | None = None


def build_tools(ctx: AgentContext) -> list:
    """Build tool list that close over ctx. ADK introspects name/signature/docstring."""

    def _once(name: str) -> str | None:
        """Return an error string if tool was already called this tick, else mark it used."""
        if name in ctx.used_tools:
            return f"REJECTED: {name} already called this tick. Wait for the next tick."
        ctx.used_tools.add(name)
        return None

    def start_skill(skill_name: str, target_handle: str, options: str = "") -> str:
        """Start a named skill on the arm.

        Args:
            skill_name: Skill to run. One of: PICK.
            target_handle: Target object handle string (from perception).
            options: Optional JSON string of key/value overrides for the skill.
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

    def override_target(skill_run_id: str, target_handle: str) -> str:
        """Override the target for the currently running skill.

        Args:
            skill_run_id: ID of the skill run to update.
            target_handle: New target object handle.
        """
        if err := _once("override_target"):
            return err
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.OVERRIDE_TARGET,
            payload=OverrideTargetPayload(
                skill_run_id=skill_run_id,
                target_handle=target_handle,
            ),
            precondition_snapshot_id=ctx.snapshot_id,
            epoch=ctx.epoch,
        )
        ctx.commands.append(cmd)
        return f"Queued OVERRIDE_TARGET run_id={skill_run_id} target={target_handle}"

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

    def track_object(target_handle: str) -> str:
        """Tell perception to start tracking a named object.

        Use a handle from SCENE_DESCRIBED detections. Perception will run
        VLM to locate the object and begin tracking it. A TARGET_ACQUIRED
        event fires once tracking is established.

        Args:
            target_handle: Object handle string (from SCENE_DESCRIBED detections).
        """
        if err := _once("track_object"):
            return err
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.TRACK_OBJECT,
            payload=TrackObjectPayload(
                target_handle=target_handle,
            ),
            precondition_snapshot_id=None,
            epoch=ctx.epoch,
        )
        ctx.commands.append(cmd)
        return f"Queued TRACK_OBJECT target={target_handle}"

    return [start_skill, abort_skill, override_target, describe_scene, track_object]
