from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langchain_core.tools import tool

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandEnvelope,
    OverrideTargetPayload,
    RequestPerceptionRefreshPayload,
    StartSkillPayload,
)
from halo.contracts.enums import CommandType, SkillName


@dataclass
class AgentContext:
    arm_id: str
    snapshot_id: str | None
    commands: list[CommandEnvelope] = field(default_factory=list)


def build_tools(ctx: AgentContext) -> list:
    """Build LangChain tool list that close over ctx."""

    @tool
    def start_skill(skill_name: str, target_handle: str, options: dict = {}) -> str:
        """Start a named skill on the arm.

        Args:
            skill_name: Skill to run. One of: PICK, PLACE.
            target_handle: Target object handle string (from perception).
            options: Optional key/value overrides for the skill.
        """
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.START_SKILL,
            payload=StartSkillPayload(
                skill_name=SkillName(skill_name),
                target_handle=target_handle,
                options=options,
            ),
            precondition_snapshot_id=ctx.snapshot_id,
        )
        ctx.commands.append(cmd)
        return f"Queued START_SKILL {skill_name} target={target_handle}"

    @tool
    def abort_skill(skill_run_id: str, reason: str) -> str:
        """Abort the currently running skill.

        Args:
            skill_run_id: ID of the skill run to abort (from snapshot.skill.skill_run_id).
            reason: Human-readable reason for aborting.
        """
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
        )
        ctx.commands.append(cmd)
        return f"Queued ABORT_SKILL run_id={skill_run_id} reason={reason}"

    @tool
    def override_target(skill_run_id: str, target_handle: str) -> str:
        """Override the target for the currently running skill.

        Args:
            skill_run_id: ID of the skill run to update.
            target_handle: New target object handle.
        """
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
        )
        ctx.commands.append(cmd)
        return f"Queued OVERRIDE_TARGET run_id={skill_run_id} target={target_handle}"

    @tool
    def request_perception_refresh(mode: str = "reacquire", reason: str = "") -> str:
        """Ask TargetPerceptionService to re-run VLM localisation.

        Args:
            mode: Refresh mode. Use "reacquire" to trigger a full VLM pass.
            reason: Human-readable reason for the refresh request.
        """
        cmd = CommandEnvelope(
            command_id=str(uuid.uuid4()),
            arm_id=ctx.arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.REQUEST_PERCEPTION_REFRESH,
            payload=RequestPerceptionRefreshPayload(
                mode=mode,
                reason=reason,
            ),
            precondition_snapshot_id=ctx.snapshot_id,
        )
        ctx.commands.append(cmd)
        return f"Queued REQUEST_PERCEPTION_REFRESH mode={mode}"

    return [start_skill, abort_skill, override_target, request_perception_refresh]
