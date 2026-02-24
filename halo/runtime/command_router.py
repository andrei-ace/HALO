from __future__ import annotations

import asyncio
import time

from halo.contracts.commands import (
    AbortSkillPayload,
    CommandAck,
    CommandEnvelope,
    OverrideTargetPayload,
)
from halo.contracts.enums import CommandAckStatus
from halo.contracts.events import EventEnvelope, EventType
from halo.runtime.state_store import RuntimeStateStore


class CommandRouter:
    """
    Validates and dispatches planner commands.

    Enforces three checks in order:

    1. Idempotency — duplicate command_id → ALREADY_APPLIED (accepted commands
       are remembered; the same command_id always returns ALREADY_APPLIED).
    2. Precondition — if precondition_snapshot_id is set and does not match the
       current snapshot → REJECTED_STALE.
    3. Skill-run match — for ABORT_SKILL / OVERRIDE_TARGET, the payload's
       skill_run_id must match the current skill → REJECTED_WRONG_SKILL_RUN.

    Rejected commands are not cached; they may be retried with a new command_id
    once the caller has corrected the precondition.
    """

    def __init__(self, store: RuntimeStateStore, bus: object | None = None) -> None:
        self._store = store
        self._bus = bus  # EventBus | None
        self._accepted: set[str] = set()  # command_ids that were accepted
        self._lock = asyncio.Lock()

    async def submit(self, cmd: CommandEnvelope) -> CommandAck:
        async with self._lock:
            # --- 1. Idempotency ---
            if cmd.command_id in self._accepted:
                return CommandAck(
                    command_id=cmd.command_id,
                    status=CommandAckStatus.ALREADY_APPLIED,
                )

            # --- Fetch latest snapshot once if any check needs it ---
            needs_snapshot = cmd.precondition_snapshot_id is not None or isinstance(
                cmd.payload, (AbortSkillPayload, OverrideTargetPayload)
            )
            latest = await self._store.get_latest_snapshot(cmd.arm_id) if needs_snapshot else None

            # --- 2. Precondition check ---
            if cmd.precondition_snapshot_id is not None:
                current_id = latest.snapshot_id if latest else None
                if current_id != cmd.precondition_snapshot_id:
                    return await self._reject(
                        cmd,
                        CommandAckStatus.REJECTED_STALE,
                        f"expected snapshot {cmd.precondition_snapshot_id!r}, "
                        f"current is {current_id!r}",
                    )

            # --- 3. Skill-run match (abort / override only) ---
            if isinstance(cmd.payload, (AbortSkillPayload, OverrideTargetPayload)):
                current_skill = latest.skill if latest else None
                current_run_id = current_skill.skill_run_id if current_skill else None
                if current_run_id != cmd.payload.skill_run_id:
                    return await self._reject(
                        cmd,
                        CommandAckStatus.REJECTED_WRONG_SKILL_RUN,
                        f"expected skill_run_id {cmd.payload.skill_run_id!r}, "
                        f"current is {current_run_id!r}",
                    )

            # --- Accept ---
            ack = CommandAck(command_id=cmd.command_id, status=CommandAckStatus.ACCEPTED)
            self._accepted.add(cmd.command_id)
            await self._store.add_command_ack(cmd.arm_id, ack)
            await self._publish(cmd, EventType.COMMAND_ACCEPTED)
            return ack

    async def _reject(
        self, cmd: CommandEnvelope, status: CommandAckStatus, reason: str
    ) -> CommandAck:
        ack = CommandAck(command_id=cmd.command_id, status=status, reason=reason)
        await self._store.add_command_ack(cmd.arm_id, ack)
        await self._publish(cmd, EventType.COMMAND_REJECTED)
        return ack

    async def _publish(self, cmd: CommandEnvelope, event_type: EventType) -> None:
        if self._bus is None:
            return
        evt = EventEnvelope(
            event_id=self._bus.make_event_id(),  # type: ignore[union-attr]
            type=event_type,
            ts_ms=int(time.time() * 1000),
            arm_id=cmd.arm_id,
            data={"command_id": cmd.command_id, "command_type": str(cmd.type)},
        )
        if hasattr(cmd.payload, "target_handle"):
            evt = EventEnvelope(
                event_id=evt.event_id,
                type=evt.type,
                ts_ms=evt.ts_ms,
                arm_id=evt.arm_id,
                data={**evt.data, "target_handle": cmd.payload.target_handle},
            )
        await self._bus.publish(evt)  # type: ignore[union-attr]
