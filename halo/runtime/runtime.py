from __future__ import annotations

from halo.contracts.commands import CommandAck, CommandEnvelope
from halo.contracts.snapshots import PlannerSnapshot
from halo.runtime.command_router import CommandRouter
from halo.runtime.event_bus import EventBus
from halo.runtime.state_store import RuntimeStateStore


class HALORuntime:
    """
    Top-level runtime object. Wires together RuntimeStateStore, EventBus,
    and CommandRouter, and exposes the two planner-facing APIs:

        snap = await rt.get_latest_runtime_snapshot("arm0")
        ack  = await rt.submit_command(cmd)

    Services update state directly via rt.store and rt.bus:

        await rt.store.update_skill("arm0", skill_info)
        await rt.bus.publish(event)
        q = rt.bus.subscribe("arm0")
    """

    def __init__(self) -> None:
        self.store = RuntimeStateStore()
        self.bus = EventBus()
        self.router = CommandRouter(self.store, self.bus)

    def register_arm(self, arm_id: str) -> None:
        """Register an arm. Must be called before any arm-specific operations."""
        self.store.register_arm(arm_id)

    async def get_latest_runtime_snapshot(self, arm_id: str) -> PlannerSnapshot:
        """
        The single planner-facing read API.

        Builds a fresh PlannerSnapshot from the current store state plus the
        recent-events ring from the EventBus. The result is cached and *replaces*
        (never appends) the previous snapshot, so the planner always sees exactly
        one snapshot — the latest.
        """
        recent = self.bus.get_recent_events(arm_id)
        return await self.store.build_and_cache_snapshot(arm_id, recent)

    async def submit_command(self, cmd: CommandEnvelope) -> CommandAck:
        """
        Submit a planner command. Returns a CommandAck immediately.

        Enforces in order:
          - Idempotency (duplicate command_id → ALREADY_APPLIED)
          - Precondition (stale snapshot_id → REJECTED_STALE)
          - Skill-run match for abort/override (REJECTED_WRONG_SKILL_RUN)
        """
        return await self.router.submit(cmd)
