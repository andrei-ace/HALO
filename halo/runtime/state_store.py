from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque

from halo.contracts.commands import CommandAck
from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    SafetyState,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.events import EventEnvelope
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    SkillInfo,
    TargetInfo,
)


def _default_perception() -> PerceptionInfo:
    return PerceptionInfo(
        tracking_status=TrackingStatus.IDLE,
        failure_code=PerceptionFailureCode.OK,
        reacquire_fail_count=0,
        vlm_job_pending=False,
    )


def _default_act() -> ActInfo:
    return ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False)


def _default_progress() -> ProgressInfo:
    return ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0)


def _default_outcome() -> OutcomeInfo:
    return OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False)


def _default_safety() -> SafetyInfo:
    return SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=())


class RuntimeStateStore:
    """
    Single source of truth for HALO runtime state, partitioned by arm_id.

    Backed by in-process asyncio.Lock-protected dicts. The public interface is
    stable so the backing store can be swapped to Redis or ZeroMQ without
    changing call sites.
    """

    COMMAND_ACK_RING_SIZE = 10

    def __init__(self) -> None:
        self._skill: dict[str, SkillInfo | None] = {}
        self._target: dict[str, TargetInfo | None] = {}
        self._perception: dict[str, PerceptionInfo] = {}
        self._act: dict[str, ActInfo] = {}
        self._progress: dict[str, ProgressInfo] = {}
        self._outcome: dict[str, OutcomeInfo] = {}
        self._safety: dict[str, SafetyInfo] = {}
        self._acks: dict[str, deque[CommandAck]] = {}
        self._latest_snapshot: dict[str, PlannerSnapshot | None] = {}
        self._snap_counter: dict[str, itertools.count] = {}
        self._lock = asyncio.Lock()

    def register_arm(self, arm_id: str) -> None:
        """Initialise per-arm state with safe defaults. Idempotent."""
        if arm_id in self._skill:
            return
        self._skill[arm_id] = None
        self._target[arm_id] = None
        self._perception[arm_id] = _default_perception()
        self._act[arm_id] = _default_act()
        self._progress[arm_id] = _default_progress()
        self._outcome[arm_id] = _default_outcome()
        self._safety[arm_id] = _default_safety()
        self._acks[arm_id] = deque(maxlen=self.COMMAND_ACK_RING_SIZE)
        self._latest_snapshot[arm_id] = None
        self._snap_counter[arm_id] = itertools.count(1)

    def _require_arm(self, arm_id: str) -> None:
        if arm_id not in self._skill:
            raise KeyError(f"arm_id '{arm_id}' not registered — call register_arm() first")

    # ------------------------------------------------------------------
    # Per-field update methods
    # ------------------------------------------------------------------

    async def update_skill(self, arm_id: str, skill: SkillInfo | None) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._skill[arm_id] = skill

    async def update_target(self, arm_id: str, target: TargetInfo | None) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._target[arm_id] = target

    async def update_perception(self, arm_id: str, perception: PerceptionInfo) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._perception[arm_id] = perception

    async def update_act(self, arm_id: str, act: ActInfo) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._act[arm_id] = act

    async def update_progress(self, arm_id: str, progress: ProgressInfo) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._progress[arm_id] = progress

    async def update_outcome(self, arm_id: str, outcome: OutcomeInfo) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._outcome[arm_id] = outcome

    async def update_safety(self, arm_id: str, safety: SafetyInfo) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._safety[arm_id] = safety

    async def add_command_ack(self, arm_id: str, ack: CommandAck) -> None:
        async with self._lock:
            self._require_arm(arm_id)
            self._acks[arm_id].append(ack)

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------

    async def build_and_cache_snapshot(
        self,
        arm_id: str,
        recent_events: list[EventEnvelope],
    ) -> PlannerSnapshot:
        """
        Assemble a PlannerSnapshot from the current per-arm state and cache it
        as the latest snapshot. Old snapshots are replaced, not appended.
        """
        async with self._lock:
            self._require_arm(arm_id)
            snap_num = next(self._snap_counter[arm_id])
            snapshot = PlannerSnapshot(
                snapshot_id=f"snap-{arm_id}-{snap_num}",
                ts_ms=int(time.time() * 1000),
                arm_id=arm_id,
                skill=self._skill[arm_id],
                target=self._target[arm_id],
                perception=self._perception[arm_id],
                act=self._act[arm_id],
                progress=self._progress[arm_id],
                outcome=self._outcome[arm_id],
                safety=self._safety[arm_id],
                command_acks=tuple(self._acks[arm_id]),
                recent_events=tuple(recent_events[-8:]),
            )
            self._latest_snapshot[arm_id] = snapshot
        return snapshot

    async def get_latest_snapshot(self, arm_id: str) -> PlannerSnapshot | None:
        async with self._lock:
            self._require_arm(arm_id)
            return self._latest_snapshot[arm_id]
