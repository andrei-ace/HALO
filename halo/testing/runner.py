"""HeadlessRunner — orchestrates HALO services without a TUI.

Replaces the TUI as the service wiring layer for tests and CLI usage.
Creates a HALORuntime, builds enabled services, and manages their lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from halo.contracts.actions import Action, ActionChunk
from halo.contracts.commands import CommandEnvelope, StartSkillPayload
from halo.contracts.enums import CommandType, PhaseId
from halo.contracts.events import EventEnvelope, EventType
from halo.contracts.snapshots import PlannerSnapshot
from halo.runtime.runtime import HALORuntime
from halo.services.control_service.config import ControlServiceConfig
from halo.services.control_service.service import ControlService
from halo.services.planner_service.config import PlannerServiceConfig
from halo.services.planner_service.service import PlannerService
from halo.services.skill_runner_service.config import SkillRunnerConfig
from halo.services.skill_runner_service.service import AbortPickFn, SimPhaseFn, SkillRunnerService, StartPickFn
from halo.services.target_perception_service.config import TargetPerceptionServiceConfig
from halo.services.target_perception_service.service import TargetPerceptionService
from halo.testing.event_recorder import EventRecorder

logger = logging.getLogger(__name__)

# Type aliases matching service constructor signatures
DecideFn = Callable[[PlannerSnapshot], Awaitable[list]]
ApplyFn = Callable[[str, Action], Awaitable[None]]
ChunkFn = Callable[[str, PhaseId, object], Awaitable[ActionChunk | None]]
PushFn = Callable[[ActionChunk], Awaitable[None]]
ObserveFn = Callable[[str, str], Awaitable[object | None]]
VlmFn = Callable[..., Awaitable[object]]
CaptureFn = Callable[[str], Awaitable[object]]
TrackerFactoryFn = Callable[..., Awaitable[tuple]]


@dataclass
class RunnerConfig:
    """Configuration for HeadlessRunner."""

    arm_id: str = "arm0"
    max_duration_s: float = 60.0

    # Service enable flags — set False to omit a service for component tests
    enable_planner: bool = True
    enable_perception: bool = True
    enable_skill_runner: bool = True
    enable_control: bool = True

    # Per-service config overrides (None → use defaults)
    planner_config: PlannerServiceConfig = field(default_factory=PlannerServiceConfig)
    perception_config: TargetPerceptionServiceConfig = field(default_factory=TargetPerceptionServiceConfig)
    skill_runner_config: SkillRunnerConfig = field(default_factory=SkillRunnerConfig)
    control_config: ControlServiceConfig = field(default_factory=ControlServiceConfig)


class HeadlessRunner:
    """Headless HALO service orchestrator for tests and CLI.

    Creates a HALORuntime, registers an arm, builds enabled services
    with the provided callables, and manages start/stop lifecycle.

    When both SkillRunner and Control are enabled and no explicit ``push_fn``
    is provided, auto-wires ``push_fn → control_svc.push_chunk``.

    Services are exposed as public attributes for direct test access:
    ``runtime``, ``planner_svc``, ``perception_svc``, ``skill_runner_svc``,
    ``control_svc``, ``recorder``.
    """

    def __init__(
        self,
        config: RunnerConfig = RunnerConfig(),
        *,
        # Planner
        decide_fn: DecideFn | None = None,
        # Perception
        observe_fn: ObserveFn | None = None,
        vlm_fn: VlmFn | None = None,
        capture_fn: CaptureFn | None = None,
        tracker_factory_fn: TrackerFactoryFn | None = None,
        # SkillRunner (ACT mode)
        chunk_fn: ChunkFn | None = None,
        push_fn: PushFn | None = None,
        # SkillRunner (Sim mode)
        start_pick_fn: StartPickFn | None = None,
        abort_pick_fn: AbortPickFn | None = None,
        sim_phase_fn: SimPhaseFn | None = None,
        # Control
        apply_fn: ApplyFn | None = None,
    ) -> None:
        self._config = config
        arm_id = config.arm_id

        # Build runtime
        self.runtime = HALORuntime()
        self.runtime.register_arm(arm_id)

        # EventRecorder — always active
        self.recorder = EventRecorder(self.runtime.bus, arm_id)

        # Build services based on enable flags
        self.control_svc: ControlService | None = None
        self.skill_runner_svc: SkillRunnerService | None = None
        self.perception_svc: TargetPerceptionService | None = None
        self.planner_svc: PlannerService | None = None

        # Detect sim mode
        _sim_mode = start_pick_fn is not None

        # Control service (build first so push_fn can reference it)
        if config.enable_control and not _sim_mode:
            if apply_fn is None:
                raise ValueError("apply_fn is required when enable_control=True")
            self.control_svc = ControlService(
                arm_id=arm_id,
                runtime=self.runtime,
                apply_fn=apply_fn,
                config=config.control_config,
            )

        # SkillRunner — ACT or sim mode
        if config.enable_skill_runner:
            if _sim_mode:
                self.skill_runner_svc = SkillRunnerService(
                    arm_id=arm_id,
                    runtime=self.runtime,
                    config=config.skill_runner_config,
                    start_pick_fn=start_pick_fn,
                    abort_pick_fn=abort_pick_fn,
                    sim_phase_fn=sim_phase_fn,
                )
            else:
                if chunk_fn is None:
                    raise ValueError("chunk_fn is required when enable_skill_runner=True (ACT mode)")
                effective_push_fn = push_fn
                if effective_push_fn is None and self.control_svc is not None:
                    effective_push_fn = self.control_svc.push_chunk
                if effective_push_fn is None:
                    raise ValueError("push_fn is required when enable_skill_runner=True and enable_control=False")
                self.skill_runner_svc = SkillRunnerService(
                    arm_id=arm_id,
                    runtime=self.runtime,
                    chunk_fn=chunk_fn,
                    push_fn=effective_push_fn,
                    config=config.skill_runner_config,
                )

        # Perception
        if config.enable_perception:
            self.perception_svc = TargetPerceptionService(
                arm_id=arm_id,
                runtime=self.runtime,
                observe_fn=observe_fn,
                vlm_fn=vlm_fn,
                capture_fn=capture_fn,
                tracker_factory_fn=tracker_factory_fn,
                config=config.perception_config,
            )

        # Planner
        if config.enable_planner:
            if decide_fn is None:
                raise ValueError("decide_fn is required when enable_planner=True")
            self.planner_svc = PlannerService(
                arm_id=arm_id,
                runtime=self.runtime,
                decide_fn=decide_fn,
                config=config.planner_config,
            )

        self._running = False
        self._cmd_route_task: asyncio.Task | None = None
        self._cmd_route_queue: asyncio.Queue | None = None
        # Store submitted commands by id for routing after acceptance
        self._pending_commands: dict[str, CommandEnvelope] = {}
        # Wrap submit_command to intercept commands
        self._orig_submit = self.runtime.submit_command
        self.runtime.submit_command = self._intercepted_submit  # type: ignore[assignment]

    async def _intercepted_submit(self, cmd: CommandEnvelope):
        """Intercept submit_command to capture commands for routing."""
        self._pending_commands[cmd.command_id] = cmd
        return await self._orig_submit(cmd)

    @property
    def arm_id(self) -> str:
        return self._config.arm_id

    # -- command routing ---------------------------------------------------

    async def _route_commands(self) -> None:
        """Listen for COMMAND_ACCEPTED/REJECTED events and route to services.

        The CommandRouter only validates + acks commands; this task dispatches
        accepted START_SKILL/ABORT_SKILL commands to the SkillRunnerService.
        TargetPerceptionService already listens for DESCRIBE_SCENE and SKILL_STARTED(TRACK).

        Rejected commands are also popped from ``_pending_commands`` to prevent
        unbounded growth in long-running sessions.
        """
        assert self._cmd_route_queue is not None
        while True:
            event: EventEnvelope = await self._cmd_route_queue.get()

            if event.type == EventType.COMMAND_REJECTED:
                cmd_id = event.data.get("command_id")
                if cmd_id:
                    self._pending_commands.pop(cmd_id, None)
                continue

            if event.type != EventType.COMMAND_ACCEPTED:
                continue

            cmd_id = event.data.get("command_id")
            if not cmd_id:
                continue
            cmd = self._pending_commands.pop(cmd_id, None)
            if cmd is None:
                continue

            if cmd.type == CommandType.START_SKILL and self.skill_runner_svc is not None:
                payload = cmd.payload
                assert isinstance(payload, StartSkillPayload)
                await self.skill_runner_svc.start_skill(
                    skill_name=payload.skill_name,
                    skill_run_id=f"run-{cmd.command_id[:8]}",
                    target_handle=payload.target_handle,
                )
            elif cmd.type == CommandType.ABORT_SKILL and self.skill_runner_svc is not None:
                await self.skill_runner_svc.abort_skill()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start all enabled services and the event recorder.

        Start order: recorder → control → skill_runner → perception → planner.
        Control starts first so push_fn is ready; planner starts last so it
        sees a fully wired system.
        """
        if self._running:
            return
        self._running = True

        await self.recorder.start()

        if self.control_svc is not None:
            await self.control_svc.start()
        if self.skill_runner_svc is not None:
            await self.skill_runner_svc.start()
        if self.perception_svc is not None:
            await self.perception_svc.start()

        # Start command routing before planner (so we catch its startup command).
        # Always start — even without skill runner — to drain _pending_commands
        # and prevent unbounded growth from DESCRIBE_SCENE commands.
        self._cmd_route_queue = self.runtime.bus.subscribe(self._config.arm_id, maxsize=0)
        self._cmd_route_task = asyncio.create_task(self._route_commands(), name="cmd-route")

        if self.planner_svc is not None:
            await self.planner_svc.start()

        logger.info("HeadlessRunner started (arm=%s)", self._config.arm_id)

    async def stop(self) -> None:
        """Stop all services gracefully.

        Stop order: planner → perception → skill_runner → control → recorder.
        Reverse of start — planner stops first to prevent new commands.
        """
        if not self._running:
            return
        self._running = False

        if self.planner_svc is not None:
            await self.planner_svc.stop()

        # Stop command routing after planner
        if self._cmd_route_task is not None:
            self._cmd_route_task.cancel()
            try:
                await self._cmd_route_task
            except asyncio.CancelledError:
                pass
            self._cmd_route_task = None
        if self._cmd_route_queue is not None:
            self.runtime.bus.unsubscribe(self._config.arm_id, self._cmd_route_queue)
            self._cmd_route_queue = None
        self._pending_commands.clear()

        if self.perception_svc is not None:
            await self.perception_svc.stop()
        if self.skill_runner_svc is not None:
            await self.skill_runner_svc.stop()
        if self.control_svc is not None:
            await self.control_svc.stop()

        await self.recorder.stop()
        logger.info("HeadlessRunner stopped (arm=%s)", self._config.arm_id)

    async def run(self, until: Callable[[], bool] | None = None) -> None:
        """Start services, run until predicate or timeout, then stop.

        *until*: sync callable returning True to stop. Checked every 50ms.
        If None, runs for ``max_duration_s``.
        """
        await self.start()
        try:
            t0 = time.monotonic()
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= self._config.max_duration_s:
                    break
                if until is not None and until():
                    break
                await asyncio.sleep(0.05)
        finally:
            await self.stop()
