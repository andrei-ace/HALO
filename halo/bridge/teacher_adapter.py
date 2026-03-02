"""Bridge adapter: wraps SimClient.teacher_step() into an async TeacherStepFn."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from halo.services.skill_runner_service.service import TeacherStepFn, TeacherStepResult

if TYPE_CHECKING:
    from halo.bridge.sim_client import SimClient


def make_teacher_step_fn(client: SimClient) -> TeacherStepFn:
    """Create an async TeacherStepFn from a SimClient instance.

    The underlying ``client.teacher_step()`` is synchronous (blocking ZMQ);
    we run it in the default executor to avoid blocking the event loop.
    """

    async def teacher_step_fn(arm_id: str) -> TeacherStepResult:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, client.teacher_step)
        return TeacherStepResult(
            phase_id=resp["phase_id"],
            done=resp["done"],
            action=tuple(float(v) for v in resp["action"]),
        )

    return teacher_step_fn
