"""ChunkFn adapter: teacher generates action chunks (placeholder for ACT later).

For milestone 1: teacher IK generates chunks based on privileged sim state.
Receives privileged state (cube_pos) from sim via ZMQ.

Plugs into SkillRunnerService(arm_id, runtime, chunk_fn=make_sim_teacher_chunk_fn(), ...).
"""

from __future__ import annotations

import math
import uuid
from typing import Awaitable, Callable

import msgpack
import zmq

from halo.bridge.config import SimBridgeConfig
from halo.bridge.transforms import world_to_ee_frame
from halo.contracts.actions import Action, ActionChunk
from halo.contracts.enums import PhaseId

# Type alias matching SkillRunnerService.ChunkFn
ChunkFn = Callable[[str, PhaseId, object], Awaitable[ActionChunk | None]]


def make_sim_teacher_chunk_fn(
    config: SimBridgeConfig | None = None,
    chunk_horizon_steps: int = 10,
) -> ChunkFn:
    """Create a chunk_fn that generates teacher actions from privileged sim state.

    For milestone 1, this generates simple IK-based actions using the
    cube_pos received from sim observations. Later replaced by ACT model.
    """
    if config is None:
        config = SimBridgeConfig()

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(config.sim_obs_url)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout_ms)

    async def chunk_fn(arm_id: str, phase: PhaseId, snap: object) -> ActionChunk | None:
        # Get latest observation from sim
        try:
            raw = socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        msg = msgpack.unpackb(raw, raw=False)

        ee_pos = msg.get("ee_pos", [0.0, 0.0, 0.0])
        ee_quat = msg.get("ee_quat", [1.0, 0.0, 0.0, 0.0])
        cube_pos = msg.get("cube_pos", [0.0, 0.0, 0.0])

        # World-frame delta, then rotate into EE frame
        world_delta = [cube_pos[i] - ee_pos[i] for i in range(3)]
        delta_ee = world_to_ee_frame(world_delta, ee_quat)
        dist = math.sqrt(sum(d * d for d in delta_ee))

        # Determine gripper command based on phase
        gripper_cmd = 0.0  # open
        if phase in (PhaseId.CLOSE_GRIPPER, PhaseId.VERIFY_GRASP, PhaseId.LIFT):
            gripper_cmd = 1.0  # close

        # Scale delta to a reasonable step size
        max_step = 0.01
        if dist > 0:
            scale = min(max_step / dist, 1.0)
            step_delta = [d * scale for d in delta_ee]
        else:
            step_delta = [0.0, 0.0, 0.0]

        # Generate chunk of identical actions (teacher repeats same command)
        action = Action(
            dx=step_delta[0],
            dy=step_delta[1],
            dz=step_delta[2],
            droll=0.0,
            dpitch=0.0,
            dyaw=0.0,
            gripper_cmd=gripper_cmd,
        )
        actions = tuple(action for _ in range(chunk_horizon_steps))

        return ActionChunk(
            chunk_id=str(uuid.uuid4()),
            arm_id=arm_id,
            phase_id=phase,
            actions=actions,
            ts_ms=0,
        )

    return chunk_fn
