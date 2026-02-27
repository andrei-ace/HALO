"""ObserveFn/CaptureFn adapters: receives observations from Isaac Lab sim.

Plugs into TargetPerceptionService(arm_id, runtime, observe_fn=..., capture_fn=...).
"""

from __future__ import annotations

import math
from typing import Awaitable, Callable

import msgpack
import zmq

from halo.bridge.config import SimBridgeConfig
from halo.bridge.transforms import world_to_ee_frame
from halo.contracts.snapshots import TargetInfo

# Type alias matching TargetPerceptionService.ObserveFn(arm_id, target_handle)
ObserveFn = Callable[[str, str], Awaitable[TargetInfo | None]]


def make_sim_observe_fn(config: SimBridgeConfig | None = None) -> ObserveFn:
    """Create an observe_fn that receives observations from Isaac Lab via ZeroMQ SUB.

    Returns an async callable matching the ObserveFn(arm_id, target_handle)
    signature expected by TargetPerceptionService.
    """
    if config is None:
        config = SimBridgeConfig()

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(config.sim_obs_url)
    socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
    socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout_ms)

    async def observe_fn(arm_id: str, target_handle: str) -> TargetInfo | None:
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
        distance = math.sqrt(sum(d * d for d in delta_ee))

        return TargetInfo(
            handle=target_handle,
            hint_valid=True,
            confidence=0.95,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(delta_ee[0], delta_ee[1], delta_ee[2]),
            distance_m=distance,
        )

    return observe_fn
