"""ApplyFn adapter: sends actions to Isaac Lab sim via ZeroMQ.

Plugs into ControlService(arm_id, runtime, apply_fn=make_sim_apply_fn()).

Uses ``zmq.asyncio`` so the socket lives exclusively on the event loop
thread — no worker threads, no thread-affinity concerns.  On recv timeout
the socket is recreated to reset the REQ state machine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import msgpack
import zmq
import zmq.asyncio

from halo.bridge import BridgeTransportError
from halo.bridge.config import SimBridgeConfig
from halo.contracts.actions import Action

logger = logging.getLogger(__name__)

# Type alias matching ControlService.apply_fn signature
ApplyFn = Callable[[str, Action], Awaitable[None]]

PhaseGetter = Callable[[], int]


def _phase_idle() -> int:
    return 0


def make_sim_apply_fn(
    config: SimBridgeConfig | None = None,
    phase_getter: PhaseGetter | None = None,
) -> ApplyFn:
    """Create an apply_fn that sends actions to Isaac Lab via ZeroMQ.

    Returns an async callable matching ControlService's apply_fn signature:
        async def apply_fn(arm_id: str, action: Action) -> None

    *phase_getter* is called on every tick to include the current HALO
    PhaseId in the action message so the sim server can gate the wrist
    camera accordingly.  Defaults to IDLE (0) if not provided.

    The socket uses ``zmq.asyncio`` and lives entirely on the event loop
    thread — no cross-thread access.  ``asyncio.wait_for`` enforces the
    recv timeout without blocking the loop.  On timeout or ZMQ error the
    socket is automatically recreated so the REQ state machine cannot
    wedge.
    """
    if config is None:
        config = SimBridgeConfig()
    get_phase = phase_getter or _phase_idle
    timeout_s = config.recv_timeout_ms / 1000.0

    ctx = zmq.asyncio.Context.instance()
    state: dict[str, zmq.asyncio.Socket | None] = {"socket": None}

    def _new_socket() -> zmq.asyncio.Socket:
        if state["socket"] is not None:
            state["socket"].close(linger=0)
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(config.sim_action_url)
        state["socket"] = sock
        return sock

    _new_socket()

    async def apply_fn(arm_id: str, action: Action) -> None:
        msg = {
            "type": "action",
            "arm_id": arm_id,
            "action": [action.dx, action.dy, action.dz, action.droll, action.dpitch, action.dyaw, action.gripper_cmd],
            "phase_id": get_phase(),
            "ts_ms": int(time.time() * 1000),
        }
        packed = msgpack.packb(msg, use_bin_type=True)
        sock = state["socket"]
        try:
            await sock.send(packed)
            reply_data = await asyncio.wait_for(sock.recv(), timeout=timeout_s)
            msgpack.unpackb(reply_data, raw=False)
        except asyncio.TimeoutError:
            logger.warning("sim_apply_fn: recv timed out (%.1f s), recreating socket", timeout_s)
            _new_socket()
            raise BridgeTransportError(f"recv timed out ({timeout_s:.1f} s)")
        except zmq.ZMQError as exc:
            logger.warning("sim_apply_fn: ZMQ error: %s, recreating socket", exc)
            _new_socket()
            raise BridgeTransportError(f"ZMQ error: {exc}") from exc

    return apply_fn
