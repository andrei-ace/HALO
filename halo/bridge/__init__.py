"""HALO-side bridge adapters for connecting to MuJoCo sim via ZeroMQ.

2-channel ZMQ architecture:
    TelemetryStream (SUB): frames + state from sim
    CommandRPC (REQ): step, reset, teacher_step, configure, set_hint
"""


class BridgeTransportError(Exception):
    """Raised when communication with the sim fails (timeout, ZMQ error).

    ControlService catches this to write STALE status so the rest of the
    system knows actuation is not happening.
    """


def make_teacher_step_fn(client):
    """Re-export from teacher_adapter for convenience.

    Usage::

        from halo.bridge import make_teacher_step_fn
        fn = make_teacher_step_fn(sim_client)
    """
    from halo.bridge.teacher_adapter import make_teacher_step_fn as _make

    return _make(client)
