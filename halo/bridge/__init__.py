"""HALO-side bridge adapters for connecting to MuJoCo sim via ZeroMQ.

2-channel ZMQ architecture:
    TelemetryStream (SUB): frames + state from sim
    CommandRPC (REQ): step, reset, start_pick, configure, set_hint
"""


class BridgeTransportError(Exception):
    """Raised when communication with the sim fails (timeout, ZMQ error).

    ControlService catches this to write STALE status so the rest of the
    system knows actuation is not happening.
    """
