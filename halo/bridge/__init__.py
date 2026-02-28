"""HALO-side bridge adapters for connecting to MuJoCo sim via ZeroMQ.

4-channel ZMQ architecture:
    Ch1 (SUB): telemetry — frames + state from sim
    Ch2 (PUB): tracking hints to sim
    Ch3 (REQ): commands — step, reset, teacher_step
    Ch4 (REP): queries — VLM/tracker from sim
"""


class BridgeTransportError(Exception):
    """Raised when communication with the sim fails (timeout, ZMQ error).

    ControlService catches this to write STALE status so the rest of the
    system knows actuation is not happening.
    """
