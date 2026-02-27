"""HALO-side bridge adapters for connecting to Isaac Lab sim via ZeroMQ."""


class BridgeTransportError(Exception):
    """Raised when action delivery to the sim fails (timeout, ZMQ error).

    ControlService catches this to write STALE status so the rest of the
    system knows actuation is not happening.
    """
