"""LeaseManager — split-brain prevention for cognitive backend switching.

Only one backend (local or cloud) holds the active lease at a time.
Commands from a backend whose epoch doesn't match the current lease are
dropped by the Switchboard.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass
class Lease:
    """An active lease granting a backend the right to issue commands."""

    epoch: int
    holder: str  # "local" or "cloud"
    granted_at_ms: int
    token: str  # UUID, unique per grant
    ttl_ms: int = 30_000

    @property
    def expired(self) -> bool:
        return (int(time.monotonic() * 1000) - self.granted_at_ms) > self.ttl_ms


class LeaseManager:
    """Manages lease grant/renew/revoke lifecycle.

    Thread-safe for single-writer (Switchboard controls the lifecycle).
    """

    def __init__(self) -> None:
        self._epoch: int = 0
        self._lease: Lease | None = None

    @property
    def current_epoch(self) -> int:
        return self._epoch

    @property
    def current_lease(self) -> Lease | None:
        return self._lease

    @property
    def current_token(self) -> str | None:
        """Token of the active lease, or None."""
        return self._lease.token if self._lease else None

    def grant(self, holder: str, ttl_ms: int = 30_000) -> Lease:
        """Grant a new lease to *holder*, incrementing the epoch."""
        self._epoch += 1
        self._lease = Lease(
            epoch=self._epoch,
            holder=holder,
            granted_at_ms=int(time.monotonic() * 1000),
            token=uuid.uuid4().hex,
            ttl_ms=ttl_ms,
        )
        return self._lease

    def renew(self, epoch: int, ttl_ms: int | None = None) -> bool:
        """Renew the lease if *epoch* matches. Returns True on success."""
        if self._lease is None or self._lease.epoch != epoch:
            return False
        self._lease.granted_at_ms = int(time.monotonic() * 1000)
        if ttl_ms is not None:
            self._lease.ttl_ms = ttl_ms
        return True

    def revoke(self, epoch: int) -> None:
        """Revoke the lease if *epoch* matches. No-op if already revoked."""
        if self._lease is not None and self._lease.epoch == epoch:
            self._lease = None

    def is_valid(self, epoch: int) -> bool:
        """Check if a given epoch corresponds to the active, non-expired lease."""
        if self._lease is None:
            return False
        return self._lease.epoch == epoch and not self._lease.expired

    def is_valid_token(self, token: str) -> bool:
        """Check if token matches the active, non-expired lease."""
        if self._lease is None:
            return False
        return self._lease.token == token and not self._lease.expired
