"""Unit tests for LeaseManager."""

from __future__ import annotations

from halo.cognitive.lease import Lease, LeaseManager


def test_grant_increments_epoch():
    mgr = LeaseManager()
    assert mgr.current_epoch == 0
    lease = mgr.grant("local")
    assert lease.epoch == 1
    assert lease.holder == "local"
    assert mgr.current_epoch == 1


def test_grant_replaces_previous():
    mgr = LeaseManager()
    lease1 = mgr.grant("local")
    lease2 = mgr.grant("cloud")
    assert lease2.epoch == 2
    assert lease2.holder == "cloud"
    assert mgr.current_lease is lease2
    # Old epoch is no longer valid
    assert not mgr.is_valid(lease1.epoch)


def test_renew_success():
    mgr = LeaseManager()
    lease = mgr.grant("local")
    old_ts = lease.granted_at_ms
    assert mgr.renew(lease.epoch)
    assert mgr.current_lease.granted_at_ms >= old_ts


def test_renew_wrong_epoch():
    mgr = LeaseManager()
    mgr.grant("local")
    assert not mgr.renew(epoch=999)


def test_revoke():
    mgr = LeaseManager()
    lease = mgr.grant("local")
    mgr.revoke(lease.epoch)
    assert mgr.current_lease is None
    assert not mgr.is_valid(lease.epoch)


def test_revoke_wrong_epoch_noop():
    mgr = LeaseManager()
    lease = mgr.grant("local")
    mgr.revoke(epoch=999)
    # Lease should still be active
    assert mgr.current_lease is not None
    assert mgr.is_valid(lease.epoch)


def test_is_valid():
    mgr = LeaseManager()
    assert not mgr.is_valid(0)  # no lease yet
    lease = mgr.grant("local")
    assert mgr.is_valid(lease.epoch)
    assert not mgr.is_valid(lease.epoch + 1)


def test_lease_expired():
    lease = Lease(epoch=1, holder="local", granted_at_ms=0, token="tok", ttl_ms=1)
    # granted_at_ms=0 and ttl_ms=1 means it expired long ago
    assert lease.expired


def test_lease_not_expired():
    import time

    lease = Lease(
        epoch=1,
        holder="local",
        granted_at_ms=int(time.monotonic() * 1000),
        token="tok",
        ttl_ms=30_000,
    )
    assert not lease.expired


def test_grant_generates_token():
    mgr = LeaseManager()
    lease = mgr.grant("local")
    assert lease.token is not None
    assert len(lease.token) == 32  # hex UUID


def test_current_token():
    mgr = LeaseManager()
    assert mgr.current_token is None
    lease = mgr.grant("local")
    assert mgr.current_token == lease.token


def test_is_valid_token():
    mgr = LeaseManager()
    lease = mgr.grant("local")
    assert mgr.is_valid_token(lease.token)
    assert not mgr.is_valid_token("wrong-token")


def test_token_changes_on_grant():
    mgr = LeaseManager()
    l1 = mgr.grant("local")
    l2 = mgr.grant("cloud")
    assert l1.token != l2.token
    assert mgr.is_valid_token(l2.token)
    assert not mgr.is_valid_token(l1.token)
