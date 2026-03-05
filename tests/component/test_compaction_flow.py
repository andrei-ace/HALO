"""Tests for compaction data flow correctness.

Covers:
- MessageHistory.apply_compaction boundary correctness
- CompactionPlugin._find_compaction_boundary with various MessageHistory states
- Repeated compactions (summary re-compaction behavior)
- Cross-backend sync: Cloud compaction → Local MessageHistory
- SESSION_COMPACTED event published to EventBus
- RemoteCognitiveBackend compaction callback
- Full cycle: Cloud compaction → switch Local → switch Cloud → Cloud compaction again
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from halo.cognitive.compaction_plugin import CompactionPlugin
from halo.cognitive.compactor import CompactionResult, MessageHistory
from halo.cognitive.config import BackendType, CognitiveConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import Switchboard
from halo.contracts.events import EventType
from halo.runtime.event_bus import EventBus
from halo.services.target_perception_service.vlm_parser import VlmScene

# ---------------------------------------------------------------------------
# MessageHistory.apply_compaction
# ---------------------------------------------------------------------------


class TestApplyCompaction:
    def test_basic_compaction(self):
        """Compact first 2 messages, retain the rest."""
        mh = MessageHistory()
        id1 = mh.append("user", "msg1")
        mh.append("model", "reply1")
        mh.append("user", "msg2")
        mh.append("model", "reply2")

        result = mh.apply_compaction(id1, "summary")
        # Compacted the first record only (up to id1 inclusive)
        assert result.compacted_count == 1
        assert result.retained_count == 3  # model1, user2, model2

    def test_compact_two_records(self):
        """Compact first user+model pair."""
        mh = MessageHistory()
        mh.append("user", "msg1")
        id2 = mh.append("model", "reply1")
        mh.append("user", "msg2")
        mh.append("model", "reply2")

        result = mh.apply_compaction(id2, "summary")
        assert result.compacted_count == 2
        assert result.retained_count == 2  # user2, model2

        # History now: [summary, user2, model2]
        records = mh.get_all()
        assert len(records) == 3
        assert records[0].is_summary is True
        assert records[0].text == "summary"
        assert records[1].role == "user"
        assert records[1].text == "msg2"

    def test_compact_everything(self):
        """Compact all records — only summary remains."""
        mh = MessageHistory()
        mh.append("user", "msg1")
        id2 = mh.append("model", "reply1")

        result = mh.apply_compaction(id2, "summary")
        assert result.compacted_count == 2
        assert result.retained_count == 0

        records = mh.get_all()
        assert len(records) == 1
        assert records[0].is_summary is True

    def test_unknown_msg_id_raises(self):
        """apply_compaction raises ValueError for unknown msg_id."""
        mh = MessageHistory()
        mh.append("user", "msg1")

        with pytest.raises(ValueError, match="not found"):
            mh.apply_compaction("nonexistent", "summary")

    def test_repeated_compaction(self):
        """After compaction, new messages + second compaction works correctly."""
        mh = MessageHistory()
        mh.append("user", "msg1")
        id_m1 = mh.append("model", "reply1")

        # First compaction
        r1 = mh.apply_compaction(id_m1, "summary1")
        assert r1.compacted_count == 2
        assert r1.retained_count == 0

        # Add more messages
        mh.append("user", "msg2")
        mh.append("model", "reply2")
        mh.append("user", "msg3")
        id_m3 = mh.append("model", "reply3")

        # History: [summary1, user2, model2, user3, model3]
        assert mh.count() == 5

        # Second compaction up to model3 — compacts everything
        r2 = mh.apply_compaction(id_m3, "summary2")
        assert r2.compacted_count == 5
        assert r2.retained_count == 0

        records = mh.get_all()
        assert len(records) == 1
        assert records[0].text == "summary2"

    def test_partial_second_compaction(self):
        """Second compaction that only covers some of the new messages."""
        mh = MessageHistory()
        mh.append("user", "msg1")
        id_m1 = mh.append("model", "reply1")

        # First compaction
        mh.apply_compaction(id_m1, "summary1")

        # Add more messages
        mh.append("user", "msg2")
        id_m2 = mh.append("model", "reply2")
        mh.append("user", "msg3")
        mh.append("model", "reply3")

        # History: [summary1, user2, model2, user3, model3]
        # Compact up to model2 — keep user3, model3
        r = mh.apply_compaction(id_m2, "summary2")
        assert r.compacted_count == 3  # summary1 + user2 + model2
        assert r.retained_count == 2  # user3, model3

        records = mh.get_all()
        assert len(records) == 3  # summary2, user3, model3
        assert records[0].is_summary is True
        assert records[0].text == "summary2"
        assert records[1].text == "msg3"


# ---------------------------------------------------------------------------
# CompactionPlugin._find_compaction_boundary
# ---------------------------------------------------------------------------


class TestFindCompactionBoundary:
    def test_compact_one_invocation(self):
        """1 invocation compacted = first user+model pair."""
        mh = MessageHistory()
        mh.append("user", "u1")
        id_m1 = mh.append("model", "m1")
        mh.append("user", "u2")
        mh.append("model", "m2")

        boundary = CompactionPlugin._find_compaction_boundary(mh, 1)
        assert boundary == id_m1

    def test_compact_two_invocations(self):
        """2 invocations compacted = first two user+model pairs."""
        mh = MessageHistory()
        mh.append("user", "u1")
        mh.append("model", "m1")
        mh.append("user", "u2")
        id_m2 = mh.append("model", "m2")
        mh.append("user", "u3")
        mh.append("model", "m3")

        boundary = CompactionPlugin._find_compaction_boundary(mh, 2)
        assert boundary == id_m2

    def test_with_leading_summary(self):
        """After compaction, summary(model) + user+model pairs."""
        mh = MessageHistory()
        mh.append("user", "u1")
        id_m1 = mh.append("model", "m1")
        mh.apply_compaction(id_m1, "summary")

        # History: [summary(model)]
        # Add 3 more invocations
        mh.append("user", "u2")
        mh.append("model", "m2")
        mh.append("user", "u3")
        mh.append("model", "m3")
        mh.append("user", "u4")
        mh.append("model", "m4")

        # History: [summary, u2, m2, u3, m3, u4, m4]
        # Compact 1 invocation: should compact summary + u2 + m2
        boundary = CompactionPlugin._find_compaction_boundary(mh, 1)
        assert boundary == mh.get_all()[2].msg_id  # m2's msg_id

    def test_compact_zero_returns_none(self):
        """compacted_inv_count=0 returns None (nothing to compact)."""
        mh = MessageHistory()
        mh.append("user", "u1")
        mh.append("model", "m1")

        boundary = CompactionPlugin._find_compaction_boundary(mh, 0)
        assert boundary is None

    def test_empty_history_returns_none(self):
        mh = MessageHistory()
        boundary = CompactionPlugin._find_compaction_boundary(mh, 1)
        assert boundary is None

    def test_not_enough_invocations_returns_none(self):
        """Only 1 invocation but asked for 2."""
        mh = MessageHistory()
        mh.append("user", "u1")
        mh.append("model", "m1")

        boundary = CompactionPlugin._find_compaction_boundary(mh, 2)
        assert boundary is None

    def test_user_without_model_response(self):
        """Current invocation (user only, no model yet)."""
        mh = MessageHistory()
        mh.append("user", "u1")
        mh.append("model", "m1")
        mh.append("user", "u2")  # no model response yet

        # Compact 1 invocation: u1+m1
        boundary = CompactionPlugin._find_compaction_boundary(mh, 1)
        records = mh.get_all()
        assert boundary == records[1].msg_id  # m1

    def test_summary_recompaction_counts(self):
        """Verify counts when summary gets re-compacted with next invocation."""
        mh = MessageHistory()
        # Build: u1, m1
        mh.append("user", "u1")
        id_m1 = mh.append("model", "m1")
        # Compact
        r1 = mh.apply_compaction(id_m1, "sum1")
        assert r1.compacted_count == 2
        assert r1.retained_count == 0
        # History: [sum1(model)]

        # Add 5 invocations
        for i in range(2, 7):
            mh.append("user", f"u{i}")
            mh.append("model", f"m{i}")
        # History: [sum1, u2, m2, u3, m3, u4, m4, u5, m5, u6, m6] = 11 records

        # Compact 1 invocation (like interval=5, overlap=4 → compact 1)
        boundary = CompactionPlugin._find_compaction_boundary(mh, 1)
        # Should be m2 (index 2): compact [sum1, u2, m2]
        assert boundary == mh.get_all()[2].msg_id

        r2 = mh.apply_compaction(boundary, "sum2")
        assert r2.compacted_count == 3  # sum1 + u2 + m2
        assert r2.retained_count == 8  # u3..m6
        assert mh.get_all()[0].text == "sum2"

    def test_higher_compaction_ratio(self):
        """interval=5, overlap=1 → compact 4 invocations."""
        mh = MessageHistory()
        for i in range(1, 6):
            mh.append("user", f"u{i}")
            mh.append("model", f"m{i}")
        # 10 records, 5 invocations

        boundary = CompactionPlugin._find_compaction_boundary(mh, 4)
        # compact u1..m4 (8 records), retain u5, m5
        assert boundary == mh.get_all()[7].msg_id  # m4

        r = mh.apply_compaction(boundary, "summary")
        assert r.compacted_count == 8
        assert r.retained_count == 2


# ---------------------------------------------------------------------------
# Switchboard: SESSION_COMPACTED event + EventBus
# ---------------------------------------------------------------------------


def _mock_agent(backend="cloud"):
    mock = MagicMock()
    mock.decide = AsyncMock(return_value=[])
    mock.last_reasoning = ""
    mock.reset_loop_state = MagicMock()
    mock.reset_session = AsyncMock()
    mock.inject_handoff_context = AsyncMock()
    mock._pending_handoff = None
    mock.last_compaction = None
    mock.msg_history = MagicMock()
    mock.msg_history.clear = MagicMock()
    mock.msg_history.apply_compaction = MagicMock(side_effect=ValueError("not found"))
    return mock


def _make_switchboard(*, active=BackendType.CLOUD, bus=None):
    mock_cloud_agent = _mock_agent()
    mock_cloud_vlm = AsyncMock(return_value=VlmScene(scene="", detections=[]))
    mock_local_agent = _mock_agent(backend="local")

    with (
        patch("halo.cognitive.cloud_backend.PlannerAgent", return_value=mock_cloud_agent),
        patch("halo.services.target_perception_service.vlm_fn.make_vlm_fn", return_value=mock_cloud_vlm),
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_local_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.cloud_backend import CloudCognitiveBackend
        from halo.cognitive.local_backend import LocalCognitiveBackend

        cloud = CloudCognitiveBackend()
        local = LocalCognitiveBackend()
        ctx_store = ContextStore()
        lease_mgr = LeaseManager()
        cfg = CognitiveConfig(active=active, enable_failover=True)

        sb = Switchboard(
            config=cfg,
            local=local,
            cloud=cloud,
            lease_mgr=lease_mgr,
            context_store=ctx_store,
            bus=bus,
            arm_id="arm0",
        )

    return sb, ctx_store, cloud, local, mock_cloud_agent, mock_local_agent


@pytest.mark.asyncio
async def test_session_compacted_event_published():
    """SESSION_COMPACTED is published to EventBus on compaction."""
    bus = EventBus()
    bus._ensure_arm("arm0")
    queue = bus.subscribe("arm0", maxsize=0)

    sb, _, _, _, _, _ = _make_switchboard(bus=bus)

    result = CompactionResult(
        summary="Test summary",
        up_to_msg_id="msg-42",
        compacted_count=5,
        retained_count=3,
        ts_ms=1000,
    )
    await sb._sync_compaction_to_inactive(result)

    # Should have received SESSION_COMPACTED event
    assert not queue.empty()
    evt = queue.get_nowait()
    assert evt.type == EventType.SESSION_COMPACTED
    assert evt.arm_id == "arm0"
    assert evt.data["compacted_count"] == 5
    assert evt.data["retained_count"] == 3
    assert evt.data["summary"] == "Test summary"
    assert evt.data["up_to_msg_id"] == "msg-42"


@pytest.mark.asyncio
async def test_session_compacted_event_not_published_without_bus():
    """No crash when bus is None."""
    sb, _, _, _, _, _ = _make_switchboard(bus=None)

    result = CompactionResult(summary="Test", up_to_msg_id="x", compacted_count=1, retained_count=0, ts_ms=100)
    # Should not raise
    await sb._sync_compaction_to_inactive(result)


# ---------------------------------------------------------------------------
# RemoteCognitiveBackend: compaction callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_backend_compaction_callback():
    """RemoteCognitiveBackend calls _on_compaction when cloud returns compacted=true."""
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    backend = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    backend._last_reasoning = ""
    backend._on_compaction = AsyncMock()
    backend._run_logger = None

    # Mock the HTTP client response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "commands": [],
        "reasoning": "ok",
        "compacted": True,
        "compaction": {
            "summary": "Cloud summary",
            "up_to_msg_id": "abc123",
            "compacted_count": 10,
            "retained_count": 4,
        },
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    backend._client = mock_client
    backend._session_id = "test-session"

    from halo.contracts.enums import ActStatus, PerceptionFailureCode, SafetyState, SkillOutcomeState, TrackingStatus
    from halo.contracts.snapshots import (
        ActInfo,
        OutcomeInfo,
        PerceptionInfo,
        PlannerSnapshot,
        ProgressInfo,
        SafetyInfo,
        TargetInfo,
    )

    snap = PlannerSnapshot(
        snapshot_id="snap-001",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle=None,
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(0, 0, 0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.IDLE,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
        held_object_handle=None,
    )

    await backend.decide(snap)

    backend._on_compaction.assert_awaited_once()
    call_result = backend._on_compaction.call_args[0][0]
    assert isinstance(call_result, CompactionResult)
    assert call_result.summary == "Cloud summary"
    assert call_result.compacted_count == 10
    assert call_result.retained_count == 4


@pytest.mark.asyncio
async def test_remote_backend_no_callback_when_not_compacted():
    """RemoteCognitiveBackend does NOT call callback when compacted=false."""
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    backend = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    backend._last_reasoning = ""
    backend._on_compaction = AsyncMock()
    backend._run_logger = None

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "commands": [],
        "reasoning": "ok",
        "compacted": False,
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    backend._client = mock_client
    backend._session_id = "test-session"

    from halo.contracts.enums import ActStatus, PerceptionFailureCode, SafetyState, SkillOutcomeState, TrackingStatus
    from halo.contracts.snapshots import (
        ActInfo,
        OutcomeInfo,
        PerceptionInfo,
        PlannerSnapshot,
        ProgressInfo,
        SafetyInfo,
        TargetInfo,
    )

    snap = PlannerSnapshot(
        snapshot_id="snap-001",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle=None,
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(0, 0, 0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.IDLE,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
        held_object_handle=None,
    )

    await backend.decide(snap)
    backend._on_compaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_backend_no_callback_when_none():
    """RemoteCognitiveBackend handles _on_compaction=None gracefully."""
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    backend = RemoteCognitiveBackend.__new__(RemoteCognitiveBackend)
    backend._last_reasoning = ""
    backend._on_compaction = None
    backend._run_logger = None

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "commands": [],
        "reasoning": "ok",
        "compacted": True,
        "compaction": {"summary": "s", "up_to_msg_id": "x", "compacted_count": 1, "retained_count": 0},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    backend._client = mock_client
    backend._session_id = "test-session"

    from halo.contracts.enums import ActStatus, PerceptionFailureCode, SafetyState, SkillOutcomeState, TrackingStatus
    from halo.contracts.snapshots import (
        ActInfo,
        OutcomeInfo,
        PerceptionInfo,
        PlannerSnapshot,
        ProgressInfo,
        SafetyInfo,
        TargetInfo,
    )

    snap = PlannerSnapshot(
        snapshot_id="snap-001",
        ts_ms=1000,
        arm_id="arm0",
        skill=None,
        target=TargetInfo(
            handle=None,
            hint_valid=False,
            confidence=0.0,
            obs_age_ms=0,
            time_skew_ms=0,
            delta_xyz_ee=(0, 0, 0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.IDLE,
            failure_code=PerceptionFailureCode.OK,
            reacquire_fail_count=0,
            vlm_job_pending=False,
        ),
        act=ActInfo(status=ActStatus.IDLE, buffer_fill_ms=0, buffer_low=False),
        progress=ProgressInfo(elapsed_ms=0, no_progress_ms=0, delta_distance=0.0),
        outcome=OutcomeInfo(state=SkillOutcomeState.IN_PROGRESS, reason_code=None, needs_verify=False),
        safety=SafetyInfo(state=SafetyState.OK, reflex_active=False, reason_codes=()),
        command_acks=(),
        recent_events=(),
        held_object_handle=None,
    )

    # Should not raise even with _on_compaction=None
    await backend.decide(snap)


# ---------------------------------------------------------------------------
# Switchboard wires callback for RemoteCognitiveBackend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switchboard_wires_remote_backend():
    """Switchboard wires set_on_compaction on RemoteCognitiveBackend."""
    from halo.cognitive.config import RemoteCloudConfig
    from halo.cognitive.remote_backend import RemoteCognitiveBackend

    remote = RemoteCognitiveBackend(config=RemoteCloudConfig(service_url="http://fake:8080"))

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=_mock_agent()),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.local_backend import LocalCognitiveBackend

        local = LocalCognitiveBackend()

    cfg = CognitiveConfig(active=BackendType.CLOUD)
    Switchboard(config=cfg, local=local, cloud=remote)

    assert remote._on_compaction is not None


# ---------------------------------------------------------------------------
# Cross-backend sync: compaction UUID mismatch → fallback to clear()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_backend_sync_uuid_mismatch_clears():
    """When cloud up_to_msg_id not found in local MessageHistory, fallback to clear()."""
    sb, _, _, _, _, mock_local_agent = _make_switchboard()

    # local agent's msg_history.apply_compaction raises ValueError (UUID mismatch)
    mock_local_agent.msg_history.apply_compaction = MagicMock(side_effect=ValueError("not found"))

    result = CompactionResult(
        summary="Cloud summary",
        up_to_msg_id="cloud-uuid-xyz",
        compacted_count=10,
        retained_count=3,
        ts_ms=1000,
    )
    await sb._sync_compaction_to_inactive(result)

    # apply_compaction was attempted
    mock_local_agent.msg_history.apply_compaction.assert_called_once_with("cloud-uuid-xyz", "Cloud summary")
    # Fell back to clear()
    mock_local_agent.msg_history.clear.assert_called_once()
    # Session was still reset + handoff injected
    mock_local_agent.reset_session.assert_awaited_once()
    mock_local_agent.inject_handoff_context.assert_awaited_once()


# ---------------------------------------------------------------------------
# Full cycle: Cloud → compaction → Local → Cloud (MessageHistory state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_cycle_cloud_compact_switch_local_switch_cloud():
    """Full cycle with real MessageHistory to verify state after round-trip."""
    # Use real MessageHistory instances to track actual state
    cloud_mh = MessageHistory()
    local_mh = MessageHistory()

    mock_cloud_agent = _mock_agent()
    mock_cloud_agent.msg_history = cloud_mh
    mock_cloud_vlm = AsyncMock(return_value=VlmScene(scene="", detections=[]))

    mock_local_agent = _mock_agent()
    mock_local_agent.msg_history = local_mh

    with (
        patch("halo.cognitive.cloud_backend.PlannerAgent", return_value=mock_cloud_agent),
        patch("halo.services.target_perception_service.vlm_fn.make_vlm_fn", return_value=mock_cloud_vlm),
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_local_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.cloud_backend import CloudCognitiveBackend
        from halo.cognitive.local_backend import LocalCognitiveBackend

        cloud = CloudCognitiveBackend()
        local = LocalCognitiveBackend()

    ctx_store = ContextStore()
    lease_mgr = LeaseManager()
    cfg = CognitiveConfig(active=BackendType.CLOUD, enable_failover=True)

    sb = Switchboard(
        config=cfg,
        local=local,
        cloud=cloud,
        lease_mgr=lease_mgr,
        context_store=ctx_store,
        arm_id="arm0",
    )

    # -- Phase 1: Cloud runs, builds up MessageHistory --
    for i in range(5):
        cloud_mh.append("user", f"snap_{i}")
        cloud_mh.append("model", f"reply_{i}")
    assert cloud_mh.count() == 10

    # Simulate compaction on cloud: compact first 3 invocations
    records = cloud_mh.get_all()
    boundary_id = records[5].msg_id  # model of 3rd invocation (index 5)
    cloud_result = cloud_mh.apply_compaction(boundary_id, "Cloud compaction summary")
    assert cloud_result.compacted_count == 6
    assert cloud_result.retained_count == 4  # 2 invocations retained
    assert cloud_mh.count() == 5  # summary + 4 retained

    # Sync compaction to inactive (local)
    await sb._sync_compaction_to_inactive(cloud_result)

    # Local MessageHistory should be cleared (UUID mismatch)
    assert local_mh.count() == 0

    # -- Phase 2: Switch to Local --
    await sb.switch_to(BackendType.LOCAL, reason="test")
    assert sb.active_type == BackendType.LOCAL

    # Local agent runs, builds history
    for i in range(3):
        local_mh.append("user", f"local_snap_{i}")
        local_mh.append("model", f"local_reply_{i}")
    assert local_mh.count() == 6

    # -- Phase 3: Switch back to Cloud --
    await sb.switch_to(BackendType.CLOUD, reason="test failback")
    assert sb.active_type == BackendType.CLOUD

    # Cloud MessageHistory still has its post-compaction state
    # (switch_to doesn't modify the active backend's msg_history)
    assert cloud_mh.count() == 5  # summary + 4 retained records

    # Cloud can still run more invocations and compact again
    for i in range(5):
        cloud_mh.append("user", f"snap2_{i}")
        cloud_mh.append("model", f"reply2_{i}")
    assert cloud_mh.count() == 15

    # Second compaction on cloud
    records2 = cloud_mh.get_all()
    # Compact first 3 new invocations (indices 0-6: summary + 2 retained + 1 new pair)
    boundary_id2 = records2[6].msg_id
    cloud_result2 = cloud_mh.apply_compaction(boundary_id2, "Cloud compaction summary 2")
    assert cloud_result2.compacted_count == 7
    assert cloud_result2.retained_count == 8
    assert cloud_mh.get_all()[0].is_summary is True
    assert cloud_mh.get_all()[0].text == "Cloud compaction summary 2"


# ---------------------------------------------------------------------------
# SESSION_COMPACTED filtered from recent_events in runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_compacted_filtered_from_recent_events():
    """SESSION_COMPACTED should be filtered from recent_events in snapshots."""
    from halo.runtime.runtime import HALORuntime

    runtime = HALORuntime()
    runtime.register_arm("arm0")

    from halo.contracts.events import EventEnvelope

    # Publish a regular event
    await runtime.bus.publish(
        EventEnvelope(
            event_id="evt-1",
            type=EventType.SKILL_SUCCEEDED,
            ts_ms=1000,
            arm_id="arm0",
            data={},
        )
    )
    # Publish SESSION_COMPACTED
    await runtime.bus.publish(
        EventEnvelope(
            event_id="evt-2",
            type=EventType.SESSION_COMPACTED,
            ts_ms=2000,
            arm_id="arm0",
            data={"summary": "test"},
        )
    )
    # Publish another regular event
    await runtime.bus.publish(
        EventEnvelope(
            event_id="evt-3",
            type=EventType.TARGET_ACQUIRED,
            ts_ms=3000,
            arm_id="arm0",
            data={},
        )
    )

    snap = await runtime.get_latest_runtime_snapshot("arm0")
    event_types = [e.type for e in snap.recent_events]

    # SESSION_COMPACTED should be filtered out
    assert EventType.SESSION_COMPACTED not in event_types
    assert EventType.SKILL_SUCCEEDED in event_types
    assert EventType.TARGET_ACQUIRED in event_types
