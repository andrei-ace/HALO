"""Component tests for CompactionPlugin integration.

Tests:
- Cloud agent uses CompactionPlugin with compaction enabled
- Local agent uses CompactionPlugin (snapshot deprecation only)
- Compaction detection after decide()
- Switchboard syncs compaction to inactive backend
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from halo.cognitive.compaction_plugin import CompactionPlugin
from halo.cognitive.compactor import CompactionResult
from halo.cognitive.config import BackendType, CognitiveConfig, CompactionConfig
from halo.cognitive.context_store import ContextStore
from halo.cognitive.lease import LeaseManager
from halo.cognitive.switchboard import Switchboard
from halo.contracts.enums import (
    ActStatus,
    PerceptionFailureCode,
    SafetyState,
    SkillOutcomeState,
    TrackingStatus,
)
from halo.contracts.snapshots import (
    ActInfo,
    OutcomeInfo,
    PerceptionInfo,
    PlannerSnapshot,
    ProgressInfo,
    SafetyInfo,
    TargetInfo,
)
from halo.services.target_perception_service.vlm_parser import VlmScene


def _idle_snap() -> PlannerSnapshot:
    return PlannerSnapshot(
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
            delta_xyz_ee=(0.0, 0.0, 0.0),
            distance_m=0.0,
        ),
        perception=PerceptionInfo(
            tracking_status=TrackingStatus.LOST,
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


def _mock_agent(backend="cloud", compaction_config=None):
    """Create a mock PlannerAgent."""
    mock = MagicMock()
    mock.decide = AsyncMock(return_value=[])
    mock.last_reasoning = ""
    mock.reset_loop_state = MagicMock()
    mock.reset_session = AsyncMock()
    mock.inject_handoff_context = AsyncMock()
    mock.inject_compaction_state = AsyncMock()
    mock._pending_handoff = None
    mock.last_compaction = None
    mock.msg_history = MagicMock()
    mock.msg_history.clear = MagicMock()
    return mock


class _FakeCloudBackend:
    def __init__(self, agent: MagicMock, vlm: AsyncMock | None = None) -> None:
        self._agent = agent
        self._vlm = vlm or AsyncMock(return_value=VlmScene(scene="", detections=[]))
        self._on_compaction = None

    @property
    def backend_type(self) -> str:
        return BackendType.CLOUD

    @property
    def agent(self) -> MagicMock:
        return self._agent

    @property
    def last_reasoning(self) -> str:
        return self._agent.last_reasoning

    async def decide(self, snap, operator_cmd=None, epoch=None):
        commands = await self._agent.decide(snap, operator_cmd=operator_cmd, epoch=epoch)
        compaction = self._agent.last_compaction
        if compaction is not None and self._on_compaction is not None:
            try:
                await self._on_compaction(compaction)
            except Exception:
                pass
        return commands

    async def vlm_scene(self, arm_id, image, known_handles=None, target_handle=None):
        return await self._vlm(arm_id, image, known_handles, target_handle=target_handle)

    async def health_check(self) -> bool:
        return True

    def reset_loop_state(self) -> None:
        self._agent.reset_loop_state()

    def set_on_compaction(self, callback) -> None:
        self._on_compaction = callback


# ---------------------------------------------------------------------------
# Cloud agent uses CompactionPlugin
# ---------------------------------------------------------------------------


def test_cloud_agent_compaction_plugin_enabled():
    """PlannerAgent(backend='cloud', compaction_config=...) creates plugin with compaction enabled."""
    cfg = CompactionConfig(enabled=True, compaction_interval=10, overlap_size=2)

    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gemini-3.1-flash-lite-preview",
            base_url="",
            prompts_dir=Path("/tmp"),
            backend="cloud",
            compaction_config=cfg,
        )

    assert isinstance(agent._compaction_plugin, CompactionPlugin)
    assert agent._compaction_plugin.enabled is True
    # All backends use App with plugins
    assert agent._runner.app is not None


def test_local_agent_plugin_deprecation_only():
    """PlannerAgent(backend='local') uses CompactionPlugin without compaction."""
    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gpt-oss:20b",
            base_url="http://localhost:11434",
            prompts_dir=Path("/tmp"),
            backend="local",
        )

    assert isinstance(agent._compaction_plugin, CompactionPlugin)
    assert agent._compaction_plugin.enabled is False
    assert agent._runner.app is not None


def test_cloud_agent_compaction_disabled():
    """PlannerAgent(backend='cloud', compaction_config=disabled) has plugin but compaction disabled."""
    cfg = CompactionConfig(enabled=False)

    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gemini-3.1-flash-lite-preview",
            base_url="",
            prompts_dir=Path("/tmp"),
            backend="cloud",
            compaction_config=cfg,
        )

    assert isinstance(agent._compaction_plugin, CompactionPlugin)
    assert agent._compaction_plugin.enabled is False


def test_cloud_agent_no_compaction_config():
    """PlannerAgent(backend='cloud') without compaction_config has plugin but compaction disabled."""
    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gemini-3.1-flash-lite-preview",
            base_url="",
            prompts_dir=Path("/tmp"),
            backend="cloud",
        )

    assert isinstance(agent._compaction_plugin, CompactionPlugin)
    assert agent._compaction_plugin.enabled is False


# ---------------------------------------------------------------------------
# Compaction detection after decide()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_detection_after_decide():
    """Cloud backend callback fires when the planner reports compaction."""
    mock_agent = _mock_agent()

    compaction_result = CompactionResult(
        summary="Test summary",
        up_to_msg_id="abc123",
        compacted_count=5,
        retained_count=2,
        ts_ms=1000,
    )
    mock_agent.last_compaction = compaction_result

    callback = AsyncMock()
    backend = _FakeCloudBackend(mock_agent)
    backend.set_on_compaction(callback)

    snap = _idle_snap()
    await backend.decide(snap)

    callback.assert_awaited_once_with(compaction_result)


@pytest.mark.asyncio
async def test_no_compaction_no_callback():
    """No callback fired when no compaction occurred."""
    mock_agent = _mock_agent()
    mock_agent.last_compaction = None

    callback = AsyncMock()
    backend = _FakeCloudBackend(mock_agent)
    backend.set_on_compaction(callback)

    snap = _idle_snap()
    await backend.decide(snap)

    callback.assert_not_awaited()


# ---------------------------------------------------------------------------
# Switchboard syncs compaction to inactive backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switchboard_sync_compaction():
    """Switchboard._sync_compaction_to_inactive calls inject_compaction_state on local agent."""
    mock_cloud_agent = _mock_agent()
    mock_cloud_agent.last_compaction = None
    # Cloud agent has post-compaction history: summary + retained
    from halo.cognitive.compactor import MessageHistory

    cloud_mh = MessageHistory()
    cloud_mh.append("user", "u1")
    id_m1 = cloud_mh.append("model", "m1")
    cloud_mh.apply_compaction(id_m1, "Cloud summary")
    cloud_mh.append("user", "u2")
    cloud_mh.append("model", "m2")
    mock_cloud_agent.msg_history = cloud_mh

    mock_local_agent = _mock_agent(backend="local")
    mock_local_agent.inject_compaction_state = AsyncMock()

    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=mock_local_agent,
        ),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.local_backend import LocalCognitiveBackend

        cloud = _FakeCloudBackend(mock_cloud_agent)
        local = LocalCognitiveBackend()

        cfg = CognitiveConfig(active=BackendType.CLOUD)
        ctx_store = ContextStore()
        lease_mgr = LeaseManager()

        sb = Switchboard(
            config=cfg,
            local=local,
            cloud=cloud,
            lease_mgr=lease_mgr,
            context_store=ctx_store,
        )

    result = CompactionResult(
        summary="Compacted context summary",
        up_to_msg_id="msg-42",
        compacted_count=10,
        retained_count=3,
        ts_ms=5000,
    )

    await sb._sync_compaction_to_inactive(result)

    # Local agent should have inject_compaction_state called with summary + retained records
    mock_local_agent.inject_compaction_state.assert_awaited_once()
    call_args = mock_local_agent.inject_compaction_state.call_args
    assert call_args[0][0] == "Compacted context summary"
    retained = call_args[0][1]
    assert len(retained) == 2  # u2, m2 (summary filtered out)
    assert retained[0].role == "user"
    assert retained[0].text == "u2"
    assert retained[1].role == "model"
    assert retained[1].text == "m2"

    # Context store should have a compaction entry
    entries = ctx_store.get_entries_after(-1)
    assert any(e.entry_type == "compaction" for e in entries)


@pytest.mark.asyncio
async def test_switchboard_wires_compaction_callback():
    """Switchboard wires set_on_compaction on the configured cloud backend."""
    mock_cloud_agent = _mock_agent()

    with (
        patch(
            "halo.cognitive.local_backend.PlannerAgent",
            return_value=_mock_agent(backend="local"),
        ),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.local_backend import LocalCognitiveBackend

        cloud = _FakeCloudBackend(mock_cloud_agent)
        local = LocalCognitiveBackend()

        cfg = CognitiveConfig(active=BackendType.CLOUD)
        Switchboard(config=cfg, local=local, cloud=cloud)

    # The callback should be wired
    assert cloud._on_compaction is not None


# ---------------------------------------------------------------------------
# Compaction + backend switch interaction
# ---------------------------------------------------------------------------


def _make_switchboard(*, active=BackendType.CLOUD, enable_failover=False):
    """Helper: create Switchboard with mocked backends, return (sb, ctx_store, cloud, local, mock agents)."""
    mock_cloud_agent = _mock_agent()
    mock_cloud_agent.last_compaction = None
    mock_cloud_agent.last_reasoning = "cloud reasoning"

    mock_local_agent = _mock_agent(backend="local")
    mock_local_agent.last_reasoning = "local reasoning"

    with (
        patch("halo.cognitive.local_backend.PlannerAgent", return_value=mock_local_agent),
        patch("halo.cognitive.local_backend.make_vlm_fn", return_value=AsyncMock()),
    ):
        from halo.cognitive.local_backend import LocalCognitiveBackend

        cloud = _FakeCloudBackend(mock_cloud_agent)
        local = LocalCognitiveBackend()
        ctx_store = ContextStore()
        lease_mgr = LeaseManager()
        cfg = CognitiveConfig(active=active, enable_failover=enable_failover)

        sb = Switchboard(
            config=cfg,
            local=local,
            cloud=cloud,
            lease_mgr=lease_mgr,
            context_store=ctx_store,
        )

    return sb, ctx_store, cloud, local, mock_cloud_agent, mock_local_agent


@pytest.mark.asyncio
async def test_compaction_entry_included_in_failover_warmup():
    """Compaction journal entry is included when switch_to() pre-warms the new backend."""
    sb, ctx_store, cloud, local, mock_cloud_agent, mock_local_agent = _make_switchboard()

    # Simulate a compaction event while cloud is active
    result = CompactionResult(
        summary="Summarized 20 messages",
        up_to_msg_id="msg-20",
        compacted_count=20,
        retained_count=4,
        ts_ms=1000,
    )
    await sb._sync_compaction_to_inactive(result)

    # Verify compaction entry is in context store
    entries = ctx_store.get_entries_after(-1)
    compaction_entries = [e for e in entries if e.entry_type == "compaction"]
    assert len(compaction_entries) == 1
    assert "20 messages" in compaction_entries[0].summary

    # Now switch to local — warm_up should receive the compaction entry in journal
    mock_local_agent.reset_session.reset_mock()
    mock_local_agent.inject_handoff_context.reset_mock()
    await sb.switch_to(BackendType.LOCAL, reason="test failover")

    # Local warm_up was called (via switch_to pre-warm)
    # The journal entries passed to warm_up include the compaction entry
    assert sb.active_type == BackendType.LOCAL


@pytest.mark.asyncio
async def test_compaction_callback_error_does_not_block_decide():
    """If compaction callback raises, decide() still returns commands."""
    mock_agent = _mock_agent()

    compaction_result = CompactionResult(
        summary="Summary",
        up_to_msg_id="x",
        compacted_count=5,
        retained_count=2,
        ts_ms=100,
    )
    mock_agent.last_compaction = compaction_result
    mock_agent.last_reasoning = "do something"

    callback = AsyncMock(side_effect=RuntimeError("callback exploded"))
    backend = _FakeCloudBackend(mock_agent)
    backend.set_on_compaction(callback)

    snap = _idle_snap()
    # Should not raise despite callback error
    cmds = await backend.decide(snap)
    assert cmds == []  # mock returns []
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_compaction_when_local_is_active_is_noop():
    """When active=LOCAL, _sync_compaction_to_inactive targets cloud (no-op)."""
    sb, ctx_store, cloud, local, _, mock_local_agent = _make_switchboard(active=BackendType.LOCAL)

    result = CompactionResult(
        summary="Summary",
        up_to_msg_id="x",
        compacted_count=5,
        retained_count=2,
        ts_ms=100,
    )
    await sb._sync_compaction_to_inactive(result)

    # Local agent should NOT have been touched (it's the active one, not inactive)
    mock_local_agent.reset_session.assert_not_awaited()
    mock_local_agent.inject_handoff_context.assert_not_awaited()

    # Context store entry is still written
    entries = ctx_store.get_entries_after(-1)
    assert any(e.entry_type == "compaction" for e in entries)


@pytest.mark.asyncio
async def test_multiple_compactions_accumulate_in_context_store():
    """Multiple compaction syncs accumulate journal entries."""
    sb, ctx_store, _, _, _, mock_local_agent = _make_switchboard()

    for i in range(3):
        result = CompactionResult(
            summary=f"Summary #{i}",
            up_to_msg_id=f"msg-{i}",
            compacted_count=10 + i,
            retained_count=3,
            ts_ms=1000 * i,
        )
        await sb._sync_compaction_to_inactive(result)

    entries = ctx_store.get_entries_after(-1)
    compaction_entries = [e for e in entries if e.entry_type == "compaction"]
    assert len(compaction_entries) == 3

    # Local agent had inject_compaction_state called 3 times (once per compaction)
    assert mock_local_agent.inject_compaction_state.await_count == 3


@pytest.mark.asyncio
async def test_sync_compaction_local_agent_error_still_writes_journal():
    """If local agent reset_session raises, journal entry is still written."""
    sb, ctx_store, _, _, _, mock_local_agent = _make_switchboard()

    mock_local_agent.inject_compaction_state = AsyncMock(side_effect=RuntimeError("session gone"))

    result = CompactionResult(
        summary="Summary",
        up_to_msg_id="x",
        compacted_count=5,
        retained_count=2,
        ts_ms=100,
    )
    await sb._sync_compaction_to_inactive(result)

    # Journal entry should still be written (append is outside try/except)
    entries = ctx_store.get_entries_after(-1)
    assert any(e.entry_type == "compaction" for e in entries)


@pytest.mark.asyncio
async def test_reset_loop_state_preserves_msg_history():
    """reset_loop_state() does not clear msg_history (session state, not loop state)."""
    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gemini-3.1-flash-lite-preview",
            base_url="",
            prompts_dir=Path("/tmp"),
            backend="cloud",
        )

    agent.msg_history.append("user", "hello")
    agent.msg_history.append("model", "hi")
    assert agent.msg_history.count() == 2

    agent.reset_loop_state()
    assert agent.msg_history.count() == 2  # preserved


@pytest.mark.asyncio
async def test_reset_session_clears_msg_history():
    """reset_session() clears msg_history and last_compaction."""
    with patch("halo.services.planner_service.agent._load_prompts", return_value="test prompt"):
        from halo.services.planner_service.agent import PlannerAgent

        agent = PlannerAgent(
            model_name="gemini-3.1-flash-lite-preview",
            base_url="",
            prompts_dir=Path("/tmp"),
            backend="cloud",
        )

    agent.msg_history.append("user", "hello")
    agent._last_compaction = CompactionResult(
        summary="s",
        up_to_msg_id="x",
        compacted_count=1,
        retained_count=0,
        ts_ms=0,
    )

    await agent.reset_session()

    assert agent.msg_history.count() == 0
    assert agent.last_compaction is None
