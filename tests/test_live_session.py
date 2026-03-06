"""Unit tests for LivePlannerSession — mock Runner/LiveRequestQueue, verify decide(), audio, loop detection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from halo.cognitive.config import CloudConfig
from halo.cognitive.live_session import LivePlannerSession, LiveSessionState, _command_key
from halo.contracts.commands import CommandEnvelope, DescribeScenePayload
from halo.contracts.enums import (
    ActStatus,
    CommandType,
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


def _make_cmd(cmd_type: CommandType = CommandType.DESCRIBE_SCENE) -> CommandEnvelope:
    return CommandEnvelope(
        command_id="cmd-1",
        arm_id="arm0",
        issued_at_ms=1000,
        type=cmd_type,
        payload=DescribeScenePayload(reason="test"),
    )


# ---------------------------------------------------------------------------
# LiveSessionState
# ---------------------------------------------------------------------------


def test_live_session_state_defaults():
    state = LiveSessionState()
    assert state.connected is False
    assert state.last_transcription_in == ""
    assert state.last_transcription_out == ""
    assert state.turn_active is False
    assert state.reconnect_count == 0


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def test_session_construction():
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))
    assert session.state.connected is False
    assert session.last_reasoning == ""


# ---------------------------------------------------------------------------
# decide() — mock the run_live loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_sends_content_and_waits_for_turn_complete():
    """decide() should send content to queue and wait for turn_complete."""
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))

    # Mock ADK internals
    mock_session_service = MagicMock()
    mock_session_service.create_session = AsyncMock()

    # Simulate: start() creates everything, then decide() pushes content.
    # We'll manually control the turn_complete event.
    with (
        patch("halo.cognitive.live_session.InMemorySessionService", return_value=mock_session_service),
        patch("halo.cognitive.live_session.Agent"),
        patch("halo.cognitive.live_session.Runner") as mock_runner_cls,
    ):
        # Make run_live return an empty async iterator
        mock_runner = mock_runner_cls.return_value

        async def fake_run_live(**kwargs):
            # Just keep alive until cancelled
            try:
                while True:
                    await asyncio.sleep(100)
            except asyncio.CancelledError:
                return
            yield  # make it an async generator  # noqa: RET503

        mock_runner.run_live = fake_run_live

        await session.start()

    # Simulate turn_complete firing shortly after decide() is called
    snap = _idle_snap()

    async def set_turn_complete():
        await asyncio.sleep(0.05)
        session._turn_complete_event.set()

    asyncio.create_task(set_turn_complete())

    commands = await session.decide(snap, operator_cmd="pick cube")
    assert isinstance(commands, list)

    await session.stop()


# ---------------------------------------------------------------------------
# Command accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pending_commands():
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))
    cmd = _make_cmd()

    # Enqueue a command
    session._pending_commands.put_nowait(cmd)
    drained = session.drain_pending_commands()
    assert len(drained) == 1
    assert drained[0] is cmd

    # Queue should be empty now
    assert session.drain_pending_commands() == []


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def test_command_key_stability():
    cmd = _make_cmd()
    k1 = _command_key(cmd)
    k2 = _command_key(cmd)
    assert k1 == k2
    assert cmd.type.value in k1


@pytest.mark.asyncio
async def test_loop_detection_clears_after_threshold():
    """When the same command repeats MAX_LOOP_RETRIES times, decide() returns []."""
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))

    # Don't actually start the live session — just test loop detection directly
    session._started = True

    cmd = _make_cmd()

    for i in range(LivePlannerSession.MAX_LOOP_RETRIES):
        # Pre-enqueue the command in the pending queue (simulates voice-triggered)
        session._pending_commands.put_nowait(cmd)

        # Immediately signal turn_complete so decide() doesn't block
        async def set_tc():
            await asyncio.sleep(0.01)
            session._turn_complete_event.set()

        asyncio.create_task(set_tc())

        snap = _idle_snap()
        result = await session.decide(snap)

        if i < LivePlannerSession.MAX_LOOP_RETRIES - 1:
            assert len(result) >= 1
        else:
            # On the 4th identical command, loop detection should kick in
            assert result == []
            assert "Loop detected" in session.last_reasoning


# ---------------------------------------------------------------------------
# Audio event routing
# ---------------------------------------------------------------------------


def test_handle_event_audio_to_playback():
    """Audio inline_data in events should be enqueued to playback."""
    mock_playback = MagicMock()
    session = LivePlannerSession(config=CloudConfig(), audio_playback=mock_playback)

    # Create a mock event with audio inline_data
    event = MagicMock()
    event.interrupted = False
    event.partial = False
    event.turn_complete = False
    event.input_transcription = None
    event.output_transcription = None

    part = MagicMock()
    part.text = None
    part.inline_data = MagicMock()
    part.inline_data.mime_type = "audio/pcm;rate=24000"
    part.inline_data.data = b"\x00\x01" * 100

    event.content = MagicMock()
    event.content.parts = [part]

    session._handle_event(event)
    mock_playback.enqueue.assert_called_once_with(b"\x00\x01" * 100)


def test_handle_event_interrupted_clears_playback():
    """Interrupted event should clear the playback buffer."""
    mock_playback = MagicMock()
    session = LivePlannerSession(config=CloudConfig(), audio_playback=mock_playback)

    event = MagicMock()
    event.interrupted = True
    event.content = None
    event.input_transcription = None
    event.output_transcription = None
    event.turn_complete = False

    session._handle_event(event)
    mock_playback.clear.assert_called_once()


def test_handle_event_transcriptions():
    """Transcription events should update session state."""
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))

    event = MagicMock()
    event.interrupted = False
    event.content = None
    event.turn_complete = False
    event.input_transcription = MagicMock()
    event.input_transcription.text = "pick up the cube"
    event.output_transcription = MagicMock()
    event.output_transcription.text = "Starting pick skill"

    session._handle_event(event)
    assert session.state.last_transcription_in == "pick up the cube"
    assert session.state.last_transcription_out == "Starting pick skill"


def test_handle_event_turn_complete_drains_commands():
    """turn_complete should move ctx.commands to pending_commands and set event."""
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))
    cmd = _make_cmd()
    session._ctx.commands.append(cmd)

    event = MagicMock()
    event.interrupted = False
    event.content = None
    event.input_transcription = None
    event.output_transcription = None
    event.turn_complete = True

    session._handle_event(event)

    # Commands should be in pending queue
    assert not session._pending_commands.empty()
    drained = session.drain_pending_commands()
    assert len(drained) == 1
    assert drained[0] is cmd

    # ctx should be cleared
    assert len(session._ctx.commands) == 0
    assert len(session._ctx.used_tools) == 0

    # Event should be set
    assert session._turn_complete_event.is_set()


def test_handle_event_text_updates_reasoning():
    """Non-partial text in event should update last_reasoning."""
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))

    event = MagicMock()
    event.interrupted = False
    event.partial = False
    event.turn_complete = False
    event.input_transcription = None
    event.output_transcription = None

    part = MagicMock()
    part.text = "I'll start a pick skill."
    part.inline_data = None

    event.content = MagicMock()
    event.content.parts = [part]

    session._handle_event(event)
    assert session.last_reasoning == "I'll start a pick skill."
    assert session.state.last_text_out == "I'll start a pick skill."


# ---------------------------------------------------------------------------
# RunConfig building
# ---------------------------------------------------------------------------


def test_build_run_config_audio_enabled():
    session = LivePlannerSession(config=CloudConfig(audio_enabled=True, voice_name="Kore"))
    rc = session._build_run_config()
    assert "AUDIO" in rc.response_modalities
    assert rc.speech_config is not None
    assert rc.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"


def test_build_run_config_audio_disabled():
    session = LivePlannerSession(config=CloudConfig(audio_enabled=False))
    rc = session._build_run_config()
    assert rc.response_modalities == ["TEXT"]
    assert rc.speech_config is None


def test_build_run_config_session_resumption():
    session = LivePlannerSession(config=CloudConfig(session_resumption=True))
    rc = session._build_run_config()
    assert rc.session_resumption is not None
    assert rc.session_resumption.transparent is True


def test_build_run_config_no_session_resumption():
    session = LivePlannerSession(config=CloudConfig(session_resumption=False))
    rc = session._build_run_config()
    assert rc.session_resumption is None


def test_build_run_config_context_compression():
    session = LivePlannerSession(config=CloudConfig(context_compression=True))
    rc = session._build_run_config()
    assert rc.context_window_compression is not None


def test_build_run_config_no_context_compression():
    session = LivePlannerSession(config=CloudConfig(context_compression=False))
    rc = session._build_run_config()
    assert rc.context_window_compression is None
