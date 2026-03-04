"""Unit tests for audio_io — AudioState, AudioCapture, AudioPlayback buffer logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from halo.cognitive.audio_io import AudioCapture, AudioComponents, AudioPlayback, AudioState, make_audio_components


def test_audio_state_defaults():
    state = AudioState()
    assert state.capturing is False
    assert state.playing is False
    assert state.muted is False
    assert state.input_level == 0.0
    assert state.output_level == 0.0


def test_playback_enqueue_and_clear():
    """AudioPlayback buffer enqueue/clear works without a real audio device."""
    pb = AudioPlayback(sample_rate=24000)
    assert len(pb._buffer) == 0

    pb.enqueue(b"\x00\x01" * 100)
    assert len(pb._buffer) == 200

    pb.enqueue(b"\x02\x03" * 50)
    assert len(pb._buffer) == 300

    pb.clear()
    assert len(pb._buffer) == 0


def test_playback_clear_resets_output_level():
    pb = AudioPlayback()
    pb._state.output_level = 0.5
    pb.clear()
    assert pb.state.output_level == 0.0


def test_capture_mute_property():
    capture = AudioCapture(on_audio=lambda b: None)
    assert capture.muted is False
    assert capture.state.muted is False

    capture.muted = True
    assert capture.muted is True
    assert capture.state.muted is True


def test_capture_callback_muted_does_not_forward():
    """When muted, the callback should not call on_audio."""
    received = []
    capture = AudioCapture(on_audio=received.append)
    capture.muted = True

    # Simulate a callback with fake PCM data
    fake_pcm = b"\x00\x01" * 1600  # 100ms at 16kHz int16
    capture._callback(fake_pcm, 1600, None, None)

    assert len(received) == 0


def test_capture_callback_unmuted_forwards():
    """When unmuted, the callback should call on_audio with PCM bytes."""
    received = []
    capture = AudioCapture(on_audio=received.append)

    fake_pcm = b"\x00\x01" * 1600
    capture._callback(fake_pcm, 1600, None, None)

    assert len(received) == 1
    assert received[0] == fake_pcm


def test_make_audio_components_import_failure():
    """When sounddevice is not installed, make_audio_components returns unavailable."""
    with patch("halo.cognitive.audio_io._lazy_import_sounddevice", side_effect=ImportError("no sounddevice")):
        result = make_audio_components(on_audio=lambda b: None)

    assert isinstance(result, AudioComponents)
    assert result.available is False
    assert result.capture is None
    assert result.playback is None


def test_make_audio_components_success():
    """When sounddevice is available, make_audio_components returns working components."""
    mock_sd = MagicMock()
    with patch("halo.cognitive.audio_io._lazy_import_sounddevice", return_value=mock_sd):
        result = make_audio_components(on_audio=lambda b: None, input_sample_rate=16000, output_sample_rate=24000)

    assert result.available is True
    assert result.capture is not None
    assert result.playback is not None
