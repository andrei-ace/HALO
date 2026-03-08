"""Audio capture and playback for Gemini Live API voice interaction.

AudioCapture wraps a sounddevice InputStream (16kHz mono PCM).
AudioPlayback wraps a sounddevice OutputStream (24kHz mono PCM).
Both are optional — the Live API works in text-only mode without audio.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioState:
    """Observable audio state for TUI display."""

    capturing: bool = False
    playing: bool = False
    muted: bool = False
    input_level: float = 0.0
    output_level: float = 0.0


def _lazy_import_sounddevice():
    """Import sounddevice lazily, raising a clear error if not installed."""
    try:
        import sounddevice  # noqa: F811

        return sounddevice
    except ImportError:
        msg = (
            "sounddevice is required for audio I/O. Install it with: uv sync --extra dev  (or: pip install sounddevice)"
        )
        raise ImportError(msg) from None


class AudioCapture:
    """Captures 16kHz mono PCM audio from the default input device.

    Calls ``on_audio(pcm_bytes)`` per chunk (~100ms = 3200 bytes at 16kHz int16).
    """

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
    ) -> None:
        self._on_audio = on_audio
        self._sample_rate = sample_rate
        self._chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self._muted = False
        self._stream = None
        self._state = AudioState()

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._muted = value
        self._state.muted = value

    @property
    def state(self) -> AudioState:
        return self._state

    def start(self) -> None:
        sd = _lazy_import_sounddevice()
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        self._state.capturing = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._state.capturing = False
        self._state.input_level = 0.0

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        pcm = bytes(indata)
        # Compute simple RMS level (0..1 range approximation)
        if len(pcm) >= 2:
            import struct

            samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            self._state.input_level = min(rms / 32768.0, 1.0)
        if not self._muted:
            self._on_audio(pcm)


class AudioPlayback:
    """Plays 24kHz mono PCM audio on the default output device.

    Thread-safe ``enqueue()`` adds PCM chunks to a FIFO buffer.
    ``clear()`` flushes the buffer (e.g. on interruption).
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self._state = AudioState()

    @property
    def state(self) -> AudioState:
        return self._state

    def enqueue(self, pcm_bytes: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm_bytes)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
        self._state.output_level = 0.0

    def start(self) -> None:
        sd = _lazy_import_sounddevice()
        self._stream = sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()
        self._state.playing = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._state.playing = False
        self._state.output_level = 0.0

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        nbytes = frames * 2  # int16 = 2 bytes per sample
        with self._lock:
            available = min(len(self._buffer), nbytes)
            if available > 0:
                outdata[:available] = bytes(self._buffer[:available])
                del self._buffer[:available]
                # Compute output level
                import struct

                samples = struct.unpack(f"<{available // 2}h", outdata[:available])
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                self._state.output_level = min(rms / 32768.0, 1.0)
            else:
                self._state.output_level = 0.0
            # Zero-fill remaining
            if available < nbytes:
                outdata[available:nbytes] = b"\x00" * (nbytes - available)


@dataclass
class AudioComponents:
    """Bundle of audio capture + playback + shared state, or None if unavailable."""

    capture: AudioCapture | None = None
    playback: AudioPlayback | None = None
    available: bool = False


def make_audio_components(
    on_audio: Callable[[bytes], None],
    input_sample_rate: int = 16000,
    output_sample_rate: int = 24000,
) -> AudioComponents:
    """Try to create audio components; returns AudioComponents with available=False on failure."""
    try:
        _lazy_import_sounddevice()
        capture = AudioCapture(on_audio=on_audio, sample_rate=input_sample_rate)
        playback = AudioPlayback(sample_rate=output_sample_rate)
        return AudioComponents(capture=capture, playback=playback, available=True)
    except ImportError:
        logger.warning("sounddevice not installed — audio disabled, text-only Live API mode")
        return AudioComponents()
