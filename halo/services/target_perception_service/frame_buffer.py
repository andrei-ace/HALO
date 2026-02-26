from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapturedFrame:
    """A single camera frame captured during VLM inference.

    The ``image`` field is opaque — its concrete type (numpy array, PIL Image,
    bytes, …) is determined by the ``CaptureFn`` implementation.
    """

    image: Any
    ts_ms: int
    arm_id: str


class FrameRingBuffer:
    """Append-only frame buffer with read-cursor support for replay.

    Used during VLM inference to accumulate camera frames.  When the VLM
    completes, the replay task reads frames incrementally via
    :meth:`read_from` while ``tick()`` keeps appending new ones.

    Not thread-safe — callers must serialise via asyncio (single event loop).
    """

    def __init__(self, max_size: int = 300) -> None:
        self._frames: list[CapturedFrame] = []
        self._active: bool = False
        self._max_size = max_size

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Clear any previous data and begin accepting pushes."""
        self._frames.clear()
        self._active = True

    def stop(self) -> None:
        """Stop accepting new pushes.  Buffered frames remain readable."""
        self._active = False

    def clear(self) -> None:
        """Discard all buffered frames and deactivate."""
        self._frames.clear()
        self._active = False

    # -- write ----------------------------------------------------------------

    def push(self, frame: CapturedFrame) -> None:
        """Append *frame* if the buffer is active and below the safety cap."""
        if self._active and len(self._frames) < self._max_size:
            self._frames.append(frame)

    # -- read -----------------------------------------------------------------

    def read_from(self, idx: int) -> tuple[list[CapturedFrame], int]:
        """Return ``(frames[idx:], new_cursor)`` for incremental replay.

        The caller is responsible for advancing its own read cursor.
        """
        batch = self._frames[idx:]
        return batch, len(self._frames)

    # -- introspection --------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    def __len__(self) -> int:
        return len(self._frames)
