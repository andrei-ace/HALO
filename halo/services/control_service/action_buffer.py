from __future__ import annotations

from collections import deque

from halo.contracts.actions import JointPositionAction, JointPositionChunk


class ActionBuffer:
    """
    Ordered deque of actions fed from JointPositionChunks.
    Not thread-safe on its own — callers must serialise via asyncio.Lock.
    """

    def __init__(self) -> None:
        self._deque: deque[JointPositionAction] = deque()

    def push_chunk(self, chunk: JointPositionChunk) -> None:
        """Append all actions from chunk to the right of the deque."""
        self._deque.extend(chunk.actions)

    def pop_action(self) -> JointPositionAction | None:
        """Pop leftmost (oldest) action; returns None if empty."""
        if not self._deque:
            return None
        return self._deque.popleft()

    def trim_to_ms(self, target_ms: int, rate_hz: float) -> int:
        """Drop from right until len ≤ target_ms * rate_hz / 1000. Returns number removed."""
        target_len = int(target_ms * rate_hz / 1000)
        removed = 0
        while len(self._deque) > target_len:
            self._deque.pop()
            removed += 1
        return removed

    def fill_ms(self, rate_hz: float) -> int:
        """Estimate of buffered duration in milliseconds."""
        if rate_hz <= 0:
            return 0
        return int(len(self._deque) / rate_hz * 1000)

    def is_low(self, threshold_ms: int, rate_hz: float) -> bool:
        return self.fill_ms(rate_hz) < threshold_ms

    @property
    def size(self) -> int:
        return len(self._deque)
