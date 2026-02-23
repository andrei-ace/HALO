from __future__ import annotations

import math
from dataclasses import dataclass, field

from halo.contracts.actions import Action, ActionChunk


@dataclass
class _ChunkEntry:
    chunk: ActionChunk
    push_tick: int


class TemporalEnsemblingBuffer:
    """
    ACT temporal ensembling buffer.

    Overlapping chunks are blended per-timestep with exponential decay weighting:
        w = exp(-temp * (current_tick - push_tick))
    Newer chunks (smaller age) receive higher weight. temp=0.0 gives equal weights.

    Not thread-safe — callers must serialise via asyncio.Lock.
    """

    def __init__(self, temp: float = 0.01) -> None:
        self._entries: list[_ChunkEntry] = []
        self._tick: int = 0
        self._temp: float = temp

    # --- Public API ---

    def push_chunk(self, chunk: ActionChunk) -> None:
        """Append chunk tracked at the current tick."""
        self._entries.append(_ChunkEntry(chunk=chunk, push_tick=self._tick))
        self._prune()

    def pop_action(self) -> Action | None:
        """
        Blend all chunks that cover the current tick, advance tick, return blended action.
        Returns None if no chunk covers the current tick.
        """
        self._prune()
        if not self._entries:
            return None

        weighted: list[tuple[Action, float]] = []
        for entry in self._entries:
            offset = self._tick - entry.push_tick
            if 0 <= offset < len(entry.chunk.actions):
                w = math.exp(-self._temp * offset)
                weighted.append((entry.chunk.actions[offset], w))

        if not weighted:
            self._tick += 1
            return None

        total_w = sum(w for _, w in weighted)
        blended = Action(
            dx=sum(a.dx * w for a, w in weighted) / total_w,
            dy=sum(a.dy * w for a, w in weighted) / total_w,
            dz=sum(a.dz * w for a, w in weighted) / total_w,
            droll=sum(a.droll * w for a, w in weighted) / total_w,
            dpitch=sum(a.dpitch * w for a, w in weighted) / total_w,
            dyaw=sum(a.dyaw * w for a, w in weighted) / total_w,
            gripper_cmd=sum(a.gripper_cmd * w for a, w in weighted) / total_w,
        )
        self._tick += 1
        return blended

    def trim_to_ms(self, target_ms: int, rate_hz: float) -> int:
        """
        Trim all chunks so no action extends beyond target_ms from now.
        Returns total number of actions removed.
        Called on PHASE_ENTER to discard stale tail actions.
        """
        target_ticks = int(target_ms * rate_hz / 1000)
        cutoff_tick = self._tick + target_ticks

        removed = 0
        new_entries: list[_ChunkEntry] = []
        for entry in self._entries:
            keep_len = cutoff_tick - entry.push_tick
            if keep_len <= 0:
                removed += len(entry.chunk.actions)
                # discard entire entry
            elif keep_len < len(entry.chunk.actions):
                removed += len(entry.chunk.actions) - keep_len
                new_chunk = ActionChunk(
                    chunk_id=entry.chunk.chunk_id,
                    arm_id=entry.chunk.arm_id,
                    phase_id=entry.chunk.phase_id,
                    actions=entry.chunk.actions[:keep_len],
                    ts_ms=entry.chunk.ts_ms,
                )
                new_entries.append(_ChunkEntry(chunk=new_chunk, push_tick=entry.push_tick))
            else:
                new_entries.append(entry)

        self._entries = new_entries
        return removed

    def fill_ms(self, rate_hz: float) -> int:
        """Estimate of future buffered duration in milliseconds."""
        if not self._entries or rate_hz <= 0:
            return 0
        max_covered = max(e.push_tick + len(e.chunk.actions) for e in self._entries)
        remaining = max(0, max_covered - self._tick)
        return int(remaining * 1000 / rate_hz)

    def is_low(self, threshold_ms: int, rate_hz: float) -> bool:
        return self.fill_ms(rate_hz) < threshold_ms

    @property
    def size(self) -> int:
        """Future ticks serviceable (0 if empty)."""
        if not self._entries:
            return 0
        max_covered = max(e.push_tick + len(e.chunk.actions) for e in self._entries)
        return max(0, max_covered - self._tick)

    # --- Private ---

    def _prune(self) -> None:
        """Remove entries whose last action has already been consumed."""
        self._entries = [
            e for e in self._entries
            if e.push_tick + len(e.chunk.actions) > self._tick
        ]
