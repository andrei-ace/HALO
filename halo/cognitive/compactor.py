"""MessageHistory — UUID-tracked message list for compaction detection and cross-backend sync.

ADK handles the actual compaction via ``CompactionPlugin`` (see
``halo/cognitive/compaction_plugin.py``).  This module provides:

- ``MessageRecord``: a UUID-tagged message (user or model)
- ``MessageHistory``: parallel tracking alongside ADK sessions
- ``CompactionResult``: outcome of a detected compaction event
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class MessageRecord:
    """Single message in the parallel tracking list."""

    msg_id: str  # uuid4 hex
    role: str  # "user" | "model"
    text: str  # full message text
    ts_ms: int
    is_summary: bool = False  # True if this replaces compacted messages


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a detected compaction event."""

    summary: str  # compacted summary text
    up_to_msg_id: str  # last compacted message UUID
    compacted_count: int
    retained_count: int
    ts_ms: int


class MessageHistory:
    """Parallel UUID-tracked message list alongside an ADK session.

    Each ``append()`` call assigns a UUID to a message so that compaction
    boundaries can be precisely identified and propagated across backends.
    """

    def __init__(self) -> None:
        self._records: list[MessageRecord] = []

    def append(self, role: str, text: str) -> str:
        """Append a message and return its UUID."""
        msg_id = uuid.uuid4().hex
        record = MessageRecord(
            msg_id=msg_id,
            role=role,
            text=text,
            ts_ms=int(time.monotonic() * 1000),
        )
        self._records.append(record)
        return msg_id

    def count(self) -> int:
        return len(self._records)

    def get_all(self) -> list[MessageRecord]:
        return list(self._records)

    def get_after(self, msg_id: str) -> list[MessageRecord]:
        """Return records after the given msg_id (exclusive)."""
        for i, rec in enumerate(self._records):
            if rec.msg_id == msg_id:
                return list(self._records[i + 1 :])
        return list(self._records)  # msg_id not found — return all

    def apply_compaction(self, up_to_msg_id: str, summary_text: str) -> CompactionResult:
        """Replace records up to ``up_to_msg_id`` (inclusive) with a single summary record.

        Returns a ``CompactionResult`` with counts and the summary.
        Raises ``ValueError`` if ``up_to_msg_id`` is not found.
        """
        cut_idx = -1
        for i, rec in enumerate(self._records):
            if rec.msg_id == up_to_msg_id:
                cut_idx = i
                break
        if cut_idx < 0:
            msg = f"msg_id {up_to_msg_id!r} not found in history"
            raise ValueError(msg)

        compacted_count = cut_idx + 1
        retained = self._records[cut_idx + 1 :]

        summary_record = MessageRecord(
            msg_id=uuid.uuid4().hex,
            role="model",
            text=summary_text,
            ts_ms=int(time.monotonic() * 1000),
            is_summary=True,
        )
        self._records = [summary_record, *retained]

        return CompactionResult(
            summary=summary_text,
            up_to_msg_id=up_to_msg_id,
            compacted_count=compacted_count,
            retained_count=len(retained),
            ts_ms=int(time.time() * 1000),
        )

    def replace_all(self, records: list[MessageRecord]) -> None:
        """Replace all records (used for cross-backend mirroring)."""
        self._records = list(records)

    def truncate(self, count: int) -> None:
        """Truncate history to the first *count* records (rollback)."""
        self._records = self._records[:count]

    def clear(self) -> None:
        self._records.clear()
