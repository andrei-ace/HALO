"""ContextStore — append-only journal + snapshot for backend handoff.

Captures *what the planner knows and has decided* so that context can be
transferred when switching between local and cloud backends without losing
plan/history/summary.

All in-memory for v0 (bounded to MAX_ENTRIES).  SQLite in v1 if needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_ENTRIES = 200


@dataclass(frozen=True)
class ContextEntry:
    """Single entry in the cognitive journal."""

    cursor: int  # monotonic, auto-assigned by ContextStore
    ts_ms: int
    epoch: int  # lease epoch at time of recording
    backend: str  # "local" or "cloud"
    entry_type: str  # "decision", "scene", "event", "operator"
    summary: str
    data: dict = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Point-in-time cognitive state for handoff."""

    ts_ms: int
    epoch: int
    active_target_handle: str | None
    held_object_handle: str | None
    known_scene_handles: list[str]
    last_scene_description: str
    pending_operator_instruction: str | None
    recent_decisions: list[str]
    cursor_applied_up_to: int


class ContextStore:
    """Append-only journal with snapshot/handoff support.

    Thread-safe for single-writer usage (PlannerService ticks are serialized).
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: list[ContextEntry] = []
        self._next_cursor: int = 0
        self._max_entries = max_entries
        # Tracked state for snapshot generation
        self._active_target_handle: str | None = None
        self._held_object_handle: str | None = None
        self._known_scene_handles: list[str] = []
        self._last_scene_description: str = ""
        self._pending_operator_instruction: str | None = None

    def append(
        self,
        epoch: int,
        backend: str,
        entry_type: str,
        summary: str,
        data: dict | None = None,
    ) -> ContextEntry:
        """Append a new entry to the journal.

        Updates tracked state based on entry_type:
        - "scene": updates known_scene_handles + last_scene_description
        - "operator": sets pending_operator_instruction
        - "decision": clears pending_operator_instruction (consumed)
        """
        entry = ContextEntry(
            cursor=self._next_cursor,
            ts_ms=int(time.monotonic() * 1000),
            epoch=epoch,
            backend=backend,
            entry_type=entry_type,
            summary=summary,
            data=data or {},
        )
        self._next_cursor += 1
        self._entries.append(entry)

        # Update tracked state
        if entry_type == "scene":
            self._last_scene_description = summary
            self._known_scene_handles = data.get("handles", []) if data else []
        elif entry_type == "operator":
            self._pending_operator_instruction = summary
        elif entry_type == "decision":
            self._pending_operator_instruction = None

        # Trim if over limit
        if len(self._entries) > self._max_entries:
            excess = len(self._entries) - self._max_entries
            self._entries = self._entries[excess:]

        return entry

    def set_active_target(self, handle: str | None) -> None:
        self._active_target_handle = handle

    def set_held_object(self, handle: str | None) -> None:
        self._held_object_handle = handle

    def take_snapshot(self, epoch: int) -> ContextSnapshot:
        """Build a ContextSnapshot from current journal + tracked state."""
        recent_decisions = [e.summary for e in self._entries if e.entry_type == "decision"][-5:]  # last 5 decisions

        return ContextSnapshot(
            ts_ms=int(time.monotonic() * 1000),
            epoch=epoch,
            active_target_handle=self._active_target_handle,
            held_object_handle=self._held_object_handle,
            known_scene_handles=list(self._known_scene_handles),
            last_scene_description=self._last_scene_description,
            pending_operator_instruction=self._pending_operator_instruction,
            recent_decisions=recent_decisions,
            cursor_applied_up_to=self._next_cursor - 1 if self._entries else -1,
        )

    def get_handoff_context(self, epoch: int) -> str:
        """Generate a text summary suitable for injection as the first user
        message in a new backend's ADK session.

        This gives the new backend enough context to continue seamlessly.
        """
        snap = self.take_snapshot(epoch)
        parts: list[str] = ["[Context handoff from previous backend]"]

        if snap.last_scene_description:
            parts.append(f"Last scene analysis: {snap.last_scene_description}")

        if snap.known_scene_handles:
            parts.append(f"Known objects: {', '.join(snap.known_scene_handles)}")

        if snap.active_target_handle:
            parts.append(f"Active target: {snap.active_target_handle}")

        if snap.held_object_handle:
            parts.append(f"Currently holding: {snap.held_object_handle}")

        if snap.recent_decisions:
            parts.append("Recent decisions:")
            for d in snap.recent_decisions:
                parts.append(f"  - {d}")

        if snap.pending_operator_instruction:
            parts.append(f"Pending operator instruction: {snap.pending_operator_instruction}")

        return "\n".join(parts)

    def get_entries_after(self, cursor: int, limit: int = 50) -> list[ContextEntry]:
        """Return entries with cursor > the given value, up to limit."""
        return [e for e in self._entries if e.cursor > cursor][:limit]

    @property
    def latest_cursor(self) -> int:
        """The cursor of the most recent entry, or -1 if empty."""
        return self._next_cursor - 1 if self._entries else -1

    def __len__(self) -> int:
        return len(self._entries)
