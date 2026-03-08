from __future__ import annotations

from collections import deque

from halo.services.skill_runner_service.skill_run import QueuedSkill


class SkillQueue:
    def __init__(self, max_size: int = 16) -> None:
        self._max_size = max_size
        self._items: deque[QueuedSkill] = deque()

    def enqueue(self, item: QueuedSkill) -> bool:
        if len(self._items) >= self._max_size:
            return False
        # Skip duplicates: same (skill_name, target_handle) already queued
        for existing in self._items:
            if existing.skill_name == item.skill_name and existing.target_handle == item.target_handle:
                return True  # silently skip, not an error
        self._items.append(item)
        return True

    def dequeue(self) -> QueuedSkill | None:
        if not self._items:
            return None
        return self._items.popleft()

    def peek(self) -> QueuedSkill | None:
        if not self._items:
            return None
        return self._items[0]

    def clear(self) -> int:
        count = len(self._items)
        self._items.clear()
        return count

    @property
    def items(self) -> tuple[QueuedSkill, ...]:
        return tuple(self._items)

    @property
    def size(self) -> int:
        return len(self._items)
