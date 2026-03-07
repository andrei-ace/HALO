"""Tests for SkillQueue."""

from halo.contracts.enums import SkillName
from halo.services.skill_runner_service.queue import SkillQueue
from halo.services.skill_runner_service.skill_run import QueuedSkill


def _qs(name: str = "cube-1", skill: SkillName = SkillName.PICK) -> QueuedSkill:
    return QueuedSkill(
        skill_name=skill,
        skill_run_id=f"run-{name}",
        target_handle=name,
        variant="default",
        options={},
        enqueued_at_ms=1000,
    )


def test_enqueue_dequeue():
    q = SkillQueue()
    assert q.size == 0
    assert q.enqueue(_qs("a"))
    assert q.size == 1
    item = q.dequeue()
    assert item is not None
    assert item.target_handle == "a"
    assert q.size == 0


def test_dequeue_empty_returns_none():
    q = SkillQueue()
    assert q.dequeue() is None


def test_peek_does_not_remove():
    q = SkillQueue()
    q.enqueue(_qs("a"))
    assert q.peek() is not None
    assert q.size == 1


def test_clear_returns_count():
    q = SkillQueue()
    q.enqueue(_qs("a"))
    q.enqueue(_qs("b"))
    assert q.clear() == 2
    assert q.size == 0


def test_max_size_rejects():
    q = SkillQueue(max_size=2)
    assert q.enqueue(_qs("a"))
    assert q.enqueue(_qs("b"))
    assert not q.enqueue(_qs("c"))
    assert q.size == 2


def test_fifo_order():
    q = SkillQueue()
    q.enqueue(_qs("a"))
    q.enqueue(_qs("b"))
    q.enqueue(_qs("c"))
    assert q.dequeue().target_handle == "a"
    assert q.dequeue().target_handle == "b"
    assert q.dequeue().target_handle == "c"


def test_items_returns_tuple():
    q = SkillQueue()
    q.enqueue(_qs("a"))
    q.enqueue(_qs("b"))
    items = q.items
    assert isinstance(items, tuple)
    assert len(items) == 2
