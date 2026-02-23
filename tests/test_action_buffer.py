"""Tests for ActionBuffer: push, pop, trim, fill, is_low."""

import pytest

from halo.contracts.actions import Action, ActionChunk
from halo.contracts.enums import PhaseId
from halo.services.control_service.action_buffer import ActionBuffer

RATE = 50.0  # Hz


def _chunk(
    n: int,
    arm_id: str = "arm0",
    phase: PhaseId = PhaseId.APPROACH_PREGRASP,
) -> ActionChunk:
    actions = tuple(Action(float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(n))
    return ActionChunk(chunk_id=f"c-{n}", arm_id=arm_id, phase_id=phase, actions=actions, ts_ms=0)


def test_push_then_pop_oldest_first():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(3))
    assert buf.pop_action().dx == 0.0
    assert buf.pop_action().dx == 1.0
    assert buf.pop_action().dx == 2.0


def test_pop_empty_returns_none():
    buf = ActionBuffer()
    assert buf.pop_action() is None


def test_fill_ms_matches_push_count():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(5))
    # 5 actions at 50 Hz → 100 ms
    assert buf.fill_ms(RATE) == 100


def test_fill_ms_empty_is_zero():
    buf = ActionBuffer()
    assert buf.fill_ms(RATE) == 0


def test_trim_to_ms_removes_from_right():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(10))  # 200 ms at 50 Hz
    removed = buf.trim_to_ms(100, RATE)  # keep ≤ 5 actions
    assert removed == 5
    assert buf.size == 5


def test_trim_to_ms_never_below_zero():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(2))
    removed = buf.trim_to_ms(0, RATE)
    assert buf.size == 0
    assert removed == 2


def test_trim_to_ms_no_op_when_already_within():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(3))  # 60 ms at 50 Hz
    removed = buf.trim_to_ms(200, RATE)  # target 10 actions — nothing to trim
    assert removed == 0
    assert buf.size == 3


def test_is_low_triggers_at_correct_threshold():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(4))  # 80 ms at 50 Hz
    # 80 < 100 → low
    assert buf.is_low(threshold_ms=100, rate_hz=RATE) is True


def test_is_low_false_when_above_threshold():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(6))  # 120 ms at 50 Hz
    assert buf.is_low(threshold_ms=100, rate_hz=RATE) is False


def test_size_property():
    buf = ActionBuffer()
    assert buf.size == 0
    buf.push_chunk(_chunk(7))
    assert buf.size == 7


def test_multiple_chunks_appended_in_order():
    buf = ActionBuffer()
    buf.push_chunk(_chunk(2))      # dx=0, dx=1
    buf.push_chunk(_chunk(2))      # dx=0, dx=1 (second chunk)
    assert buf.size == 4
    assert buf.pop_action().dx == 0.0  # first chunk first
    assert buf.pop_action().dx == 1.0
    assert buf.pop_action().dx == 0.0  # second chunk after
    assert buf.pop_action().dx == 1.0
