"""Tests for TemporalEnsemblingBuffer."""

import math

import pytest

from halo.contracts.actions import Action, ActionChunk
from halo.contracts.enums import PhaseId
from halo.services.control_service.te_buffer import TemporalEnsemblingBuffer

RATE = 10.0  # Hz


def _chunk(n: int, dx_start: float = 0.0, arm: str = "arm0") -> ActionChunk:
    """Create an ActionChunk with n actions; dx increments from dx_start."""
    actions = tuple(
        Action(dx=float(dx_start + i), dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
        for i in range(n)
    )
    return ActionChunk(
        chunk_id=f"c-{n}-{dx_start}",
        arm_id=arm,
        phase_id=PhaseId.APPROACH_PREGRASP,
        actions=actions,
        ts_ms=0,
    )


def _const_chunk(n: int, dx: float, arm: str = "arm0") -> ActionChunk:
    """Create an ActionChunk with n actions all having the same dx."""
    actions = tuple(
        Action(dx=dx, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper_cmd=0.0)
        for _ in range(n)
    )
    return ActionChunk(
        chunk_id=f"const-{n}-{dx}",
        arm_id=arm,
        phase_id=PhaseId.APPROACH_PREGRASP,
        actions=actions,
        ts_ms=0,
    )


def _buf(temp: float = 0.0) -> TemporalEnsemblingBuffer:
    return TemporalEnsemblingBuffer(temp=temp)


# ---------------------------------------------------------------------------
# Single-chunk (FIFO-compatible) tests
# ---------------------------------------------------------------------------

def test_single_chunk_pops_in_order():
    buf = _buf()
    buf.push_chunk(_chunk(3))
    assert buf.pop_action().dx == 0.0
    assert buf.pop_action().dx == 1.0
    assert buf.pop_action().dx == 2.0


def test_pop_empty_returns_none():
    buf = _buf()
    assert buf.pop_action() is None


def test_fill_ms_single_chunk():
    buf = _buf()
    buf.push_chunk(_chunk(5))
    assert buf.fill_ms(RATE) == 500  # 5 actions at 10 Hz → 500 ms


def test_fill_ms_empty_is_zero():
    buf = _buf()
    assert buf.fill_ms(RATE) == 0


def test_is_low_when_below_threshold():
    buf = _buf()
    buf.push_chunk(_chunk(2))  # 200 ms at 10 Hz
    assert buf.is_low(threshold_ms=300, rate_hz=RATE) is True


def test_is_low_false_when_above_threshold():
    buf = _buf()
    buf.push_chunk(_chunk(5))  # 500 ms at 10 Hz
    assert buf.is_low(threshold_ms=300, rate_hz=RATE) is False


def test_size_after_push():
    buf = _buf()
    buf.push_chunk(_chunk(7))
    assert buf.size == 7


def test_size_decrements_after_pop():
    buf = _buf()
    buf.push_chunk(_chunk(4))
    buf.pop_action()
    assert buf.size == 3


# ---------------------------------------------------------------------------
# Ensembling tests (temp=0.0 → equal weights → simple average)
# ---------------------------------------------------------------------------

def test_two_chunks_same_tick_blend_equal_weight():
    """Two chunks pushed at the same tick get equal weight → 50/50 blend."""
    buf = _buf(temp=0.0)
    buf.push_chunk(_const_chunk(3, dx=0.0))
    buf.push_chunk(_const_chunk(3, dx=1.0))
    result = buf.pop_action()
    assert result is not None
    assert abs(result.dx - 0.5) < 1e-9


def test_three_chunks_blend_average():
    """Three equal-weight chunks → simple mean."""
    buf = _buf(temp=0.0)
    buf.push_chunk(_const_chunk(3, dx=0.0))
    buf.push_chunk(_const_chunk(3, dx=1.0))
    buf.push_chunk(_const_chunk(3, dx=2.0))
    result = buf.pop_action()
    assert result is not None
    assert abs(result.dx - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Weight-decay tests (temp=1.0 → strong decay)
# ---------------------------------------------------------------------------

def test_newer_chunk_gets_higher_weight():
    """
    chunk_A pushed at tick 0, chunk_B pushed at tick 1.
    At tick 1: chunk_A has age=1 (w≈0.368), chunk_B has age=0 (w=1.0).
    Blend is biased toward chunk_B.
    """
    buf = _buf(temp=1.0)
    chunk_a = _const_chunk(3, dx=0.0)
    buf.push_chunk(chunk_a)
    buf.pop_action()  # consume tick 0; _tick advances to 1

    chunk_b = _const_chunk(3, dx=1.0)
    buf.push_chunk(chunk_b)
    result = buf.pop_action()  # blend at tick 1

    assert result is not None
    # chunk_A: offset=1, w=exp(-1)≈0.368, dx=0.0
    # chunk_B: offset=0, w=1.0, dx=1.0
    # blend = 1.0 / (1.0 + 0.368) ≈ 0.731
    expected = 1.0 / (1.0 + math.exp(-1.0))
    assert abs(result.dx - expected) < 1e-9
    assert result.dx > 0.5  # biased toward newer chunk_B


# ---------------------------------------------------------------------------
# Expiry tests
# ---------------------------------------------------------------------------

def test_expired_chunk_not_blended():
    """A chunk with 1 action expires after one pop; second pop returns None."""
    buf = _buf()
    buf.push_chunk(_chunk(1))
    buf.pop_action()
    assert buf.pop_action() is None


def test_second_chunk_fills_after_first_expires():
    """
    chunk_A has 1 action, chunk_B has 3 actions; both pushed at tick 0.
    Pop 1 → blend of both.
    Pop 2 → only chunk_B (chunk_A expired).
    Pop 3 → only chunk_B.
    """
    buf = _buf(temp=0.0)
    chunk_a = _const_chunk(1, dx=10.0)
    chunk_b = _const_chunk(3, dx=20.0)
    buf.push_chunk(chunk_a)
    buf.push_chunk(chunk_b)

    r0 = buf.pop_action()
    assert r0 is not None
    # both contribute equally at tick=0
    assert abs(r0.dx - 15.0) < 1e-9

    r1 = buf.pop_action()
    assert r1 is not None
    # only chunk_B at offset=1
    assert abs(r1.dx - 20.0) < 1e-9

    r2 = buf.pop_action()
    assert r2 is not None
    # only chunk_B at offset=2
    assert abs(r2.dx - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Trim tests
# ---------------------------------------------------------------------------

def test_trim_single_chunk_at_boundary():
    """Push chunk(10); trim_to_ms(300, 10) → keep 3 actions; size=3."""
    buf = _buf()
    buf.push_chunk(_chunk(10))
    buf.trim_to_ms(300, RATE)
    assert buf.size == 3


def test_trim_removes_entire_chunk_if_beyond_cutoff():
    """
    Push chunk(2), exhaust it, then push chunk(5).
    trim_to_ms(0) → discard all; size=0.
    """
    buf = _buf()
    buf.push_chunk(_chunk(2))
    buf.pop_action()
    buf.pop_action()
    buf.pop_action()  # tick now = 3; chunk(2) already expired

    buf.push_chunk(_chunk(5))  # pushed at tick=3
    buf.trim_to_ms(0, RATE)   # cutoff_tick = 3+0 = 3; keep_len = 3-3 = 0
    assert buf.size == 0


def test_trim_noop_when_already_within():
    """Push chunk(3); trim_to_ms(500, 10) target=5 ticks > len=3 → no removal."""
    buf = _buf()
    buf.push_chunk(_chunk(3))
    removed = buf.trim_to_ms(500, RATE)
    assert removed == 0
    assert buf.size == 3


def test_trim_multiple_chunks():
    """
    Push chunk_A(5) and chunk_B(5) at tick 0.
    trim_to_ms(200, 10) → cutoff_tick=2; each truncated to 2 actions.
    fill_ms = 200 ms.
    """
    buf = _buf()
    buf.push_chunk(_chunk(5, dx_start=0.0))
    buf.push_chunk(_chunk(5, dx_start=10.0))
    buf.trim_to_ms(200, RATE)
    assert buf.fill_ms(RATE) == 200


def test_trim_returns_removed_count():
    """Push chunk(10); trim_to_ms(200, 10) → 10-2=8 removed."""
    buf = _buf()
    buf.push_chunk(_chunk(10))
    removed = buf.trim_to_ms(200, RATE)
    assert removed == 8


# ---------------------------------------------------------------------------
# Phase-switch flow
# ---------------------------------------------------------------------------

def test_phase_switch_discards_tail_preserves_near_term():
    """
    Push chunk(10, dx=i); pop 2 times to advance tick to 2.
    trim_to_ms(100, 10) → target_ticks=1, cutoff_tick=3 → keep 3 actions.
    Next pop (tick=2) returns actions[2].dx = 2.0.
    """
    buf = _buf(temp=0.0)
    buf.push_chunk(_chunk(10, dx_start=0.0))

    r0 = buf.pop_action()  # tick=0 → dx=0.0; tick→1
    r1 = buf.pop_action()  # tick=1 → dx=1.0; tick→2
    assert r0.dx == 0.0
    assert r1.dx == 1.0

    buf.trim_to_ms(100, RATE)  # target=1 tick, cutoff=3; chunk[:3] kept
    assert buf.size == 1       # ticks 2 is the only remaining one (3-2=1)

    r2 = buf.pop_action()
    assert r2 is not None
    assert abs(r2.dx - 2.0) < 1e-9
