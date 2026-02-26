"""Tests for FrameRingBuffer: push, read-cursor, capacity, lifecycle."""

from halo.services.target_perception_service.frame_buffer import CapturedFrame, FrameRingBuffer


def _frame(n: int, arm_id: str = "arm0") -> CapturedFrame:
    return CapturedFrame(image=f"img_{n}", ts_ms=n * 100, arm_id=arm_id)


# ─── push / active ──────────────────────────────────────────────────────────


def test_push_while_active():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(1))
    buf.push(_frame(2))
    assert len(buf) == 2


def test_push_ignored_when_not_active():
    buf = FrameRingBuffer(max_size=10)
    buf.push(_frame(1))
    assert len(buf) == 0


def test_push_ignored_after_stop():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(1))
    buf.stop()
    buf.push(_frame(2))
    assert len(buf) == 1


# ─── read_from ───────────────────────────────────────────────────────────────


def test_read_from_returns_all_from_index():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    for i in range(5):
        buf.push(_frame(i))

    frames, cursor = buf.read_from(0)
    assert len(frames) == 5
    assert cursor == 5
    assert frames[0].image == "img_0"
    assert frames[4].image == "img_4"


def test_read_from_incremental():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(0))
    buf.push(_frame(1))

    batch1, idx = buf.read_from(0)
    assert len(batch1) == 2
    assert idx == 2

    buf.push(_frame(2))
    buf.push(_frame(3))

    batch2, idx = buf.read_from(idx)
    assert len(batch2) == 2
    assert idx == 4
    assert batch2[0].image == "img_2"


def test_read_from_empty_when_caught_up():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(0))

    _, idx = buf.read_from(0)
    frames, idx2 = buf.read_from(idx)
    assert frames == []
    assert idx2 == idx


# ─── max_size ────────────────────────────────────────────────────────────────


def test_max_size_caps_buffer():
    buf = FrameRingBuffer(max_size=3)
    buf.start()
    for i in range(10):
        buf.push(_frame(i))
    assert len(buf) == 3
    # Only first 3 frames kept (push is no-op once at cap)
    frames, _ = buf.read_from(0)
    assert frames[0].image == "img_0"
    assert frames[2].image == "img_2"


# ─── start / stop / clear ───────────────────────────────────────────────────


def test_start_clears_previous():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(0))
    buf.push(_frame(1))
    assert len(buf) == 2

    buf.start()  # restart
    assert len(buf) == 0
    assert buf.is_active


def test_stop_preserves_frames():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(0))
    buf.stop()
    assert not buf.is_active

    frames, _ = buf.read_from(0)
    assert len(frames) == 1


def test_clear_resets_everything():
    buf = FrameRingBuffer(max_size=10)
    buf.start()
    buf.push(_frame(0))
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_active


# ─── is_active property ─────────────────────────────────────────────────────


def test_is_active_lifecycle():
    buf = FrameRingBuffer(max_size=10)
    assert not buf.is_active
    buf.start()
    assert buf.is_active
    buf.stop()
    assert not buf.is_active
