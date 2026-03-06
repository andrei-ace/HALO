"""Unit tests for MessageHistory and CompactionResult."""

from __future__ import annotations

import pytest

from halo.cognitive.compactor import CompactionResult, MessageHistory, MessageRecord


def test_message_history_append_and_count():
    h = MessageHistory()
    assert h.count() == 0

    msg_id = h.append("user", "hello")
    assert isinstance(msg_id, str)
    assert len(msg_id) == 32  # uuid4 hex
    assert h.count() == 1

    h.append("model", "hi there")
    assert h.count() == 2


def test_message_history_get_all():
    h = MessageHistory()
    h.append("user", "a")
    h.append("model", "b")

    records = h.get_all()
    assert len(records) == 2
    assert records[0].role == "user"
    assert records[0].text == "a"
    assert records[1].role == "model"
    assert records[1].text == "b"
    assert all(isinstance(r, MessageRecord) for r in records)


def test_message_history_get_after():
    h = MessageHistory()
    id1 = h.append("user", "first")
    id2 = h.append("model", "second")
    h.append("user", "third")

    after_first = h.get_after(id1)
    assert len(after_first) == 2
    assert after_first[0].text == "second"
    assert after_first[1].text == "third"

    after_second = h.get_after(id2)
    assert len(after_second) == 1
    assert after_second[0].text == "third"


def test_message_history_get_after_unknown_id():
    h = MessageHistory()
    h.append("user", "a")
    h.append("model", "b")

    # Unknown ID returns all records
    after = h.get_after("nonexistent-id")
    assert len(after) == 2


def test_message_history_apply_compaction():
    h = MessageHistory()
    h.append("user", "msg1")
    h.append("model", "resp1")
    id3 = h.append("user", "msg2")
    h.append("model", "resp2")
    h.append("user", "msg3")

    result = h.apply_compaction(id3, "Summary of first 3 messages")
    assert isinstance(result, CompactionResult)
    assert result.compacted_count == 3  # id1, id2, id3
    assert result.retained_count == 2  # resp2, msg3
    assert result.summary == "Summary of first 3 messages"
    assert result.up_to_msg_id == id3

    # History should now have: summary + 2 retained
    assert h.count() == 3
    all_records = h.get_all()
    assert all_records[0].is_summary is True
    assert all_records[0].text == "Summary of first 3 messages"
    assert all_records[1].text == "resp2"
    assert all_records[2].text == "msg3"


def test_message_history_apply_compaction_all():
    """Compacting up to the last message leaves only the summary."""
    h = MessageHistory()
    h.append("user", "a")
    id2 = h.append("model", "b")

    result = h.apply_compaction(id2, "summary")
    assert result.compacted_count == 2
    assert result.retained_count == 0
    assert h.count() == 1
    assert h.get_all()[0].is_summary is True


def test_message_history_apply_compaction_unknown_id():
    h = MessageHistory()
    h.append("user", "a")

    with pytest.raises(ValueError, match="not found"):
        h.apply_compaction("bad-id", "summary")


def test_message_history_clear():
    h = MessageHistory()
    h.append("user", "a")
    h.append("model", "b")
    assert h.count() == 2

    h.clear()
    assert h.count() == 0
    assert h.get_all() == []


def test_compaction_result_fields():
    result = CompactionResult(
        summary="test summary",
        up_to_msg_id="abc123",
        compacted_count=5,
        retained_count=2,
        ts_ms=1000,
    )
    assert result.summary == "test summary"
    assert result.up_to_msg_id == "abc123"
    assert result.compacted_count == 5
    assert result.retained_count == 2
    assert result.ts_ms == 1000


def test_message_record_defaults():
    r = MessageRecord(msg_id="x", role="user", text="hi", ts_ms=100)
    assert r.is_summary is False

    r2 = MessageRecord(msg_id="y", role="model", text="summary", ts_ms=200, is_summary=True)
    assert r2.is_summary is True
