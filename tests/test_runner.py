"""Tests for halo.testing.HeadlessRunner."""

import asyncio

import pytest

from halo.contracts.actions import ZERO_JOINT_ACTION
from halo.contracts.events import EventType
from halo.testing.mock_fns import (
    make_mock_apply_fn,
    make_mock_capture_fn_with_latency,
    make_mock_chunk_fn,
    make_mock_decide_fn,
    make_mock_tracker_factory_fn_with_latency,
    make_mock_vlm_fn,
)
from halo.testing.runner import HeadlessRunner, RunnerConfig

ARM = "arm0"


def _make_runner(
    *,
    enable_planner=True,
    enable_perception=True,
    enable_skill_runner=True,
    enable_control=True,
    decide_fn=None,
    apply_fn=None,
    chunk_fn=None,
    push_fn=None,
    vlm_fn=None,
    capture_fn=None,
    tracker_factory_fn=None,
    max_duration_s=5.0,
    **kwargs,
) -> HeadlessRunner:
    config = RunnerConfig(
        arm_id=ARM,
        max_duration_s=max_duration_s,
        enable_planner=enable_planner,
        enable_perception=enable_perception,
        enable_skill_runner=enable_skill_runner,
        enable_control=enable_control,
    )
    return HeadlessRunner(
        config=config,
        decide_fn=decide_fn or (make_mock_decide_fn() if enable_planner else None),
        apply_fn=apply_fn or (make_mock_apply_fn() if enable_control else None),
        chunk_fn=chunk_fn or (make_mock_chunk_fn() if enable_skill_runner else None),
        push_fn=push_fn,
        vlm_fn=vlm_fn or (make_mock_vlm_fn() if enable_perception else None),
        capture_fn=capture_fn or (make_mock_capture_fn_with_latency() if enable_perception else None),
        tracker_factory_fn=tracker_factory_fn
        or (make_mock_tracker_factory_fn_with_latency() if enable_perception else None),
        initial_joint_state=ZERO_JOINT_ACTION if enable_control else None,
        **kwargs,
    )


# -- construction -----------------------------------------------------------


def test_creates_all_services():
    runner = _make_runner()
    assert runner.runtime is not None
    assert runner.planner_svc is not None
    assert runner.perception_svc is not None
    assert runner.skill_runner_svc is not None
    assert runner.control_svc is not None
    assert runner.recorder is not None


def test_partial_wiring_planner_only():
    runner = _make_runner(
        enable_perception=False,
        enable_skill_runner=False,
        enable_control=False,
    )
    assert runner.planner_svc is not None
    assert runner.perception_svc is None
    assert runner.skill_runner_svc is None
    assert runner.control_svc is None


def test_partial_wiring_control_only():
    runner = _make_runner(
        enable_planner=False,
        enable_perception=False,
        enable_skill_runner=False,
    )
    assert runner.control_svc is not None
    assert runner.planner_svc is None


def test_auto_wires_push_fn():
    """When both skill_runner and control are enabled, push_fn auto-wires."""
    runner = _make_runner()
    assert runner.skill_runner_svc is not None
    assert runner.control_svc is not None
    # push_fn should be control_svc.push_chunk (bound method equality)
    assert runner.skill_runner_svc._push_fn == runner.control_svc.push_chunk


def test_explicit_push_fn_overrides_auto_wire():
    """Explicit push_fn takes priority over auto-wiring."""
    calls = []

    async def custom_push(chunk):
        calls.append(chunk)

    runner = _make_runner(push_fn=custom_push)
    assert runner.skill_runner_svc._push_fn is custom_push


def test_missing_decide_fn_raises():
    config = RunnerConfig(enable_planner=True, enable_perception=False, enable_skill_runner=False, enable_control=False)
    with pytest.raises(ValueError, match="decide_fn"):
        HeadlessRunner(config=config, decide_fn=None)


def test_missing_apply_fn_raises():
    config = RunnerConfig(enable_planner=False, enable_perception=False, enable_skill_runner=False, enable_control=True)
    with pytest.raises(ValueError, match="apply_fn"):
        HeadlessRunner(config=config, apply_fn=None, initial_joint_state=ZERO_JOINT_ACTION)


def test_missing_initial_joint_state_raises():
    config = RunnerConfig(enable_planner=False, enable_perception=False, enable_skill_runner=False, enable_control=True)
    with pytest.raises(ValueError, match="initial_joint_state"):
        HeadlessRunner(config=config, apply_fn=make_mock_apply_fn(), initial_joint_state=None)


def test_missing_chunk_fn_raises():
    config = RunnerConfig(enable_planner=False, enable_perception=False, enable_skill_runner=True, enable_control=True)
    with pytest.raises(ValueError, match="chunk_fn"):
        HeadlessRunner(
            config=config, chunk_fn=None, apply_fn=make_mock_apply_fn(), initial_joint_state=ZERO_JOINT_ACTION
        )


def test_missing_push_fn_raises_when_control_disabled():
    with pytest.raises(ValueError, match="push_fn"):
        _make_runner(enable_control=False, push_fn=None, enable_skill_runner=True)


# -- lifecycle --------------------------------------------------------------


async def test_start_stop():
    runner = _make_runner()
    await runner.start()
    assert runner._running is True
    await runner.stop()
    assert runner._running is False


async def test_start_is_idempotent():
    runner = _make_runner()
    await runner.start()
    await runner.start()  # no error
    await runner.stop()


async def test_stop_is_idempotent():
    runner = _make_runner()
    await runner.start()
    await runner.stop()
    await runner.stop()  # no error


# -- run with predicate ----------------------------------------------------


async def test_run_stops_on_predicate():
    runner = _make_runner(max_duration_s=10.0)
    call_count = 0

    def done():
        nonlocal call_count
        call_count += 1
        return call_count >= 3

    await runner.run(until=done)
    assert not runner._running
    assert call_count >= 3


async def test_run_stops_on_timeout():
    runner = _make_runner(max_duration_s=0.15)
    await runner.run()
    assert not runner._running


# -- recorder captures events ----------------------------------------------


async def test_recorder_captures_planner_events():
    """Planner start() issues DESCRIBE_SCENE → COMMAND_ACCEPTED event."""
    runner = _make_runner(
        enable_perception=False,
        enable_skill_runner=False,
        enable_control=False,
    )
    await runner.start()
    # Give the planner time to issue its startup describe_scene command
    await asyncio.sleep(0.1)
    await runner.stop()

    # The planner start() issues a DESCRIBE_SCENE command which generates a COMMAND_ACCEPTED event
    accepted = runner.recorder.events_of_type(EventType.COMMAND_ACCEPTED)
    assert len(accepted) >= 1


# -- arm_id property -------------------------------------------------------


def test_arm_id_property():
    runner = _make_runner()
    assert runner.arm_id == ARM
