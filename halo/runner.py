"""CLI entry point for headless HALO runner.

Usage:
    uv run python -m halo.runner --mock          # all mocked, instant latency
    uv run python -m halo.runner --mock --fast    # mocked with fast-integration latency
    uv run python -m halo.runner --model gpt-oss:20b --base-url http://localhost:11434  # real LLM
"""

from __future__ import annotations

import argparse
import asyncio
import logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HALO headless runner")
    p.add_argument("--arm", default="arm0", help="Arm ID (default: arm0)")
    p.add_argument("--mock", action="store_true", help="Use mock callables (no Ollama)")
    p.add_argument("--fast", action="store_true", help="Use fast-integration latency profile (with --mock)")
    p.add_argument("--duration", type=float, default=30.0, help="Max run duration in seconds")
    p.add_argument("--model", default="gpt-oss:20b", help="Planner model name")
    p.add_argument("--vlm-model", default="qwen2.5vl:3b", help="VLM model name")
    p.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


async def _run_mock(args: argparse.Namespace) -> None:
    from halo.contracts.actions import ZERO_JOINT_ACTION
    from halo.testing.mock_fns import (
        LatencyProfile,
        make_mock_apply_fn,
        make_mock_capture_fn_with_latency,
        make_mock_chunk_fn,
        make_mock_decide_fn,
        make_mock_tracker_factory_fn_with_latency,
        make_mock_vlm_fn,
    )
    from halo.testing.runner import HeadlessRunner, RunnerConfig

    latency = LatencyProfile.fast_integration() if args.fast else LatencyProfile.instant()

    config = RunnerConfig(
        arm_id=args.arm,
        max_duration_s=args.duration,
    )

    applied: list = []
    runner = HeadlessRunner(
        config=config,
        decide_fn=make_mock_decide_fn(latency),
        vlm_fn=make_mock_vlm_fn(latency),
        capture_fn=make_mock_capture_fn_with_latency(latency),
        tracker_factory_fn=make_mock_tracker_factory_fn_with_latency(latency),
        chunk_fn=make_mock_chunk_fn(latency),
        apply_fn=make_mock_apply_fn(latency, applied),
        initial_joint_state=ZERO_JOINT_ACTION,  # mock arm starts at zeros
    )

    print(f"HALO headless runner (mock, latency={'fast' if args.fast else 'instant'}, arm={args.arm})")
    print(f"Running for up to {args.duration}s. Press Ctrl+C to stop.")

    try:
        await runner.run()
    except KeyboardInterrupt:
        pass

    print(f"\nDone. Actions applied: {len(applied)}, events recorded: {len(runner.recorder.all_events)}")


async def _run_live(args: argparse.Namespace) -> None:
    from pathlib import Path

    from halo.contracts.actions import ZERO_JOINT_ACTION
    from halo.services.planner_service.agent import PlannerAgent
    from halo.services.target_perception_service.ollama_vlm_fn import make_ollama_vlm_fn
    from halo.services.target_perception_service.tracker_fn import make_tracker_factory_fn
    from halo.services.target_perception_service.video_source import VideoSource
    from halo.testing.mock_fns import make_mock_apply_fn, make_mock_chunk_fn
    from halo.testing.runner import HeadlessRunner, RunnerConfig

    prompts_dir = Path(__file__).parents[1] / "configs" / "planner"
    agent = PlannerAgent(args.model, args.base_url, prompts_dir)

    vlm_fn = make_ollama_vlm_fn(base_url=args.base_url, model=args.vlm_model)

    video_source = VideoSource()
    video_source.start()
    capture_fn = video_source.make_capture_fn(args.arm)
    tracker_factory_fn = make_tracker_factory_fn()

    config = RunnerConfig(
        arm_id=args.arm,
        max_duration_s=args.duration,
    )

    applied: list = []
    # TODO: read initial joint state from real robot/sim when available.
    # With real hardware, use the measured joint pose so that velocity
    # limiting and hold-position behaviour start from the true arm state.
    initial_state = ZERO_JOINT_ACTION
    runner = HeadlessRunner(
        config=config,
        decide_fn=agent.decide,
        vlm_fn=vlm_fn,
        capture_fn=capture_fn,
        tracker_factory_fn=tracker_factory_fn,
        chunk_fn=make_mock_chunk_fn(),  # ACT not available yet
        apply_fn=make_mock_apply_fn(log=applied),  # No real robot yet
        initial_joint_state=initial_state,
    )

    print(f"HALO headless runner (live, model={args.model}, arm={args.arm})")
    print(f"Running for up to {args.duration}s. Press Ctrl+C to stop.")

    try:
        await runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        video_source.stop()

    print(f"\nDone. Actions applied: {len(applied)}, events recorded: {len(runner.recorder.all_events)}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.mock:
        asyncio.run(_run_mock(args))
    else:
        asyncio.run(_run_live(args))


if __name__ == "__main__":
    main()
