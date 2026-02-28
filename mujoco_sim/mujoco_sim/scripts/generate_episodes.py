"""CLI entry point for teacher episode generation.

Usage::

    # Managed mode (spawns sim server automatically):
    python -m mujoco_sim.scripts.generate_episodes --num-episodes 10 --output-dir episodes

    # Standalone mode (connect to running sim server):
    python -m mujoco_sim.scripts.generate_episodes --standalone --num-episodes 10
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PICK demo episodes with scripted teacher")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to generate")
    parser.add_argument("--output-dir", type=str, default="data/episodes", help="Output directory for HDF5 files")
    parser.add_argument("--seed-base", type=int, default=0, help="Base seed (episode i uses seed_base + i)")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum steps per episode")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency (Hz)")
    parser.add_argument("--stabilize", type=float, default=5.0, help="Seconds of zero-action settling before recording")
    parser.add_argument("--save-video", action="store_true", help="Save mp4 preview alongside each HDF5 file")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:11434", help="Ollama base URL for VLM")
    parser.add_argument("--vlm-model", type=str, default="qwen2.5vl:3b", help="VLM model name")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument(
        "--standalone", action="store_true", help="Connect to an already-running sim server (don't spawn)"
    )
    parser.add_argument("--command-url", type=str, default=None, help="CommandRPC URL (standalone mode)")
    parser.add_argument("--telemetry-url", type=str, default=None, help="TelemetryStream URL (standalone mode)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Fail fast if --save-video requested but opencv not available
    if args.save_video:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("ERROR: --save-video requires opencv-python. Install with: uv sync --extra sim --extra viewer")
            raise SystemExit(1)

    from mujoco_sim.config import EnvConfig
    from mujoco_sim.runner import run_teacher

    # Timestamped subdirectory to avoid overwriting previous runs
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(args.output_dir) / run_stamp)
    print(f"[run] Output: {output_dir}/")

    env_config = EnvConfig(control_freq=args.control_freq)
    results = run_teacher(
        num_episodes=args.num_episodes,
        output_dir=output_dir,
        seed_base=args.seed_base,
        env_config=env_config,
        max_steps=args.max_steps,
        stabilize_seconds=args.stabilize,
        save_video=args.save_video,
        vlm_base_url=args.vlm_url,
        vlm_model=args.vlm_model,
        progress=not args.no_progress,
        managed=not args.standalone,
        command_url=args.command_url,
        telemetry_url=args.telemetry_url,
    )

    _print_summary(results, output_dir, args.save_video)


def _print_summary(results: list, output_dir: str, save_video: bool) -> None:
    """Print a summary table of episode generation results."""
    from mujoco_sim.runner.run_teacher import EpisodeResult

    results: list[EpisodeResult] = results  # type narrowing

    total = len(results)
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if r.error is not None]
    incomplete = [r for r in results if not r.success and r.error is None]

    steps = [r.num_steps for r in results if r.error is None]

    print()
    print("=" * 60)
    print("  Episode Generation Summary")
    print("=" * 60)
    print(f"  Total:      {total}")
    print(f"  Succeeded:  {len(succeeded)}")
    if incomplete:
        print(f"  Incomplete: {len(incomplete)}  (did not reach DONE phase)")
    if failed:
        print(f"  Failed:     {len(failed)}")
    if total > 0 and not failed:
        rate = len(succeeded) / total * 100
        print(f"  Success:    {rate:.0f}%")
    print("-" * 60)

    if steps:
        print(f"  Steps — avg: {sum(steps) / len(steps):.0f}  min: {min(steps)}  max: {max(steps)}")
        print(f"  Total frames: {sum(steps)}")

    # Disk usage
    out_path = Path(output_dir)
    hdf5_size = sum(f.stat().st_size for f in out_path.glob("*.hdf5") if f.is_file()) if out_path.exists() else 0
    mp4_size = sum(f.stat().st_size for f in out_path.glob("*.mp4") if f.is_file()) if out_path.exists() else 0
    if hdf5_size > 0:
        print(f"  Disk — HDF5: {_fmt_bytes(hdf5_size)}", end="")
        if save_video and mp4_size > 0:
            print(f"  mp4: {_fmt_bytes(mp4_size)}", end="")
        print(f"  total: {_fmt_bytes(hdf5_size + mp4_size)}")

    print("-" * 60)

    # Failed episode details
    if failed:
        print("  FAILED episodes:")
        for r in failed:
            print(f"    ep {r.episode_idx} (seed={r.seed}): {r.error}")
        print("-" * 60)

    # File list
    written = [r for r in results if r.path is not None]
    if written:
        print(f"  Output: {output_dir}/")
        for r in written:
            tag = "ok" if r.success else "inc"
            print(f"    {r.path.name}  [{tag}, {r.num_steps} steps, phase={r.final_phase}]")

    print("=" * 60)
    print()


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


if __name__ == "__main__":
    main()
