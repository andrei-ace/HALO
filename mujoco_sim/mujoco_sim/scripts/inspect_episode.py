"""Quick debug inspector for the latest episode HDF5 file.

Usage::

    python -m mujoco_sim.scripts.inspect_episode                    # latest run, ep 0
    python -m mujoco_sim.scripts.inspect_episode data/episodes/20260228_200233/ep_000000.hdf5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_latest_episode(base_dir: str = "data/episodes") -> Path | None:
    """Find the latest episode HDF5 file across all timestamped run dirs."""
    base = Path(base_dir)
    if not base.exists():
        return None
    # Find all run dirs, sorted descending (latest first)
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in run_dirs:
        hdf5s = sorted(d.glob("*.hdf5"))
        if hdf5s:
            return hdf5s[0]
    # Fallback: HDF5 directly in base dir
    hdf5s = sorted(base.glob("*.hdf5"))
    return hdf5s[0] if hdf5s else None


def inspect(path: Path) -> None:
    from mujoco_sim.constants import PHASE_DONE
    from mujoco_sim.dataset import read_episode

    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    ep = read_episode(path)
    n = len(ep)

    print(f"Steps:        {n}")
    print(f"Seed:         {ep.metadata.seed}")
    print(f"Control freq: {ep.metadata.control_freq} Hz")
    print(f"Duration:     {n / ep.metadata.control_freq:.1f} s")
    print(f"Extra:        {ep.metadata.extra}")
    print()

    # Phase analysis
    phase_ids = ep.phase_ids
    obj_poses = ep.object_poses

    if phase_ids is not None:
        unique_phases = np.unique(phase_ids)
        final_phase = int(phase_ids[-1])
        reached_done = final_phase == PHASE_DONE
        print(f"Phases seen:  {list(unique_phases)}")
        print(f"Final phase:  {final_phase} ({'DONE' if reached_done else 'INCOMPLETE'})")

        # Phase durations
        print("\nPhase durations:")
        changes = np.where(np.diff(phase_ids) != 0)[0] + 1
        boundaries = [0, *changes, n]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            pid = int(phase_ids[start])
            dur = end - start
            secs = dur / ep.metadata.control_freq
            print(f"  phase {pid:2d}: steps {start:4d}-{end - 1:4d} ({dur:4d} steps, {secs:.1f}s)")

        # Lift verification: did the cube actually move up?
        if reached_done and obj_poses is not None:
            from mujoco_sim.constants import PHASE_LIFT

            lift_mask = phase_ids == PHASE_LIFT
            if lift_mask.any():
                lift_start_z = float(obj_poses[lift_mask][0, 2])
                lift_end_z = float(obj_poses[lift_mask][-1, 2])
                cube_lift_m = lift_end_z - lift_start_z
                lifted = cube_lift_m > 0.05  # at least 5cm
                tag = "OK" if lifted else "FAILED"
                print(f"\nLift check:   cube_z {lift_start_z:.4f} → {lift_end_z:.4f}  (Δ={cube_lift_m:.4f} m)  [{tag}]")
                if not lifted:
                    print("  ⚠ Cube did not move up — grasp may have failed")
        print()

    # EE + cube positions
    ee_poses = ep.ee_poses

    print("EE position (first/last):")
    print(f"  first: {ee_poses[0, :3].round(4)}")
    print(f"  last:  {ee_poses[-1, :3].round(4)}")

    if obj_poses is not None:  # noqa: SIM102
        print("Cube position (first/last):")
        print(f"  first: {obj_poses[0, :3].round(4)}")
        print(f"  last:  {obj_poses[-1, :3].round(4)}")

        # Distance over time
        dists = np.linalg.norm(ee_poses[:, :3] - obj_poses[:, :3], axis=1)
        print("\nEE-cube distance:")
        print(f"  start: {dists[0]:.4f} m")
        print(f"  min:   {dists.min():.4f} m  (at step {dists.argmin()})")
        print(f"  final: {dists[-1]:.4f} m")

        # Sample distances at phase transitions
        if phase_ids is not None:
            print("\nDistance at phase transitions:")
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                pid = int(phase_ids[start])
                d = dists[start]
                z = ee_poses[start, 2]
                print(f"  phase {pid:2d} start (step {start:4d}): dist={d:.4f} m, ee_z={z:.4f}")

    # Gripper
    grippers = ep.gripper_array
    print(f"\nGripper (first/last): {grippers[0]:.3f} → {grippers[-1]:.3f}")

    # Actions summary
    actions = ep.actions
    action_norms = np.linalg.norm(actions[:, :3], axis=1)
    print("\nAction norms (xyz):")
    print(f"  mean: {action_norms.mean():.4f}  max: {action_norms.max():.4f}")
    print(f"  last 50 mean: {action_norms[-50:].mean():.6f}")

    # Tracker data
    bbox_arr = ep.bbox_xywh_array
    tracker_arr = ep.tracker_ok_array
    if tracker_arr is not None:
        ok_count = tracker_arr.sum()
        print(f"\nTracker: {ok_count}/{n} frames OK ({ok_count / n * 100:.0f}%)")
        if bbox_arr is not None and ok_count > 0:
            last_ok = np.where(tracker_arr)[0][-1]
            print(f"  last OK bbox: {tuple(bbox_arr[last_ok])} (step {last_ok})")
    else:
        print("\nTracker: no tracking data")

    # Check for video
    mp4 = path.with_suffix(".mp4")
    if mp4.exists():
        print(f"\nVideo: {mp4} ({mp4.stat().st_size / 1024:.1f} KB)")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = _find_latest_episode()

    if path is None or not path.exists():
        print("No episode found. Pass a path or generate one first.")
        raise SystemExit(1)

    inspect(path)


if __name__ == "__main__":
    main()
