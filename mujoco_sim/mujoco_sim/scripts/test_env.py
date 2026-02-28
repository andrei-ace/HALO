"""Acceptance script: create env, reset with seed, dump scene.png + wrist.png, verify reproducibility."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import numpy as np
    from PIL import Image

    from mujoco_sim.config.env_config import EnvConfig
    from mujoco_sim.env.robosuite_env import RobosuiteEnv

    config = EnvConfig()
    env = RobosuiteEnv(config)

    print("Environment created successfully.")
    print(f"  env_name: {config.env_name}")
    print(f"  robot: {config.robot}")
    print(f"  controller: {config.controller}")
    print(f"  action_dim: {env.action_dim}")

    # First reset with seed
    obs1 = env.reset(seed=42)
    print("\nReset 1 (seed=42):")
    for key, val in obs1.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    # Save images
    output_dir = Path(".")
    scene_path = output_dir / "scene.png"
    wrist_path = output_dir / "wrist.png"

    Image.fromarray(obs1["rgb_scene"]).save(scene_path)
    Image.fromarray(obs1["rgb_wrist"]).save(wrist_path)
    print(f"\nSaved {scene_path} ({obs1['rgb_scene'].shape})")
    print(f"Saved {wrist_path} ({obs1['rgb_wrist'].shape})")

    # Second reset with same seed — verify reproducibility
    obs2 = env.reset(seed=42)
    print("\nReset 2 (seed=42) — reproducibility check:")
    for key in ["ee_pose", "object_pose", "qpos"]:
        match = np.allclose(obs1[key], obs2[key])
        status = "PASS" if match else "FAIL"
        print(f"  {key}: {status}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
