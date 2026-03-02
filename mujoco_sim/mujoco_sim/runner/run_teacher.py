"""Episode generation loop: reset env → stabilize → run teacher → write HDF5.

Architecture: MuJoCo env + teacher run in a **SimServer** process (macOS
requires OpenGL on the main thread). ZMQ bridges the client/server:

    TelemetryStream (PUB/SUB): server publishes telemetry
    CommandRPC (REQ/REP): client sends step/reset/start_pick/configure commands

Standalone mode (``make sim-server`` already running) or managed mode
(server spawned automatically) are both supported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mujoco_sim.config import EnvConfig
from mujoco_sim.constants import PHASE_DONE, PHASE_LIFT
from mujoco_sim.dataset import EpisodeMetadata, RawEpisode, Timestep, episode_path, write_episode
from mujoco_sim.teacher.pick_teacher import PickTeacher, TeacherConfig

logger = logging.getLogger(__name__)

# Default stabilization: 5 seconds of zero-action steps to let physics settle
DEFAULT_STABILIZE_SECONDS = 5.0


@dataclass
class EpisodeResult:
    """Result of a single episode generation attempt."""

    episode_idx: int
    seed: int
    path: Path | None = None
    success: bool = False
    num_steps: int = 0
    final_phase: int = 0
    error: str | None = None


def run_teacher(
    num_episodes: int,
    output_dir: str | Path,
    *,
    seed_base: int = 0,
    env_config: EnvConfig | None = None,
    teacher_config: TeacherConfig | None = None,
    max_steps: int = 1200,
    stabilize_seconds: float = DEFAULT_STABILIZE_SECONDS,
    save_video: bool = False,
    vlm_base_url: str = "http://localhost:11434",
    vlm_model: str = "qwen2.5vl:3b",
    progress: bool = True,
    managed: bool = True,
    command_url: str | None = None,
    telemetry_url: str | None = None,
) -> list[EpisodeResult]:
    """Generate episodes by running the scripted teacher via ZMQ SimServer.

    Each episode begins with a stabilization phase (home-pose steps for
    ``stabilize_seconds``) to let physics settle before recording starts.

    The teacher runs server-side (needs MuJoCo model/data for IK).

    Args:
        num_episodes: Number of episodes to generate.
        output_dir: Directory to write HDF5 files.
        seed_base: Base seed; episode *i* uses ``seed_base + i``.
        env_config: Environment configuration (default: ``EnvConfig()``).
        teacher_config: Teacher gains/thresholds (default: ``TeacherConfig()``).
        max_steps: Maximum steps per episode (safety limit).
        stabilize_seconds: Seconds of home-pose settling before recording.
        save_video: If True, save an mp4 preview alongside each HDF5 file.
        vlm_base_url: Ollama base URL for VLM.
        vlm_model: VLM model name.
        progress: If True, show a tqdm progress bar.
        managed: If True, spawn the sim server automatically.
        command_url: Override CommandRPC URL (standalone mode).
        telemetry_url: Override TelemetryStream URL (standalone mode).

    Returns:
        List of EpisodeResult for each episode (including failures).
    """
    from halo.bridge.config import SimBridgeConfig
    from halo.bridge.sim_client import SimClient

    env_config = env_config or EnvConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stabilize_steps = int(stabilize_seconds * env_config.control_freq)

    print(f"[teacher] dataset generation — model args ignored in 2-channel mode ({vlm_model} @ {vlm_base_url})")

    # Build bridge config
    bridge_config = SimBridgeConfig(managed=managed)
    if command_url:
        bridge_config.command_url = command_url
    if telemetry_url:
        bridge_config.telemetry_url = telemetry_url

    # Connect to sim server
    client = SimClient(bridge_config)
    client.start(timeout=60.0)

    pbar = None
    if progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")
        except ImportError:
            pass

    results: list[EpisodeResult] = []

    try:
        for ep_idx in range(num_episodes):
            seed = seed_base + ep_idx
            try:
                result = _run_episode_via_zmq(
                    client=client,
                    ep_idx=ep_idx,
                    seed=seed,
                    output_dir=output_dir,
                    env_config=env_config,
                    max_steps=max_steps,
                    stabilize_steps=stabilize_steps,
                    save_video=save_video,
                )
            except Exception:
                import traceback

                err = traceback.format_exc().strip().splitlines()[-1]
                result = EpisodeResult(episode_idx=ep_idx, seed=seed, error=err)

            results.append(result)
            if pbar is not None:
                status = "ok" if result.success else ("FAIL" if result.error else "inc")
                pbar.set_postfix(seed=result.seed, steps=result.num_steps, status=status)
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        try:
            client.shutdown()
        except Exception:
            client.stop()

    return results


# ---------------------------------------------------------------------------
# Episode runner via ZMQ
# ---------------------------------------------------------------------------


def _run_episode_via_zmq(
    *,
    client,
    ep_idx: int,
    seed: int,
    output_dir: Path,
    env_config: EnvConfig,
    max_steps: int,
    stabilize_steps: int,
    save_video: bool,
) -> EpisodeResult:
    """Run one episode via SimClient CommandRPC commands.

    Uses start_pick to trigger autonomous trajectory execution on the server,
    then polls telemetry for observations until done.
    """
    import time

    # Reset
    client.reset(seed=seed)

    # Stabilize — step with home pose (zeros for now; server starts at home)
    home_action = np.zeros(6)
    for _ in range(stabilize_steps):
        client.step(home_action)

    # Trigger trajectory planning and autonomous execution
    resp = client.start_pick()
    if resp.get("type") == "start_pick_error":
        return EpisodeResult(
            episode_idx=ep_idx,
            seed=seed,
            error=f"start_pick failed: {resp.get('message', 'unknown')}",
        )

    # Recording phase — poll telemetry for observations
    meta = EpisodeMetadata(
        seed=seed,
        env_name=env_config.scene_xml,
        robot=env_config.robot,
        control_freq=env_config.control_freq,
        extra={"teacher": "pick", "stabilize_steps": str(stabilize_steps)},
    )
    episode = RawEpisode(metadata=meta)
    video_frames: list[np.ndarray] = []

    poll_interval = 1.0 / env_config.control_freq
    last_step_count = -1
    phase_id = 0

    for step_idx in range(max_steps):
        # Wait for new telemetry
        time.sleep(poll_interval)

        telemetry = client.latest_telemetry
        if telemetry is None:
            continue

        # Skip duplicate telemetry frames
        if telemetry.get("step_count", -1) == last_step_count:
            continue
        last_step_count = telemetry.get("step_count", -1)

        phase_id = telemetry.get("phase_id", 0)
        done = telemetry.get("done", False)

        qpos = telemetry["qpos"]
        qvel = telemetry["qvel"]
        ee_pose = telemetry["ee_pose"]
        object_pose = telemetry["object_pose"]
        joint_pos = telemetry["joint_pos"]
        gripper = telemetry["gripper"]
        action = telemetry["action"]
        rgb_scene = telemetry["rgb_scene"]
        rgb_wrist = telemetry["rgb_wrist"]

        # Video preview
        if save_video:
            frame = rgb_scene.copy()
            frame = _composite_wrist(frame, rgb_wrist, phase_id)
            video_frames.append(frame)

        # Record timestep
        ts = Timestep(
            rgb_scene=rgb_scene,
            rgb_wrist=rgb_wrist,
            qpos=qpos,
            qvel=qvel,
            gripper=float(gripper),
            ee_pose=ee_pose,
            action=np.array(action, copy=True),
            phase_id=phase_id,
            object_pose=object_pose,
            red_object_pose=telemetry.get("red_object_pose"),
            joint_pos=joint_pos,
        )
        episode.append(ts)

        if done:
            break

    final_phase = phase_id
    reached_done = final_phase == PHASE_DONE
    success = reached_done and _verify_lift(episode)

    path = episode_path(output_dir, ep_idx)
    write_episode(episode, path)

    if save_video and video_frames:
        video_path = path.with_suffix(".mp4")
        _write_preview_video(video_frames, video_path, fps=env_config.control_freq)

    return EpisodeResult(
        episode_idx=ep_idx,
        seed=seed,
        path=path,
        success=success,
        num_steps=len(episode),
        final_phase=final_phase,
    )


# ---------------------------------------------------------------------------
# Sync single-episode runner (no ZMQ, used by tests)
# ---------------------------------------------------------------------------


def _run_single_episode(
    *,
    env,
    teacher: PickTeacher,
    ep_idx: int,
    seed: int,
    output_dir: Path,
    env_config: EnvConfig,
    max_steps: int,
    stabilize_steps: int,
    save_video: bool = False,
    tracker_ctx: None = None,
) -> EpisodeResult:
    """Run one episode and write to HDF5.  Tests only (tracker_ctx always None)."""
    obs = env.reset(seed=seed)

    # --- Stabilization phase: step with home pose to let physics settle ---
    home_action = env.home_qpos
    for _ in range(stabilize_steps):
        obs, _, _, _ = env.step(home_action)
    logger.debug("Stabilized for %d steps (seed=%d)", stabilize_steps, seed)

    teacher.reset()

    meta = EpisodeMetadata(
        seed=seed,
        env_name=env_config.scene_xml,
        robot=env_config.robot,
        control_freq=env_config.control_freq,
        extra={"teacher": "pick", "stabilize_steps": str(stabilize_steps)},
    )
    episode = RawEpisode(metadata=meta)

    video_frames: list[np.ndarray] = []

    for step_idx in range(max_steps):
        action, phase_id, done = teacher.step(obs, env.mujoco_model, env.mujoco_data)

        if save_video:
            frame = obs["rgb_scene"].copy()
            frame = _composite_wrist(frame, obs["rgb_wrist"], phase_id)
            video_frames.append(frame)

        ts = Timestep(
            rgb_scene=obs["rgb_scene"],
            rgb_wrist=obs["rgb_wrist"],
            qpos=obs["qpos"],
            qvel=obs["qvel"],
            gripper=float(obs["gripper"]),
            ee_pose=obs["ee_pose"],
            action=np.array(action, copy=True),
            phase_id=phase_id,
            object_pose=obs.get("object_pose"),
            red_object_pose=obs.get("red_object_pose"),
            joint_pos=obs.get("joint_pos"),
        )
        episode.append(ts)

        if done:
            break

        obs, _reward, env_done, _info = env.step(action)
        if env_done:
            break

    final_phase = teacher.phase
    reached_done = final_phase == PHASE_DONE
    success = reached_done and _verify_lift(episode)

    path = episode_path(output_dir, ep_idx)
    write_episode(episode, path)

    if save_video and video_frames:
        video_path = path.with_suffix(".mp4")
        _write_preview_video(video_frames, video_path, fps=env_config.control_freq)

    return EpisodeResult(
        episode_idx=ep_idx,
        seed=seed,
        path=path,
        success=success,
        num_steps=len(episode),
        final_phase=final_phase,
    )


# ---------------------------------------------------------------------------
# Lift verification
# ---------------------------------------------------------------------------

# Minimum cube Z rise during LIFT phase to count as a successful grasp (metres).
_LIFT_THRESHOLD_M = 0.005


def _verify_lift(episode: RawEpisode) -> bool:
    """Check that the cube actually rose during the LIFT phase.

    Uses the *maximum* cube Z reached during lift (not the final Z), because
    the cube may separate from the gripper during the deceleration phase and
    fall back to the table while still counting as a successful pick.
    """
    phase_ids = episode.phase_ids
    obj_poses = episode.object_poses
    if phase_ids is None or obj_poses is None:
        return False
    lift_mask = phase_ids == PHASE_LIFT
    if not lift_mask.any():
        return False
    lift_z = obj_poses[lift_mask, 2]
    delta_z = float(lift_z.max() - lift_z[0])
    if delta_z < _LIFT_THRESHOLD_M:
        logger.warning("Lift check FAILED: cube Δz=%.4f m (threshold=%.4f m)", delta_z, _LIFT_THRESHOLD_M)
        return False
    return True


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _composite_wrist(
    frame: np.ndarray,
    rgb_wrist: np.ndarray,
    phase_id: int,
    scale: float = 0.4,
    margin: int = 8,
) -> np.ndarray:
    """Overlay wrist camera as picture-in-picture in the bottom-right corner.

    Shows black when phase is outside WRIST_ACTIVE_PHASES, live feed when active.
    """
    from mujoco_sim.constants import WRIST_ACTIVE_PHASES

    wrist = rgb_wrist
    wh, ww = wrist.shape[:2]
    th, tw = int(wh * scale), int(ww * scale)

    if phase_id in WRIST_ACTIVE_PHASES:
        import cv2

        thumb = cv2.resize(wrist, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        thumb = np.zeros((th, tw, 3), dtype=np.uint8)

    fh, fw = frame.shape[:2]
    y0 = fh - th - margin
    x0 = fw - tw - margin
    frame[y0 : y0 + th, x0 : x0 + tw] = thumb
    return frame


def _draw_tracking_overlay(
    frame: np.ndarray,
    bbox_xywh: tuple[int, int, int, int] | None,
    tracker_ok: bool,
) -> np.ndarray:
    """Draw bbox rectangle and tracking status on an RGB frame."""
    import cv2

    frame = frame.copy()

    if bbox_xywh is not None and tracker_ok:
        x, y, bw, bh = bbox_xywh
        cx, cy = x + bw // 2, y + bh // 2
        color = (0, 255, 0)  # green = tracking
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = f"TRACK: ok  center=({cx},{cy})"
    else:
        color = (255, 0, 0)  # red = lost
        label = "TRACK: LOST"

    cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def _write_preview_video(frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    """Write a list of RGB frames to an mp4 file using OpenCV."""
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
