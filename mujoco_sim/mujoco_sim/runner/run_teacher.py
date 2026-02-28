"""Episode generation loop: reset env → stabilize → run teacher → write HDF5."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mujoco_sim.config import EnvConfig
from mujoco_sim.constants import PHASE_DONE
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
    max_steps: int = 800,
    stabilize_seconds: float = DEFAULT_STABILIZE_SECONDS,
    save_video: bool = False,
    vlm_base_url: str = "http://localhost:11434",
    vlm_model: str = "qwen2.5vl:3b",
    progress: bool = True,
) -> list[EpisodeResult]:
    """Generate episodes by running the scripted teacher in the environment.

    Each episode begins with a stabilization phase (zero-action steps for
    ``stabilize_seconds``) to let physics settle before recording starts.

    Args:
        num_episodes: Number of episodes to generate.
        output_dir: Directory to write HDF5 files.
        seed_base: Base seed; episode *i* uses ``seed_base + i``.
        env_config: Environment configuration (default: ``EnvConfig()``).
        teacher_config: Teacher gains/thresholds (default: ``TeacherConfig()``).
        max_steps: Maximum steps per episode (safety limit).
        stabilize_seconds: Seconds of zero-action settling before recording.
        save_video: If True, save an mp4 preview alongside each HDF5 file.
        vlm_base_url: Ollama base URL for VLM.
        vlm_model: VLM model name.
        progress: If True, show a tqdm progress bar.

    Returns:
        List of EpisodeResult for each episode (including failures).
    """
    # Lazy import to avoid requiring robosuite at module level
    from mujoco_sim.env import RobosuiteEnv

    env_config = env_config or EnvConfig()
    teacher_config = teacher_config or TeacherConfig()
    output_dir = Path(output_dir)

    env = RobosuiteEnv(env_config)
    teacher = PickTeacher(teacher_config)
    results: list[EpisodeResult] = []

    # VLM detection + OpenCV tracker (always on)
    print(f"[track] VLM tracking — model={vlm_model} url={vlm_base_url}")
    tracker_ctx = _TrackerContext(vlm_base_url, vlm_model)

    # tqdm progress bar (graceful fallback if not installed)
    pbar = None
    if progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")
        except ImportError:
            pass

    try:
        for ep_idx in range(num_episodes):
            seed = seed_base + ep_idx
            try:
                result = _run_single_episode(
                    env=env,
                    teacher=teacher,
                    ep_idx=ep_idx,
                    seed=seed,
                    output_dir=output_dir,
                    env_config=env_config,
                    max_steps=max_steps,
                    stabilize_steps=int(stabilize_seconds * env_config.control_freq),
                    save_video=save_video,
                    tracker_ctx=tracker_ctx,
                )
                logger.info(
                    "Episode %d/%d — seed=%d, steps=%d, %s → %s",
                    ep_idx + 1,
                    num_episodes,
                    seed,
                    result.num_steps,
                    "OK" if result.success else "INCOMPLETE",
                    result.path,
                )
            except Exception:
                logger.exception("Episode %d (seed=%d) FAILED", ep_idx, seed)
                result = EpisodeResult(episode_idx=ep_idx, seed=seed, error=_format_exc())

            results.append(result)

            if pbar is not None:
                status = "ok" if result.success else ("FAIL" if result.error else "inc")
                pbar.set_postfix(seed=seed, steps=result.num_steps, status=status)
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        env.close()

    return results


# ---------------------------------------------------------------------------
# Tracker context — wraps VLM + OpenCV tracker from halo project
# ---------------------------------------------------------------------------


class _TrackerContext:
    """Holds VLM fn + tracker factory, lazily initialized."""

    def __init__(self, vlm_base_url: str, vlm_model: str) -> None:
        self._vlm_base_url = vlm_base_url
        self._vlm_model = vlm_model
        self._vlm_fn = None
        self._tracker_factory = None

    def _ensure_init(self):
        if self._vlm_fn is None:
            from halo.services.target_perception_service.ollama_vlm_fn import make_ollama_vlm_fn
            from halo.services.target_perception_service.tracker_fn import make_tracker_factory_fn

            self._vlm_fn = make_ollama_vlm_fn(
                base_url=self._vlm_base_url,
                model=self._vlm_model,
            )
            self._tracker_factory = make_tracker_factory_fn()

    def detect_and_init_tracker(self, rgb_scene: np.ndarray) -> tuple[object | None, tuple[int, int, int, int] | None]:
        """Run VLM on the scene frame, init tracker with first detection.

        Args:
            rgb_scene: (H, W, 3) uint8 RGB image (raw from robosuite, bottom-up).

        Returns:
            (update_fn, init_bbox_xywh) or (None, None) if detection failed.
        """
        import cv2
        from halo.services.target_perception_service.frame_buffer import CapturedFrame

        self._ensure_init()

        # Flip to get correct orientation (robosuite renders bottom-up)
        rgb_flipped = rgb_scene[::-1]
        bgr = cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR)

        frame = CapturedFrame(image=bgr, ts_ms=int(time.monotonic() * 1000), arm_id="arm0")

        # Call VLM (async → sync bridge)
        print(f"[track] Calling VLM ({self._vlm_model}) for object detection...")
        vlm_scene = asyncio.run(self._vlm_fn("arm0", bgr, known_handles=[], target_handle=None))
        if not vlm_scene.detections:
            print("[track] VLM returned NO detections — episode will be skipped")
            return None, None

        detection = vlm_scene.detections[0]
        print(f"[track] VLM detected: {detection.handle} bbox={detection.bbox}")

        # Init tracker
        init_hint, update_fn = asyncio.run(self._tracker_factory(frame, detection))
        print(f"[track] Tracker initialized — bbox_xywh={init_hint.bbox_xywh}")
        return update_fn, init_hint.bbox_xywh

    def update_tracker(self, update_fn: object, rgb_scene: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        """Feed a frame to the tracker, return (ok, bbox_xywh)."""
        import cv2
        from halo.services.target_perception_service.frame_buffer import CapturedFrame

        rgb_flipped = rgb_scene[::-1]
        bgr = cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR)

        frame = CapturedFrame(image=bgr, ts_ms=int(time.monotonic() * 1000), arm_id="arm0")
        hint = asyncio.run(update_fn(frame))
        if hint is None:
            return False, None
        return True, hint.bbox_xywh


# ---------------------------------------------------------------------------
# Single episode
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
    tracker_ctx: _TrackerContext | None = None,  # None only in tests
) -> EpisodeResult:
    """Run one episode and write to HDF5."""
    obs = env.reset(seed=seed)

    # --- Stabilization phase: step with zero actions to let physics settle ---
    zero_action = np.zeros(7)
    for _ in range(stabilize_steps):
        obs, _, _, _ = env.step(zero_action)
    logger.debug("Stabilized for %d steps (seed=%d)", stabilize_steps, seed)

    # --- Initialize tracker if requested (must succeed before teacher starts) ---
    update_fn = None
    current_bbox: tuple[int, int, int, int] | None = None
    tracker_active = False

    if tracker_ctx is not None:
        update_fn, current_bbox = tracker_ctx.detect_and_init_tracker(obs["rgb_scene"])
        tracker_active = update_fn is not None
        if not tracker_active:
            return EpisodeResult(
                episode_idx=ep_idx,
                seed=seed,
                error="VLM detection failed — no objects found, skipping episode",
            )
        logger.info("Tracker initialized — bbox=%s (seed=%d)", current_bbox, seed)

    # --- Recording phase (starts only after tracking is confirmed) ---
    teacher.reset()

    meta = EpisodeMetadata(
        seed=seed,
        env_name=env_config.env_name,
        robot=env_config.robot,
        control_freq=env_config.control_freq,
        extra={"teacher": "pick", "stabilize_steps": str(stabilize_steps)},
    )
    episode = RawEpisode(metadata=meta)

    video_frames: list[np.ndarray] = []
    tracker_ok = True if tracker_active else None

    for step_idx in range(max_steps):
        action, phase_id, done = teacher.step(obs)

        # Update tracker
        if tracker_active and update_fn is not None:
            tracker_ok, new_bbox = tracker_ctx.update_tracker(update_fn, obs["rgb_scene"])
            if tracker_ok:
                current_bbox = new_bbox

        # Capture frame for video preview (flip vertically — robosuite renders bottom-up)
        if save_video:
            frame = obs["rgb_scene"][::-1].copy()
            if tracker_active:
                frame = _draw_tracking_overlay(frame, current_bbox, tracker_ok)
            video_frames.append(frame)

        # Record BEFORE stepping (obs is current, action is what we're about to apply)
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
            bbox_xywh=current_bbox if tracker_active else None,
            tracker_ok=tracker_ok if tracker_active else None,
        )
        episode.append(ts)

        if done:
            break

        obs, _reward, env_done, _info = env.step(action)
        if env_done:
            break

    final_phase = teacher.phase
    success = final_phase == PHASE_DONE

    path = episode_path(output_dir, ep_idx)
    write_episode(episode, path)

    # Write video preview if requested
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
# Video helpers
# ---------------------------------------------------------------------------


def _draw_tracking_overlay(
    frame: np.ndarray,
    bbox_xywh: tuple[int, int, int, int] | None,
    tracker_ok: bool,
) -> np.ndarray:
    """Draw bbox rectangle and tracking status on an RGB frame."""
    import cv2

    frame = frame.copy()
    h, w = frame.shape[:2]

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


def _format_exc() -> str:
    """Format the current exception as a one-line string."""
    import traceback

    return traceback.format_exc().strip().splitlines()[-1]
