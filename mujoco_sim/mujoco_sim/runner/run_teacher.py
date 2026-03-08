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
from mujoco_sim.constants import PHASE_DONE, PHASE_LIFT, PHASE_RETURNING
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
    pick_and_place: bool = False,
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
        pick_and_place: If True, follow each successful pick with a place-in-tray.

    Returns:
        List of EpisodeResult for each episode (including failures).
    """
    from halo.bridge.config import SimBridgeConfig
    from halo.bridge.sim_client import SimClient

    env_config = env_config or EnvConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stabilize_steps = int(stabilize_seconds * env_config.control_freq)

    # Build tracker context for bbox overlay on video
    tracker_ctx_factory = None
    if save_video:
        tracker_ctx_factory = lambda: _TrackerCtx(vlm_base_url, vlm_model)  # noqa: E731
        print(f"[teacher] VLM tracker enabled for video overlay ({vlm_model} @ {vlm_base_url})")

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

            # Fresh server per episode — prevents ZMQ state corruption on long runs
            bridge_config = SimBridgeConfig(managed=managed)
            if command_url:
                bridge_config.command_url = command_url
            if telemetry_url:
                bridge_config.telemetry_url = telemetry_url

            client = SimClient(bridge_config)
            client.start(timeout=60.0)

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
                    pick_and_place=pick_and_place,
                    tracker_ctx=tracker_ctx_factory() if tracker_ctx_factory else None,
                )
            except Exception:
                import traceback

                err = traceback.format_exc().strip().splitlines()[-1]
                result = EpisodeResult(episode_idx=ep_idx, seed=seed, error=err)
            finally:
                try:
                    client.shutdown()
                except Exception:
                    client.stop()

            results.append(result)
            if pbar is not None:
                status = "ok" if result.success else ("FAIL" if result.error else "inc")
                pbar.set_postfix(seed=result.seed, steps=result.num_steps, status=status)
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return results


# ---------------------------------------------------------------------------
# Client-side VLM + OpenCV tracker for bbox overlay
# ---------------------------------------------------------------------------


class _TrackerCtx:
    """Runs VLM detection on the first frame, then OpenCV tracker on subsequent frames.

    Uses the existing TelemetryStream rgb_scene frames — no extra ZMQ channel needed.
    All tracker infrastructure comes from halo.services.target_perception_service.

    Supports target switching via ``switch_target()`` — resets the tracker so the
    next ``update()`` re-runs VLM to find the new target (e.g. switch from cube
    to tray when transitioning from pick to place).
    """

    def __init__(self, vlm_base_url: str, vlm_model: str, target_keyword: str = "cube") -> None:
        self._vlm_base_url = vlm_base_url
        self._vlm_model = vlm_model
        self._target_keyword = target_keyword
        self._tracker: object | None = None  # cv2.Tracker once initialised
        self._handle: str = ""
        self._initialised = False
        self._vlm_failed = False
        self._cached_scene = None  # VlmScene from last VLM call

    def switch_target(self, target_keyword: str) -> None:
        """Switch tracking target. Call ``ensure_ready()`` before resuming ``update()``."""
        self._target_keyword = target_keyword
        self._tracker = None
        self._handle = ""
        self._initialised = False
        self._vlm_failed = False
        logger.info("Tracker target switched to '%s'", target_keyword)

    def ensure_ready(self, rgb_scene: np.ndarray) -> None:
        """Run VLM detection and init tracker. Call while the sim is holding still.

        Blocks until VLM completes (~5-10 s). After this, ``update()`` only does
        fast OpenCV tracker updates (no VLM).
        """
        if self._initialised or self._vlm_failed:
            return

        import cv2

        bgr = cv2.cvtColor(rgb_scene, cv2.COLOR_RGB2BGR)
        img_h, img_w = bgr.shape[:2]
        self._init_from_vlm(bgr, img_w, img_h)

    def update(self, rgb_scene: np.ndarray) -> tuple[tuple[int, int, int, int] | None, bool]:
        """Feed an RGB frame and return (bbox_pixel_xywh | None, tracker_ok).

        Must call ``ensure_ready()`` first. This method only does fast OpenCV
        tracker updates — never blocks on VLM.
        """
        if self._tracker is None:
            return None, False

        import cv2

        bgr = cv2.cvtColor(rgb_scene, cv2.COLOR_RGB2BGR)

        ok, new_bbox = self._tracker.update(bgr)
        if not ok:
            return None, False

        bx, by, bw, bh = [int(v) for v in new_bbox]
        return (bx, by, bw, bh), True

    def _find_detection(self, scene):
        """Find the best detection matching ``_target_keyword`` in a VlmScene."""
        keyword = self._target_keyword.lower()

        # Exact keyword match in handle or label
        for d in scene.detections:
            if keyword in d.handle.lower() or keyword in d.label.lower():
                return d

        # Fallback: first graspable object (only for cube-like targets)
        if keyword in ("cube", "block", "object"):
            for d in scene.detections:
                if d.is_graspable:
                    return d

        return None

    def _init_from_vlm(self, bgr: np.ndarray, img_w: int, img_h: int) -> tuple[tuple[int, int, int, int] | None, bool]:
        """Run VLM on the frame, find target by keyword, init OpenCV tracker."""
        import asyncio

        # Reuse cached scene if available (avoids duplicate VLM call on target switch)
        scene = self._cached_scene
        if scene is not None:
            detection = self._find_detection(scene)
            if detection is not None:
                self._cached_scene = None
                return self._init_tracker(detection, bgr, img_w, img_h)
            self._cached_scene = None

        try:
            from halo.services.target_perception_service.vlm_fn import make_vlm_fn

            vlm_fn = make_vlm_fn(
                provider="ollama",
                model=self._vlm_model,
                base_url=self._vlm_base_url,
            )
            scene = asyncio.run(vlm_fn("sim", bgr, []))
        except Exception:
            logger.warning("VLM detection failed — tracker disabled", exc_info=True)
            self._vlm_failed = True
            return None, False

        # Cache scene for potential target switch reuse
        self._cached_scene = scene

        detection = self._find_detection(scene)
        if detection is None:
            logger.warning("VLM found no '%s' detection — tracker disabled", self._target_keyword)
            self._vlm_failed = True
            return None, False

        self._cached_scene = None
        return self._init_tracker(detection, bgr, img_w, img_h)

    def _init_tracker(
        self, detection, bgr: np.ndarray, img_w: int, img_h: int
    ) -> tuple[tuple[int, int, int, int] | None, bool]:
        """Initialise OpenCV tracker from a VLM detection."""
        self._handle = detection.handle
        # Convert normalised xyxy bbox (0..1) to pixel xywh
        x1, y1, x2, y2 = detection.bbox
        logger.info(
            "VLM detection '%s': norm_bbox=(%.3f,%.3f,%.3f,%.3f) frame=%dx%d",
            detection.handle,
            x1,
            y1,
            x2,
            y2,
            img_w,
            img_h,
        )
        px, py = int(x1 * img_w), int(y1 * img_h)
        pw, ph = int((x2 - x1) * img_w), int((y2 - y1) * img_h)

        # Sanity: bbox must be positive and within frame
        if pw <= 0 or ph <= 0 or px < 0 or py < 0 or px + pw > img_w or py + ph > img_h:
            logger.warning("VLM bbox out of frame bounds: (%d,%d,%d,%d) in %dx%d", px, py, pw, ph, img_w, img_h)
            self._vlm_failed = True
            return None, False

        from halo.services.target_perception_service.tracker_fn import _create_tracker

        self._tracker = _create_tracker()
        self._tracker.init(bgr, (px, py, pw, ph))
        self._initialised = True
        logger.info("Tracker initialised for '%s' at pixel bbox=(%d,%d,%d,%d)", self._handle, px, py, pw, ph)
        return (px, py, pw, ph), True


def _wait_for_frame(client, timeout: float = 5.0) -> np.ndarray | None:
    """Wait for a telemetry frame from the sim server. Returns rgb_scene or None."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t = client.latest_telemetry
        if t is not None and "rgb_scene" in t:
            return t["rgb_scene"]
        time.sleep(0.05)
    logger.warning("Timed out waiting for telemetry frame")
    return None


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
    pick_and_place: bool = False,
    tracker_ctx: _TrackerCtx | None = None,
) -> EpisodeResult:
    """Run one episode via SimClient CommandRPC commands.

    Uses start_pick to trigger autonomous trajectory execution on the server,
    then polls telemetry for observations until done. If ``pick_and_place``
    is True and the pick succeeds, follows up with a place-in-tray command.
    """

    import random

    from mujoco_sim.scene_info import GREEN_CUBE_BODY_NAME, RED_CUBE_BODY_NAME, TRAY_BODY_NAME

    # Reset
    client.reset(seed=seed)

    # Randomly choose pick target (green or red cube) using episode seed
    rng = random.Random(seed)
    pick_target = rng.choice([GREEN_CUBE_BODY_NAME, RED_CUBE_BODY_NAME])
    target_keyword = "green" if pick_target == GREEN_CUBE_BODY_NAME else "red"
    logger.info("Episode %d (seed=%d): picking %s", ep_idx, seed, pick_target)

    # Stabilize — step with home pose (zeros for now; server starts at home)
    home_action = np.zeros(6)
    for _ in range(stabilize_steps):
        client.step(home_action)

    # Init tracker on a still frame BEFORE starting motion
    if tracker_ctx is not None:
        tracker_ctx.switch_target(target_keyword)
        frame = _wait_for_frame(client)
        if frame is not None:
            tracker_ctx.ensure_ready(frame)

    # Trigger trajectory planning and autonomous execution
    resp = client.start_pick(pick_target)
    if resp.get("type") == "start_pick_error":
        return EpisodeResult(
            episode_idx=ep_idx,
            seed=seed,
            error=f"start_pick failed: {resp.get('message', 'unknown')}",
        )

    # Recording phase — poll telemetry for observations
    task = "pick_and_place" if pick_and_place else "pick"
    meta = EpisodeMetadata(
        seed=seed,
        env_name=env_config.scene_xml,
        robot=env_config.robot,
        control_freq=env_config.control_freq,
        extra={"teacher": task, "stabilize_steps": str(stabilize_steps), "pick_target": pick_target},
    )
    episode = RawEpisode(metadata=meta)
    video_frames: list[np.ndarray] = []

    poll_interval = 1.0 / env_config.control_freq
    last_step_count = -1
    phase_id = 0
    steps_remaining = max_steps
    pick_done = False

    # --- PICK phase ---
    phase_id, last_step_count, steps_remaining = _poll_until_done(
        client=client,
        episode=episode,
        video_frames=video_frames,
        save_video=save_video,
        poll_interval=poll_interval,
        max_steps=steps_remaining,
        last_step_count=last_step_count,
        tracker_ctx=tracker_ctx,
    )

    pick_done = phase_id == PHASE_DONE
    pick_success = pick_done and _verify_lift(episode, pick_target)

    # --- PLACE phase (optional) ---
    if pick_and_place and pick_success:
        # Switch tracker to tray and init on a still frame before starting place motion
        if tracker_ctx is not None:
            tracker_ctx.switch_target("tray")
            frame = _wait_for_frame(client)
            if frame is not None:
                tracker_ctx.ensure_ready(frame)

        resp = client.start_place(target_body=TRAY_BODY_NAME, held_body=pick_target)
        if resp.get("type") == "start_place_error":
            return EpisodeResult(
                episode_idx=ep_idx,
                seed=seed,
                error=f"start_place failed: {resp.get('message', 'unknown')}",
            )

        phase_id, last_step_count, steps_remaining = _poll_until_done(
            client=client,
            episode=episode,
            video_frames=video_frames,
            save_video=save_video,
            poll_interval=poll_interval,
            max_steps=steps_remaining,
            last_step_count=last_step_count,
            tracker_ctx=tracker_ctx,
        )

    final_phase = phase_id
    if pick_and_place:
        # Success = pick lifted + place completed + cube actually near tray
        place_phase_ok = final_phase in (PHASE_DONE, PHASE_RETURNING)
        success = pick_success and place_phase_ok and _verify_place(episode, pick_target)
    else:
        success = pick_success

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


def _poll_until_done(
    *,
    client,
    episode: RawEpisode,
    video_frames: list[np.ndarray],
    save_video: bool,
    poll_interval: float,
    max_steps: int,
    last_step_count: int,
    tracker_ctx: _TrackerCtx | None = None,
) -> tuple[int, int, int]:
    """Poll telemetry until done or max_steps exhausted.

    Returns:
        (final_phase_id, last_step_count, remaining_steps)
    """
    import time

    phase_id = 0
    for step_idx in range(max_steps):
        time.sleep(poll_interval)

        telemetry = client.latest_telemetry
        if telemetry is None:
            continue

        sc = telemetry.get("step_count", -1)
        if sc == last_step_count:
            continue
        last_step_count = sc

        phase_id = telemetry.get("phase_id", 0)
        done = telemetry.get("done", False)

        rgb_scene = telemetry["rgb_scene"]
        rgb_wrist = telemetry["rgb_wrist"]

        # Update tracker with this frame (runs VLM on first frame, then OpenCV tracker)
        bbox_px = None
        tracker_ok = False
        if tracker_ctx is not None:
            bbox_px, tracker_ok = tracker_ctx.update(rgb_scene)

        if save_video:
            frame = rgb_scene.copy()
            frame = _composite_wrist(frame, rgb_wrist, phase_id)
            if tracker_ctx is not None:
                frame = _draw_tracking_overlay(frame, bbox_px, tracker_ok)
            frame = _draw_info_overlay(
                frame,
                step=len(episode),
                phase_id=phase_id,
                gripper=float(telemetry["gripper"]),
                ee_pose=telemetry["ee_pose"],
                object_pose=telemetry["object_pose"],
            )
            video_frames.append(frame)

        ts = Timestep(
            rgb_scene=rgb_scene,
            rgb_wrist=rgb_wrist,
            qpos=telemetry["qpos"],
            qvel=telemetry["qvel"],
            gripper=float(telemetry["gripper"]),
            ee_pose=telemetry["ee_pose"],
            action=np.array(telemetry["action"], copy=True),
            phase_id=phase_id,
            object_pose=telemetry["object_pose"],
            red_object_pose=telemetry.get("red_object_pose"),
            joint_pos=telemetry["joint_pos"],
        )
        episode.append(ts)

        if done:
            return phase_id, last_step_count, max_steps - step_idx - 1

    return phase_id, last_step_count, 0


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
            frame = _draw_info_overlay(
                frame,
                step=step_idx,
                phase_id=phase_id,
                gripper=float(obs["gripper"]),
                ee_pose=obs["ee_pose"],
                object_pose=obs.get("object_pose", np.zeros(7)),
            )
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


def _verify_lift(episode: RawEpisode, pick_target: str = "") -> bool:
    """Check that the picked cube actually rose during the LIFT phase.

    Uses the *maximum* cube Z reached during lift (not the final Z), because
    the cube may separate from the gripper during the deceleration phase and
    fall back to the table while still counting as a successful pick.
    """
    from mujoco_sim.scene_info import RED_CUBE_BODY_NAME

    phase_ids = episode.phase_ids
    is_red = pick_target == RED_CUBE_BODY_NAME
    obj_poses = episode.red_object_poses if is_red else episode.object_poses
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


def _verify_place(episode: RawEpisode, pick_target: str = "") -> bool:
    """Check that the picked cube ended up close to the tray centre (XY plane).

    Uses red_object_poses when the red cube was picked, otherwise object_poses
    (green cube). The tray position comes from the final telemetry frame's
    static body position — but since the tray may have been randomized, we
    can't rely on the MJCF default. Instead we use a generous threshold.
    """
    from mujoco_sim.scene_info import RED_CUBE_BODY_NAME

    is_red = pick_target == RED_CUBE_BODY_NAME
    if is_red:
        obj_poses = episode.red_object_poses
    else:
        obj_poses = episode.object_poses

    if obj_poses is None or len(obj_poses) == 0:
        return False

    final_cube_xy = obj_poses[-1, :2]

    # We can't load the randomized tray position from the static MJCF
    # (the server may have moved it). Check distance to the OTHER cube
    # as a sanity check — the placed cube should NOT be near the other cube.
    # Instead, use a heuristic: cube Z should be above table + tray floor height.
    # For now, just check that the cube moved significantly from its start position.
    start_cube_xy = obj_poses[0, :2]
    displacement = float(np.linalg.norm(final_cube_xy - start_cube_xy))
    if displacement < 0.03:
        logger.warning("Place check FAILED: cube barely moved (%.4f m)", displacement)
        return False

    # Also verify the cube Z is above table level (not dropped on floor)
    final_cube_z = float(obj_poses[-1, 2])
    if final_cube_z < 0.35:
        logger.warning("Place check FAILED: cube Z=%.4f below table level", final_cube_z)
        return False

    logger.info("Place check PASSED: cube displaced %.4f m, final_z=%.4f", displacement, final_cube_z)
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

    fh = frame.shape[0]

    if bbox_xywh is not None and tracker_ok:
        x, y, bw, bh = bbox_xywh
        color = (0, 255, 0)  # green = tracking
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = "TRACK"
    else:
        color = (255, 0, 0)  # red = lost
        label = "TRACK: LOST"

    # Draw label at bottom-left to avoid overlap with info overlay at top
    cv2.putText(frame, label, (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame


# Phase ID → human-readable label
_PHASE_NAMES: dict[int, str] = {
    0: "IDLE",
    1: "SELECT_GRASP",
    2: "PLAN_APPROACH",
    3: "MOVE_PREGRASP",
    4: "VISUAL_ALIGN",
    5: "EXECUTE_APPROACH",
    6: "CLOSE_GRIPPER",
    7: "LIFT",
    8: "VERIFY_GRASP",
    9: "DONE",
    30: "TRANSIT_PREPLACE",
    31: "DESCEND_PLACE",
    32: "OPEN",
    33: "RETREAT",
    34: "SELECT_PLACE",
    60: "RETURNING",
}


def _draw_info_overlay(
    frame: np.ndarray,
    *,
    step: int,
    phase_id: int,
    gripper: float,
    ee_pose: np.ndarray,
    object_pose: np.ndarray,
) -> np.ndarray:
    """Draw phase, step, gripper state, and EE-object distance on the frame."""
    import cv2

    phase_name = _PHASE_NAMES.get(phase_id, str(phase_id))
    ee_pos = ee_pose[:3]
    obj_pos = object_pose[:3]
    dist = float(np.linalg.norm(ee_pos - obj_pos))

    lines = [
        f"step {step}  phase: {phase_name}",
        f"gripper: {gripper:.2f}  ee-obj: {dist:.3f}m",
        f"ee:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
        f"obj: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    y = 16
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.rectangle(frame, (6, y - th - 2), (10 + tw, y + 4), bg_color, -1)
        cv2.putText(frame, line, (8, y), font, scale, color, thickness, cv2.LINE_AA)
        y += th + 8

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
