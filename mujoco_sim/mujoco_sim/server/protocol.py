"""Shared message protocol for HALO <-> MuJoCo sim ZMQ bridge.

Defines message type constants, numpy↔bytes serialization, and JPEG encode/decode
helpers used by both SimServer and SimClient.

Serialization: msgpack for structured data, raw bytes for numpy arrays and JPEG frames.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

# Ch1: Telemetry (Sim PUB → HALO SUB)
MSG_TELEMETRY = "telemetry"

# Ch2: Hints (HALO PUB → Sim SUB)
MSG_TRACKING_HINT = "tracking_hint"

# Ch3: Commands (HALO REQ → Sim REP)
CMD_STEP = "step"
CMD_RESET = "reset"
CMD_GET_STATE = "get_state"
CMD_SET_STATE = "set_state"
CMD_TEACHER_STEP = "teacher_step"
CMD_CONFIGURE = "configure"
CMD_SHUTDOWN = "shutdown"

# Ch3: Responses
RESP_STEP_OK = "step_ok"
RESP_RESET_OK = "reset_ok"
RESP_STATE = "state"
RESP_TEACHER_STEP_OK = "teacher_step_ok"
RESP_OK = "ok"
RESP_ERROR = "error"

# Ch4: Queries (Sim REQ → HALO REP)
QUERY_VLM_DETECT = "vlm_detect"
QUERY_TRACKER_INIT = "tracker_init"
QUERY_TRACKER_UPDATE = "tracker_update"

# Ch4: Responses
RESP_VLM_RESULT = "vlm_result"
RESP_NO_DETECT = "no_detect"
RESP_TRACKER_INIT_OK = "tracker_init_ok"
RESP_TRACKED = "tracked"


# ---------------------------------------------------------------------------
# Numpy ↔ bytes helpers
# ---------------------------------------------------------------------------


def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a contiguous numpy array to raw bytes."""
    return np.ascontiguousarray(arr).tobytes()


def bytes_to_ndarray(
    buf: bytes, dtype: np.dtype | type = np.float64, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Deserialize raw bytes to a numpy array.

    Args:
        buf: Raw bytes from ndarray_to_bytes.
        dtype: Numpy dtype (default float64).
        shape: Optional shape to reshape into.

    Returns:
        Numpy array.
    """
    arr = np.frombuffer(buf, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr.copy()  # writable copy


# ---------------------------------------------------------------------------
# JPEG encode / decode helpers
# ---------------------------------------------------------------------------


def jpeg_encode(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB uint8 image to JPEG bytes.

    Args:
        rgb: (H, W, 3) uint8 RGB array.
        quality: JPEG quality 0-100.

    Returns:
        JPEG-compressed bytes.
    """
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def jpeg_decode(buf: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 image.

    Args:
        buf: JPEG-compressed bytes.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    import cv2

    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Telemetry message helpers
# ---------------------------------------------------------------------------


def pack_telemetry(
    *,
    ts_ms: int,
    step_count: int,
    phase_id: int,
    done: bool,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ee_pose: np.ndarray,
    object_pose: np.ndarray,
    joint_pos: np.ndarray,
    gripper: float,
    action: np.ndarray,
    rgb_scene: np.ndarray,
    rgb_wrist: np.ndarray,
    jpeg_quality: int = 85,
) -> dict:
    """Build a telemetry message dict (ready for msgpack)."""
    return {
        "type": MSG_TELEMETRY,
        "ts_ms": ts_ms,
        "step_count": step_count,
        "phase_id": phase_id,
        "done": done,
        "qpos": ndarray_to_bytes(qpos),
        "qvel": ndarray_to_bytes(qvel),
        "ee_pose": ndarray_to_bytes(ee_pose),
        "object_pose": ndarray_to_bytes(object_pose),
        "joint_pos": ndarray_to_bytes(joint_pos),
        "gripper": gripper,
        "action": ndarray_to_bytes(action),
        "rgb_scene": jpeg_encode(rgb_scene, jpeg_quality),
        "rgb_wrist": jpeg_encode(rgb_wrist, jpeg_quality),
    }


def unpack_telemetry(msg: dict) -> dict:
    """Decode a telemetry message dict into numpy arrays + native types."""
    return {
        "type": msg["type"],
        "ts_ms": msg["ts_ms"],
        "step_count": msg["step_count"],
        "phase_id": msg["phase_id"],
        "done": msg["done"],
        "qpos": bytes_to_ndarray(msg["qpos"], shape=(13,)),
        "qvel": bytes_to_ndarray(msg["qvel"], shape=(12,)),
        "ee_pose": bytes_to_ndarray(msg["ee_pose"], shape=(7,)),
        "object_pose": bytes_to_ndarray(msg["object_pose"], shape=(7,)),
        "joint_pos": bytes_to_ndarray(msg["joint_pos"], shape=(6,)),
        "gripper": msg["gripper"],
        "action": bytes_to_ndarray(msg["action"], shape=(6,)),
        "rgb_scene": jpeg_decode(msg["rgb_scene"]),
        "rgb_wrist": jpeg_decode(msg["rgb_wrist"]),
    }
