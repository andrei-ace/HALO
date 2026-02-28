"""Logs live TUI run interactions (planner prompts + VLM responses) as JSONL.

Each run creates a directory under ``runs/``::

    runs/YYYYMMDD_HHMMSS_arm0/
        run.jsonl          # planner + VLM log entries
        vlm_001.jpg        # image sent to VLM
        vlm_002.jpg
        scene_001.txt      # SCENE_DESCRIBED text
        ...
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RunLogger:
    """Writes run artifacts to a per-session directory under *runs_dir*.

    Directory name: ``YYYYMMDD_HHMMSS_<arm_id>``

    Artifacts:
    - ``run.jsonl`` — planner interactions and VLM inference entries
    - ``vlm_NNN.jpg`` — image sent to VLM for each inference call
    - ``scene_NNN.txt`` — SCENE_DESCRIBED scene text
    """

    def __init__(self, runs_dir: Path, arm_id: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = runs_dir / f"{ts}_{arm_id}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self._run_dir / "run.jsonl"
        self._file = self._jsonl_path.open("w", encoding="utf-8")
        self._vlm_counter = 0
        print(f"Run log: {self._run_dir}")

    # ------------------------------------------------------------------

    def log_interaction(
        self,
        *,
        arm_id: str,
        operator_msg: str,
        snapshot: Any,
        commands: list[dict],
        acks: list[dict],
        reasoning: str = "",
        inference_ms: int = 0,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "kind": "planner",
            "ts": datetime.now(timezone.utc).isoformat(),
            "arm_id": arm_id,
            "operator_msg": operator_msg,
            "snapshot": snapshot,
            "commands": commands,
            "acks": acks,
            "reasoning": reasoning,
            "inference_ms": inference_ms,
            "error": error,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def log_vlm_inference(
        self,
        *,
        arm_id: str,
        target_handle: str,
        model: str,
        raw_response: Any,
        target_info: Any,
        inference_ms: int = 0,
        error: str | None = None,
        image: object | None = None,
        detections: list[dict] | None = None,
    ) -> None:
        self._vlm_counter += 1
        idx = f"{self._vlm_counter:03d}"

        # Save the image with detection bounding boxes overlaid
        img_file: str | None = None
        if image is not None:
            fname = f"vlm_{idx}.jpg"
            try:
                annotated = _annotate_image(image, detections or [])
                _save_image(annotated, self._run_dir / fname)
                img_file = fname
            except Exception:
                pass

        entry: dict[str, Any] = {
            "kind": "vlm",
            "ts": datetime.now(timezone.utc).isoformat(),
            "arm_id": arm_id,
            "target_handle": target_handle,
            "model": model,
            "raw_response": raw_response,
            "target_info": target_info,
            "inference_ms": inference_ms,
            "error": error,
            "image_file": img_file,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def log_scene_described(
        self,
        *,
        scene_text: str,
        detections: list[dict],
        image: object | None = None,
        inference_ms: int = 0,
    ) -> None:
        """Save SCENE_DESCRIBED artifacts: text file.

        Also appends a ``scene`` entry to the JSONL log.
        """
        idx = f"{self._vlm_counter:03d}"

        # Save scene text
        txt_path = self._run_dir / f"scene_{idx}.txt"
        txt_path.write_text(
            f"{scene_text}\n\n---\nDetections:\n"
            + "\n".join(f"  - {d.get('handle', '?')}: {d.get('label', '?')}" for d in detections),
            encoding="utf-8",
        )

        # Reuse the corresponding VLM image instead of writing scene_NNN.jpg.
        # If we have detections with bbox data, rewrite the overlay so labels
        # match any handle stabilisation done after VLM inference.
        vlm_fname = f"vlm_{idx}.jpg"
        vlm_path = self._run_dir / vlm_fname
        img_fname = vlm_fname if vlm_path.exists() else None
        if image is not None and img_fname is not None:
            try:
                annotated = _annotate_image(image, detections)
                _save_image(annotated, vlm_path)
            except Exception:
                pass

        # JSONL entry
        entry: dict[str, Any] = {
            "kind": "scene",
            "ts": datetime.now(timezone.utc).isoformat(),
            "scene_text": scene_text,
            "detections": detections,
            "image_file": img_fname,
            "inference_ms": inference_ms,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def log_tracker(self, *, event: str, target_handle: str, detail: str = "") -> None:
        """Log a tracker lifecycle event (init, failure, etc.)."""
        entry: dict[str, Any] = {
            "kind": "tracker",
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "target_handle": target_handle,
            "detail": detail,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._run_dir

    @property
    def run_dir(self) -> Path:
        return self._run_dir


def _to_pil(image: object) -> Image.Image:
    """Convert an image (numpy BGR HWC, PIL Image, or bytes) to PIL RGB."""
    if isinstance(image, np.ndarray):
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _annotate_image(image: object, detections: list[dict]) -> Image.Image:
    """Draw red bounding boxes and handle labels on a copy of *image*.

    *detections* is a list of dicts with ``bbox`` (x1, y1, x2, y2) and
    ``handle`` keys.  Bbox coordinates are in the same space as the image
    sent to the VLM (native resolution when width <= 1024, otherwise
    resized to 1024px width).
    """
    pil = _to_pil(image)
    if not detections:
        return pil

    # Match the resize logic in ollama_vlm_fn: only resize when the
    # image is wider than the VLM input width.
    from halo.services.target_perception_service.ollama_vlm_fn import _VLM_INPUT_WIDTH

    if pil.width > _VLM_INPUT_WIDTH:
        aspect = pil.height / pil.width
        new_w = _VLM_INPUT_WIDTH
        new_h = int(new_w * aspect)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox")
        handle = det.get("handle", "")
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if handle:
            draw.text((x1, max(y1 - 16, 0)), handle, fill="red", font=font)

    return pil


def _save_image(image: object, path: Path) -> None:
    """Save an image (numpy BGR HWC, PIL Image, or bytes) to *path*.

    Format is inferred from the file extension (JPEG for .jpg, PNG for .png).
    """
    fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    if isinstance(image, np.ndarray):
        import cv2

        cv2.imwrite(str(path), image)
    elif isinstance(image, Image.Image):
        image.save(path, format=fmt)
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
        pil.save(path, format=fmt)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
