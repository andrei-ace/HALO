"""Logs live TUI run interactions (planner prompts + VLM responses) as JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes one JSONL file per live session to the runs/ directory.

    File name: ``runs/YYYYMMDD_HHMMSS_<arm_id>.jsonl``

    Two entry kinds, distinguished by the ``kind`` field:

    Planner interaction::

        {
          "kind":         "planner",
          "ts":           "2026-02-23T14:32:09.123456+00:00",
          "arm_id":       "arm0",
          "operator_msg": "Retry the grasp once.",
          "snapshot":     { … },
          "commands":     [{"id": "…", "str": "START_SKILL(PICK, cube-1)"}],
          "acks":         [{"id": "…", "status": "ACCEPTED"}],
          "reasoning":    "…",
          "inference_ms": 1240,
          "error":        null
        }

    VLM inference::

        {
          "kind":           "vlm",
          "ts":             "2026-02-23T14:32:09.123456+00:00",
          "arm_id":         "arm0",
          "target_handle":  "cube-red-01",
          "model":          "qwen3-vl:30B",
          "raw_response":   { … },    // full Ollama response dict
          "target_info":    { … },    // serialised TargetInfo or null
          "inference_ms":   2150,
          "error":          null
        }
    """

    def __init__(self, runs_dir: Path, arm_id: str) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = runs_dir / f"{ts}_{arm_id}.jsonl"
        self._file = self._path.open("w", encoding="utf-8")
        print(f"Run log: {self._path}")

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
    ) -> None:
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
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
