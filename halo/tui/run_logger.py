"""Logs live TUI run interactions (planner prompts + responses) as JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes one JSONL file per live session to the runs/ directory.

    File name: ``runs/YYYYMMDD_HHMMSS_<arm_id>.jsonl``

    Each line is a JSON object describing one planner interaction::

        {
          "ts":           "2026-02-23T14:32:09.123456+00:00",
          "arm_id":       "arm0",
          "operator_msg": "Retry the grasp once.",
          "snapshot":     { … },   // snapshot_to_dict() output
          "commands":     [{"id": "…", "str": "START_SKILL(PICK, cube-1)"}],
          "acks":         [{"id": "…", "status": "ACCEPTED"}],
          "error":        null     // or error string on failure
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
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "arm_id": arm_id,
            "operator_msg": operator_msg,
            "snapshot": snapshot,
            "commands": commands,
            "acks": acks,
            "reasoning": reasoning,
            "error": error,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
