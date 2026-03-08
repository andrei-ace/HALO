"""HALO TUI — static mockup (v0).

Run with:
    uv run python -m halo.tui.app
    uv run python -m halo.tui.app --live
    uv run python -m halo.tui.app --live --arm arm0 --model llama3.2:3b --base-url http://localhost:11434
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from halo.tui.run_logger import RunLogger

_RUNS_DIR = Path(__file__).parents[2] / "runs"


def _copy_text(text: str) -> bool:
    """Copy text to OS clipboard. Returns True on success."""
    import platform
    import subprocess

    sys = platform.system()
    try:
        if sys == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif sys == "Linux":
            for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "-bi"], ["wl-copy"]]:
                try:
                    subprocess.run(cmd, input=text.encode(), check=True)
                    return True
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            return False
        elif sys == "Windows":
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True)
        else:
            return False
        return True
    except Exception:
        return False


# ── Static fixtures ───────────────────────────────────────────────

_DATA = dict(
    arm_id="arm0",
    # ── Skill Runner state ──
    skill_name="PICK",
    skill_run_id="run-9",
    skill_phase="PREGRASP_ALIGN",
    act_status="RUNNING",
    act_buffer_ms=240,
    act_buffer_low=False,
    outcome_state="IN_PROGRESS",
    outcome_reason=None,
    elapsed_ms=3200,
    # ── Target Perception state ──
    perc_tracking="TRACKING",
    perc_target="cube-1",
    perc_center_px=(0.76, 0.65),
    perc_distance_m=0.09,
    perc_confidence=0.86,
    perc_obs_age_ms=45,
    perc_failure="OK",
    perc_hint_valid=True,
    perc_active_consumed=None,
    perc_active_total=None,
    perc_pending_consumed=None,
    perc_pending_total=None,
    perc_has_pending=False,
    # ── Control Service state ──
    ctrl_status="RUNNING",
    ctrl_safety="OK",
    ctrl_reflex=None,
    ctrl_action_xyz=(0.12, -0.05, 0.08),
    ctrl_gripper=0.0,
    servos=[
        ("J1", "shldr_pan", "OK", 0.00, 0.01),
        ("J2", "shldr_lift", "OK", -0.79, 0.02),
        ("J3", "elbow", "OK", 0.00, -0.01),
        ("J4", "wrist_flex", "OK", -2.36, 0.03),
        ("J5", "wrist_roll", "OK", 0.00, 0.00),
        ("J6", "gripper", "OK", -0.17, -0.02),
    ],
    # Prompt history — shown newest-at-bottom inside the Talk panel
    prompt_history=[
        ("14:31:44", "What objects are visible on the table?"),
        ("14:31:50", "Retry the grasp once."),
        ("14:32:05", "Use cube-2 instead, then continue with the bin."),
    ],
    services=[
        ("TargetPerception", "TRACKING", "bright_green"),
        ("SkillRunner", "RUNNING", "bright_green"),
        ("ControlService", "RUNNING", "bright_green"),
        ("Safety", "OK", "yellow"),
    ],
    backend="local",
    events=[
        ("14:32:06", "SKILL_STARTED run-9"),
        ("14:32:07", "COMMAND_ACCEPTED START_SKILL"),
        ("14:32:08", "PERCEPTION_RECOVERED"),
    ],
)

_EMPTY_DATA = dict(
    arm_id="arm0",
    skill_name=None,
    skill_run_id=None,
    skill_phase=None,
    act_status=None,
    act_buffer_ms=None,
    act_buffer_low=False,
    outcome_state=None,
    outcome_reason=None,
    elapsed_ms=None,
    perc_tracking=None,
    perc_target=None,
    perc_center_px=None,
    perc_distance_m=None,
    perc_confidence=None,
    perc_obs_age_ms=None,
    perc_failure=None,
    perc_hint_valid=None,
    perc_active_consumed=None,
    perc_active_total=None,
    perc_pending_consumed=None,
    perc_pending_total=None,
    perc_has_pending=None,
    ctrl_status=None,
    ctrl_safety=None,
    ctrl_reflex=None,
    ctrl_action_xyz=None,
    ctrl_gripper=None,
    servos=[
        ("J1", "shldr_pan", "NC", None, None),
        ("J2", "shldr_lift", "NC", None, None),
        ("J3", "elbow", "NC", None, None),
        ("J4", "wrist_flex", "NC", None, None),
        ("J5", "wrist_roll", "NC", None, None),
        ("J6", "gripper", "NC", None, None),
    ],
    prompt_history=[],
    services=[
        ("TargetPerception", "—", "#9e9e9e"),
        ("SkillRunner", "—", "#9e9e9e"),
        ("ControlService", "—", "#9e9e9e"),
        ("Safety", "—", "#9e9e9e"),
    ],
    backend="",
    events=[],
)

_LEGEND = [
    ("Tab / Shift+Tab", "Navigate between input and buttons"),
    ("T", "Focus the message input directly"),
    ("Enter", "Send message to planner"),
    ("Esc", "Clear input and return to monitoring mode"),
    ("R", "Show full planner reasoning"),
    ("Y", "Yank — copy last reasoning to clipboard instantly"),
    ("F", "Toggle OpenCV camera feed viewer (live mode only)"),
    ("M", "Toggle microphone mute (live agent only)"),
    ("Ctrl+A", "Emergency abort — blows the safety fuse"),
    ("Ctrl+R", "Reset fuse — re-enable FSM / skill execution"),
    ("Ctrl+Q", "Quit"),
    ("?", "Show / hide this legend"),
]


# ── Helpers ───────────────────────────────────────────────────────


def _bar(value: float, width: int = 8) -> Text:
    """Colored Unicode load bar (█ filled, ░ empty)."""
    n = round(value * width)
    color = "bright_green" if value < 0.45 else "yellow" if value < 0.80 else "red"
    t = Text(no_wrap=True)
    t.append("█" * n, style=color)
    t.append("░" * (width - n), style="grey30")
    return t


def _format_cmd(cmd: object) -> str:
    """Convert a CommandEnvelope to a concise human-readable string."""
    from halo.contracts.commands import (
        AbortSkillPayload,
        DescribeScenePayload,
        StartSkillPayload,
    )

    p = cmd.payload  # type: ignore[attr-defined]
    if isinstance(p, StartSkillPayload):
        return f"START_SKILL({p.skill_name.value}, {p.target_handle})"
    if isinstance(p, AbortSkillPayload):
        return f"ABORT_SKILL({p.skill_run_id[-4:]}, {p.reason})"
    if isinstance(p, DescribeScenePayload):
        return f"DESCRIBE_SCENE({p.reason})"
    return str(cmd.type)  # type: ignore[attr-defined]


def _format_event(evt: object) -> str:
    """Convert an EventEnvelope to a concise display string."""
    name = str(getattr(evt, "type", "?"))
    data = getattr(evt, "data", {}) or {}
    run_id = data.get("skill_run_id", "")
    reason = data.get("reason", "")
    cmd_type = data.get("command_type", "")
    skill_name = data.get("skill_name", "")
    if name == "SKILL_STARTED":
        return f"{name} {skill_name or run_id}"
    if name == "SKILL_SUCCEEDED":
        return f"{name} {skill_name or run_id}"
    if name == "SKILL_FAILED":
        if reason == "abort":
            return f"SKILL_ABORTED {skill_name} (operator requested stop)"
        failure_code = data.get("failure_code", "")
        detail = reason or failure_code
        return f"SKILL_FAILED {skill_name} reason={detail}" if detail else f"SKILL_FAILED {skill_name}"
    if name == "SAFETY_REFLEX_TRIGGERED" and reason:
        return f"{name} {reason}"
    if name in ("COMMAND_ACCEPTED", "COMMAND_REJECTED") and cmd_type:
        return f"{name} {cmd_type}"
    if name == "SCENE_DESCRIBED":
        count = data.get("count", 0)
        ms = data.get("inference_ms", 0)
        scene = data.get("scene", "")
        snippet = (scene[:60] + "…") if len(scene) > 60 else scene
        return f"SCENE_DESCRIBED {count} obj  ({ms} ms)  {snippet}"
    if name == "TARGET_ACQUIRED":
        handle = data.get("target_handle", "?")
        return f"TARGET_ACQUIRED {handle}"
    return name


def _snap_to_panel_data(snap: object, base: dict) -> dict:
    """Overlay live snapshot values onto *base* (typically _EMPTY_DATA)."""
    data = dict(base)

    skill = getattr(snap, "skill", None)
    act = getattr(snap, "act", None)
    progress = getattr(snap, "progress", None)
    outcome = getattr(snap, "outcome", None)
    target = getattr(snap, "target", None)
    perc = getattr(snap, "perception", None)

    # Planner / SkillRunner fields
    if skill is not None:
        data["skill_name"] = str(getattr(getattr(skill, "name", None), "value", ""))
        data["skill_run_id"] = getattr(skill, "skill_run_id", None)
        phase = getattr(skill, "phase", None)
        data["skill_phase"] = phase.name if phase is not None else None
    if act is not None:
        data["act_status"] = str(getattr(getattr(act, "status", None), "value", ""))
        data["act_buffer_ms"] = getattr(act, "buffer_fill_ms", None)
        data["act_buffer_low"] = getattr(act, "buffer_low", False)
    if progress is not None:
        data["elapsed_ms"] = getattr(progress, "elapsed_ms", None)
    if outcome is not None:
        data["outcome_state"] = str(getattr(getattr(outcome, "state", None), "value", ""))
        rc = getattr(outcome, "reason_code", None)
        data["outcome_reason"] = str(rc.value) if rc is not None else None

    # Target Perception fields
    if perc is not None:
        data["perc_tracking"] = str(getattr(getattr(perc, "tracking_status", None), "value", ""))
        data["perc_failure"] = str(getattr(getattr(perc, "failure_code", None), "value", ""))
        data["perc_active_consumed"] = getattr(perc, "active_buf_consumed", 0)
        data["perc_active_total"] = getattr(perc, "active_buf_total", 0)
        data["perc_pending_consumed"] = getattr(perc, "pending_buf_consumed", 0)
        data["perc_pending_total"] = getattr(perc, "pending_buf_total", 0)
        data["perc_has_pending"] = getattr(perc, "has_pending_tracker", False)
    if target is not None:
        data["perc_target"] = getattr(target, "handle", None)
        data["perc_center_px"] = getattr(target, "center_px", None)
        data["perc_distance_m"] = getattr(target, "distance_m", None)
        data["perc_confidence"] = getattr(target, "confidence", None)
        data["perc_obs_age_ms"] = getattr(target, "obs_age_ms", None)
        data["perc_hint_valid"] = getattr(target, "hint_valid", None)

    return data


def _derive_services(snap: object) -> list[tuple[str, str, str]]:
    """Return service_rows from a PlannerSnapshot."""
    rows: list[tuple[str, str, str]] = []

    perc = getattr(snap, "perception", None)
    if perc is not None:
        ts = getattr(perc, "tracking_status", None)
        status = str(ts.value) if ts else "UNKNOWN"
        color = (
            "bright_green" if status == "TRACKING" else "yellow" if status in ("RELOCALIZING", "REACQUIRING") else "red"
        )
        rows.append(("TargetPerception", status, color))

    skill = getattr(snap, "skill", None)
    if skill is not None:
        phase = getattr(skill, "phase", None)
        rows.append(("SkillRunner", phase.name if phase else "ACTIVE", "bright_green"))
    else:
        rows.append(("SkillRunner", "IDLE", "#9e9e9e"))

    act = getattr(snap, "act", None)
    if act is not None:
        status = str(getattr(getattr(act, "status", None), "value", "UNKNOWN"))
        color = "bright_green" if status == "RUNNING" else "yellow" if status == "BUFFER_LOW" else "#9e9e9e"
        rows.append(("ControlService", status, color))

    safety = getattr(snap, "safety", None)
    if safety is not None:
        state = str(getattr(getattr(safety, "state", None), "value", "UNKNOWN"))
        reflex = getattr(safety, "reflex_active", False)
        color = "red" if reflex or state == "FAULT" else "bright_green"
        rows.append(("Safety", f"{state} (REFLEX)" if reflex else state, color))

    return rows


# ── Panel widgets ─────────────────────────────────────────────────


class PlannerPanel(Container):
    _ROW_IDS = ("skill", "phase", "act", "outcome", "elapsed")

    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Skill Runner"

    @staticmethod
    def _build_texts(data: dict) -> list[Text]:
        texts: list[Text] = []
        no_data = data["skill_name"] is None
        # Skill + run id
        t = Text()
        t.append("Skill:    ", style="bold white")
        if no_data:
            t.append("—", style="#9e9e9e")
        else:
            t.append(data["skill_name"], style="bold #4fc3f7")
            t.append(f"  {data['skill_run_id']}", style="#9e9e9e")
        texts.append(t)
        # Phase
        t2 = Text()
        t2.append("Phase:    ", style="bold white")
        t2.append("—" if no_data else data["skill_phase"], style="#9e9e9e" if no_data else "bold white")
        texts.append(t2)
        # ACT buffer
        t3 = Text()
        t3.append("ACT:      ", style="bold white")
        if no_data:
            t3.append("—", style="#9e9e9e")
        else:
            buf = data["act_buffer_ms"]
            low = data["act_buffer_low"]
            buf_color = "yellow" if low else "bright_green"
            t3.append(data["act_status"], style=f"bold {buf_color}")
            if buf is not None:
                t3.append(f"  buffer: {buf} ms", style="#9e9e9e")
            if low:
                t3.append("  !", style="bold yellow")
        texts.append(t3)
        # Outcome
        t4 = Text()
        t4.append("Outcome:  ", style="bold white")
        if no_data:
            t4.append("—", style="#9e9e9e")
        else:
            outcome = data["outcome_state"]
            outcome_color = "bright_green" if outcome == "SUCCESS" else "red" if outcome == "FAILURE" else "#b0bcd0"
            t4.append(outcome, style=f"bold {outcome_color}")
            if data["outcome_reason"]:
                t4.append(f"  ({data['outcome_reason']})", style="yellow")
        texts.append(t4)
        # Elapsed
        t5 = Text()
        t5.append("Elapsed:  ", style="bold white")
        if no_data:
            t5.append("—", style="#9e9e9e")
        else:
            elapsed_s = (data["elapsed_ms"] or 0) / 1000
            t5.append(f"{elapsed_s:.1f} s", style="#9e9e9e")
        texts.append(t5)
        return texts

    def compose(self) -> ComposeResult:
        for rid, text in zip(self._ROW_IDS, self._build_texts(self._data)):
            yield Static(text, id=f"planner-{rid}")

    def refresh_live(self, data: dict) -> None:
        self._data = data
        for rid, text in zip(self._ROW_IDS, self._build_texts(data)):
            self.query_one(f"#planner-{rid}", Static).update(text)


class TargetPerceptionPanel(Container):
    _ROW_IDS = ("status", "target", "center", "distance", "confidence", "obs-age", "code", "active-buf", "pending-buf")

    def __init__(self, data: dict = _DATA, tracker_name: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data
        self._tracker_name = tracker_name

    def on_mount(self) -> None:
        if self._tracker_name:
            self.border_title = f"Target Perception ({self._tracker_name})"
        else:
            self.border_title = "Target Perception"

    @staticmethod
    def _build_texts(data: dict) -> list[Text]:
        def row(label: str, value: str, value_style: str) -> Text:
            t = Text()
            t.append(f"{label:<12}", style="bold white")
            t.append(value, style=value_style)
            return t

        texts: list[Text] = []
        tracking = data["perc_tracking"]
        no_data = tracking is None

        # Status
        if no_data:
            texts.append(row("Status:", "—", "#9e9e9e"))
        else:
            track_color = (
                "bright_green"
                if tracking == "TRACKING"
                else "yellow"
                if tracking in ("RELOCALIZING", "REACQUIRING")
                else "red"
            )
            t = Text()
            t.append(f"{'Status:':<12}", style="bold white")
            t.append(tracking, style=f"bold {track_color}")
            if data["perc_hint_valid"] is False:
                t.append("  invalid", style="bold red")
            texts.append(t)

        # Target
        target = data["perc_target"]
        texts.append(row("Target:", target or "—", "#4fc3f7" if target else "#9e9e9e"))

        # Bbox center (normalised 0..1)
        ctr = data.get("perc_center_px")
        if ctr is not None:
            texts.append(row("Center:", f"{ctr[0]:.2f}, {ctr[1]:.2f}", "white"))
        else:
            texts.append(row("Center:", "—", "#9e9e9e"))

        # Distance
        dist = data["perc_distance_m"]
        texts.append(
            row(
                "Distance:",
                f"{dist * 100:.0f} cm" if dist is not None else "—",
                "white" if dist is not None else "#9e9e9e",
            )
        )

        # Confidence
        conf = data["perc_confidence"]
        texts.append(
            row(
                "Confidence:",
                f"{conf * 100:.0f}%" if conf is not None else "—",
                "white" if conf is not None else "#9e9e9e",
            )
        )

        # Obs age
        age = data["perc_obs_age_ms"]
        texts.append(row("Obs age:", f"{age} ms" if age is not None else "—", "#9e9e9e"))

        # Failure code
        failure = data["perc_failure"]
        fail_color = "bright_green" if failure == "OK" else ("yellow" if failure else "#9e9e9e")
        texts.append(row("Code:", failure or "—", fail_color))

        # Active tracker buffer (consumed/total this tick)
        has_pending = data.get("perc_has_pending", False)
        a_con = data.get("perc_active_consumed")
        a_tot = data.get("perc_active_total")
        if a_tot is not None and a_tot > 0:
            a_color = "#9e9e9e" if has_pending else "bright_green"
            texts.append(row("Active:", f"{a_con}/{a_tot}", a_color))
        else:
            texts.append(row("Active:", "—", "#9e9e9e"))

        # Pending tracker buffer (consumed/remaining this tick)
        p_con = data.get("perc_pending_consumed")
        p_tot = data.get("perc_pending_total")
        if p_tot is not None and p_tot > 0:
            texts.append(row("Pending:", f"{p_con}/{p_tot}", "yellow"))
        elif has_pending:
            texts.append(row("Pending:", "0", "yellow"))
        else:
            texts.append(row("Pending:", "—", "#9e9e9e"))

        return texts

    def compose(self) -> ComposeResult:
        for rid, text in zip(self._ROW_IDS, self._build_texts(self._data)):
            yield Static(text, id=f"perc-{rid}")

    def refresh_live(self, data: dict) -> None:
        self._data = data
        for rid, text in zip(self._ROW_IDS, self._build_texts(data)):
            self.query_one(f"#perc-{rid}", Static).update(text)


class ControlServicePanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Control Service"

    def compose(self) -> ComposeResult:
        status = self._data["ctrl_status"]
        if status is None:
            yield Static(Text("no control data", style="#9e9e9e"))
            return
        status_color = "bright_green" if status == "RUNNING" else "red"
        t = Text()
        t.append("Status:   ", style="bold white")
        t.append(status, style=f"bold {status_color}")
        yield Static(t)

        safety = self._data["ctrl_safety"]
        reflex = self._data["ctrl_reflex"]
        t2 = Text()
        t2.append("Safety:   ", style="bold white")
        safety_color = "bright_green" if safety == "OK" else "red"
        t2.append(safety or "—", style=f"bold {safety_color}")
        if reflex:
            t2.append(f"  reflex: {reflex}", style="bold red")
        yield Static(t2)

        xyz = self._data["ctrl_action_xyz"]
        gripper = self._data["ctrl_gripper"]
        t3 = Text()
        t3.append("Δ EE:     ", style="bold white")
        if xyz is not None:
            t3.append(f"{xyz[0]:+.2f}  {xyz[1]:+.2f}  {xyz[2]:+.2f}", style="#b0bcd0")
            t3.append(f"   grip: {gripper:+.2f}" if gripper is not None else "", style="#9e9e9e")
        else:
            t3.append("—", style="#9e9e9e")
        yield Static(t3)


class ServosPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Joints (6DOF)"

    @staticmethod
    def _vel_bar(vel: float, width: int = 6, max_vel: float = 2.0) -> Text:
        """Velocity magnitude bar (0..max_vel mapped to width chars)."""
        frac = min(abs(vel) / max_vel, 1.0)
        n = round(frac * width)
        color = "bright_green" if frac < 0.4 else "yellow" if frac < 0.75 else "red"
        t = Text(no_wrap=True)
        t.append("█" * n, style=color)
        t.append("░" * (width - n), style="grey30")
        return t

    @staticmethod
    def _build_row(jid: str, name: str, status: str, pos: float | None, vel: float | None) -> Text:
        t = Text(no_wrap=True)
        t.append(f"{jid} ", style="bold white")
        t.append(f"{name:<10}", style="white")
        if status == "NC":
            t.append("NC   ", style="#9e9e9e")
            t.append("not connected", style="#9e9e9e")
        else:
            t.append(f"{pos:+7.2f} rad  " if pos is not None else "  —  rad  ", style="#b0bcd0")
            t.append_text(ServosPanel._vel_bar(vel if vel is not None else 0.0))
        return t

    def compose(self) -> ComposeResult:
        for i, (jid, name, status, pos, vel) in enumerate(self._data["servos"]):
            yield Static(self._build_row(jid, name, status, pos, vel), id=f"servo-{i}")

    def refresh_live(self, servos: list[tuple[str, str, str, float | None, float | None]]) -> None:
        for i, (jid, name, status, pos, vel) in enumerate(servos):
            try:
                self.query_one(f"#servo-{i}", Static).update(self._build_row(jid, name, status, pos, vel))
            except Exception:
                pass


class TalkPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Talk to Planner"
        # scroll history to bottom after layout settles
        self.call_after_refresh(lambda: self.query_one("#prompt-history", VerticalScroll).scroll_end(animate=False))

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="prompt-history"):
            for ts, msg in self._data["prompt_history"]:
                t = Text()
                t.append(ts, style="grey62")
                t.append("  ")
                t.append(msg, style="#b0bcd0")
                yield Static(t, classes="history-item")
        yield Input(placeholder="Type a command…", id="planner-input")


class SystemPanel(Container):
    _SERVICE_NAMES = ("TargetPerception", "SkillRunner", "ControlService", "Safety")

    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "System"

    @staticmethod
    def _service_text(name: str, status: str, color: str) -> Text:
        t = Text()
        t.append("● ", style=color)
        t.append(f"{name:<16}  ", style="white")
        t.append(status, style=f"bold {color}")
        return t

    @staticmethod
    def _backend_text(backend: str) -> Text:
        t = Text()
        t.append("  ")  # match "● " column
        t.append(f"{'Backend':<16}  ", style="bold white")
        label = backend.upper() if backend else "—"
        color = "bright_cyan" if label == "CLOUD" else "bright_green" if label == "LOCAL" else "#9e9e9e"
        t.append(label, style=f"bold {color}")
        return t

    def compose(self) -> ComposeResult:
        for name, status, color in self._data["services"]:
            yield Static(self._service_text(name, status, color), id=f"sys-svc-{name}")
        yield Static(self._backend_text(self._data.get("backend", "")), id="sys-backend")

    def refresh_live(self, services: list[tuple[str, str, str]], *, backend: str = "") -> None:
        """Update content in-place — no DOM churn."""
        for name, status, color in services:
            self.query_one(f"#sys-svc-{name}", Static).update(self._service_text(name, status, color))
        self.query_one("#sys-backend", Static).update(self._backend_text(backend))


class EventsPanel(VerticalScroll):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Events"

    def compose(self) -> ComposeResult:
        for ts, desc in self._data["events"]:
            t = Text()
            t.append(ts, style="grey62")
            t.append("  ")
            t.append(desc)
            yield Static(t)

    async def append_event(self, evt: object) -> None:
        """Append a live EventEnvelope row (newest at bottom). Trims to 8 rows."""
        from datetime import datetime

        ts_ms = getattr(evt, "ts_ms", 0)
        ts = datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")
        t = Text()
        t.append(ts, style="grey62")
        t.append("  ")
        t.append(_format_event(evt))
        await self.mount(Static(t))
        children = list(self.query("Static"))
        if len(children) > 8:
            await children[0].remove()
        self.scroll_end(animate=False)


class AudioPanel(Container):
    """Live audio status panel — mic/speaker state, transcriptions."""

    def on_mount(self) -> None:
        self.border_title = "Voice"

    def compose(self) -> ComposeResult:
        yield Static(Text("Mic: —", style="#9e9e9e"), id="audio-mic")
        yield Static(Text("Speaker: —", style="#9e9e9e"), id="audio-speaker")
        yield Static(Text("", style="#9e9e9e"), id="audio-transcript-in")
        yield Static(Text("", style="#9e9e9e"), id="audio-transcript-out")

    def refresh_live(
        self,
        mic_status: str,
        speaker_status: str,
        transcript_in: str = "",
        transcript_out: str = "",
        mic_level: float = 0.0,
        speaker_level: float = 0.0,
        transcript_in_finished: bool = True,
        transcript_out_finished: bool = True,
    ) -> None:
        mic_text = Text()
        mic_text.append("Mic: ", style="bold white")
        mic_text.append(mic_status, style="bright_green" if "Listening" in mic_status else "#9e9e9e")
        if mic_status == "Listening":
            bars = int(mic_level * 20)
            mic_text.append("  ")
            mic_text.append("█" * bars, style="bright_green")
            mic_text.append("░" * (20 - bars), style="#3a4060")
        self.query_one("#audio-mic", Static).update(mic_text)

        spk_text = Text()
        spk_text.append("Speaker: ", style="bold white")
        if speaker_status == "Interrupted":
            spk_text.append(speaker_status, style="#ffa726")
        elif "Speaking" in speaker_status:
            spk_text.append(speaker_status, style="bright_green")
        else:
            spk_text.append(speaker_status, style="#9e9e9e")
        if speaker_level > 0.001:
            bars = int(speaker_level * 20)
            spk_text.append("  ")
            spk_text.append("█" * bars, style="#66bb6a")
            spk_text.append("░" * (20 - bars), style="#3a4060")
        self.query_one("#audio-speaker", Static).update(spk_text)

        if transcript_in:
            tin = Text()
            tin.append("You: ", style="bold #4fc3f7")
            display_in = transcript_in[-80:] if transcript_in_finished else transcript_in[-77:] + "..."
            tin.append(display_in, style="#b0bcd0")
            self.query_one("#audio-transcript-in", Static).update(tin)

        if transcript_out:
            tout = Text()
            tout.append("AI: ", style="bold #66bb6a")
            display_out = transcript_out[-80:] if transcript_out_finished else transcript_out[-77:] + "..."
            tout.append(display_out, style="#b0bcd0")
            self.query_one("#audio-transcript-out", Static).update(tout)


class PanicPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Panic"

    def compose(self) -> ComposeResult:
        with Horizontal(id="abort-row"):
            yield Button("ABORT NOW!", id="abort-btn")
            yield Button("RESET", id="reset-btn", disabled=True)
        yield Static("Ctrl+A abort  |  Ctrl+R reset", id="abort-hint")


# ── Chrome ────────────────────────────────────────────────────────


class TitleBar(Static):
    def __init__(self, arm_id: str = _DATA["arm_id"], **kwargs) -> None:
        super().__init__(**kwargs)
        self._arm_id = arm_id

    def render(self) -> Text:
        t = Text(justify="center")
        t.append(f"HALO  —  {self._arm_id}", style="bold white")
        return t


class HintBar(Static):
    """Single-line keybinding hint that spans the full width."""

    def render(self) -> Text:
        t = Text(justify="center")
        t.append("[ ? ] legend", style="#4fc3f7")
        for key, desc in (
            ("Tab", "navigate"),
            ("T", "type"),
            ("Enter", "send"),
            ("Esc", "cancel"),
            ("M", "mic"),
            ("Ctrl+A", "abort"),
            ("Ctrl+R", "reset"),
            ("Ctrl+Q", "quit"),
        ):
            t.append("  ·  ", style="#3a4060")
            t.append(f"[ {key} ]", style="#4fc3f7")
            t.append(f" {desc}", style="#9e9e9e")
        return t


# ── Legend modal ──────────────────────────────────────────────────


class LegendScreen(ModalScreen):
    BINDINGS = [
        Binding("question_mark", "close", show=False),
        Binding("escape", "close", show=False),
    ]

    def action_close(self) -> None:
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Container(id="legend-box"):
            yield Static(" Keyboard shortcuts", id="legend-heading")
            for key, desc in _LEGEND:
                t = Text()
                t.append(f" [ {key} ]", style="bold #4fc3f7")
                t.append(f"  {desc}", style="#b0bcd0")
                yield Static(t)
            yield Static("")
            yield Static(" Press ? or Esc to close", id="legend-close-hint")


# ── Reasoning modal ───────────────────────────────────────────────


class ReasoningScreen(ModalScreen):
    """Full LLM reasoning text for the last planner response."""

    BINDINGS = [
        Binding("r", "close", show=False),
        Binding("escape", "close", show=False),
        Binding("y", "copy_all", "copy all", show=False),
    ]

    def __init__(self, reasoning: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._reasoning = reasoning

    def action_close(self) -> None:
        self.dismiss()

    def action_copy_all(self) -> None:
        if _copy_text(self._reasoning):
            self.notify("Copied to clipboard")
        else:
            self.notify("Copy failed — check clipboard tool", severity="warning")

    def compose(self) -> ComposeResult:
        with Container(id="reasoning-box"):
            yield Static(" Planner reasoning (last response)", id="reasoning-heading")
            with VerticalScroll(id="reasoning-scroll"):
                if self._reasoning:
                    for line in self._reasoning.splitlines():
                        yield Static(line or " ", classes="reasoning-line")
                else:
                    yield Static("  (no reasoning captured yet)", classes="reasoning-line")
            yield Static(" Y copy all  ·  R or Esc close", id="reasoning-close-hint")


# ── App ───────────────────────────────────────────────────────────


class HALOApp(App):
    CSS = """
    Screen {
        background: #0d1320;
        color: #b0bcd0;
    }

    TitleBar {
        height: 1;
        dock: top;
        content-align: center middle;
        background: #141d30;
        color: white;
        text-style: bold;
    }

    /* ── Body: vertical so HintBar gets exactly 1 row at the bottom ── */
    #body {
        height: 1fr;
        layout: vertical;
    }

    #main-row {
        height: 1fr;
        layout: horizontal;
    }

    HintBar {
        height: 1;
        content-align: center middle;
        background: #141d30;
    }

    #left-col {
        width: 3fr;
    }

    #right-col {
        width: 2fr;
    }

    /* ── Panel borders ── */
    PlannerPanel,
    TargetPerceptionPanel,
    ControlServicePanel,
    ServosPanel,
    TalkPanel,
    SystemPanel,
    EventsPanel,
    PanicPanel,
    AudioPanel {
        border: solid #263050;
        border-title-color: #4fc3f7;
        border-title-style: bold;
        padding: 0 1;
        height: auto;
    }

    AudioPanel            { height: auto; min-height: 6; display: none; }
    AudioPanel.visible    { display: block; }

    /* info containers shrink first; Talk + Panic hold their space */
    #left-info            { height: 1fr; min-height: 0; overflow: hidden hidden; }
    #right-info           { height: 1fr; min-height: 0; overflow: hidden hidden; }
    TalkPanel             { height: 1fr; min-height: 10; }
    PanicPanel            { height: auto; min-height: 8;  }

    PlannerPanel          { height: 1fr; min-height: 9;  }
    TargetPerceptionPanel { height: 1fr; min-height: 8;  }
    ControlServicePanel   { height: 1fr; min-height: 7;  }
    EventsPanel           { height: 1fr; min-height: 7;  }

    /* ── Talk to Planner internals ── */
    #prompt-history {
        height: 1fr;
        min-height: 3;
        border-bottom: solid #263050;
        padding: 0 0;
    }

    .history-item {
        padding: 0;
    }

    #planner-input {
        margin: 0;
        background: #111827;
        color: white;
        border: tall #3a4060;
    }

    /* ── Panic panel ── */
    #abort-row {
        align: center middle;
        height: 3;
        margin: 1 0 0 0;
    }

    #abort-btn {
        width: 30;
        height: 3;
        background: #c62828;
        color: white;
        text-style: bold;
        border: none;
    }

    #abort-btn:hover {
        background: #e53935;
    }

    #abort-btn:focus {
        background: #e53935;
        border: tall #ff8a80;
        text-style: bold reverse;
    }

    #abort-btn:disabled {
        background: #555555;
        color: #999999;
        text-style: bold;
    }

    #reset-btn {
        width: 16;
        height: 3;
        margin: 0 0 0 2;
        background: #1b5e20;
        color: white;
        text-style: bold;
        border: none;
    }

    #reset-btn:hover {
        background: #2e7d32;
    }

    #reset-btn:focus {
        background: #2e7d32;
        border: tall #69f0ae;
        text-style: bold reverse;
    }

    #reset-btn:disabled {
        background: #333333;
        color: #666666;
    }

    #abort-hint {
        content-align: center middle;
        color: #8a8a8a;
    }

    /* ── Legend modal ── */
    LegendScreen {
        align: center middle;
    }

    #legend-box {
        width: 62;
        height: auto;
        background: #141d30;
        border: solid #4fc3f7;
        padding: 1 0;
    }

    #legend-heading {
        color: #4fc3f7;
        text-style: bold;
        margin-bottom: 1;
    }

    #legend-close-hint {
        color: #8a8a8a;
    }

    /* ── Reasoning modal ── */
    ReasoningScreen {
        align: center middle;
    }

    #reasoning-box {
        width: 90;
        height: 36;
        background: #141d30;
        border: solid #4fc3f7;
        padding: 1 0;
        layout: vertical;
    }

    #reasoning-heading {
        color: #4fc3f7;
        text-style: bold;
        height: 1;
    }

    #reasoning-scroll {
        height: 1fr;
        padding: 0 2;
    }

    .reasoning-line {
        color: #b0bcd0;
    }

    #reasoning-close-hint {
        color: #8a8a8a;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+a", "emergency_abort", "ABORT", priority=True, show=False),
        Binding("ctrl+r", "reset_fuse", "RESET", priority=True, show=False),
        Binding("t", "focus_input", "type", show=False),
        Binding("enter", "send_message", "send", show=False),
        Binding("escape", "cancel_input", "cancel", show=False),
        Binding("r", "show_reasoning", "reasoning", show=False),
        Binding("y", "yank_reasoning", "yank", show=False),
        Binding("f", "toggle_feed", "feed", show=False),
        Binding("m", "toggle_mic", "mic", show=False),
        Binding("question_mark", "show_legend", "legend", show=False),
    ]

    _abort_cooldown: float = 0.0  # monotonic time of last abort
    _fuse_blown: bool = False  # safety fuse — blocks FSM/skill execution when True
    _last_reasoning: str = ""
    _event_queue: object | None = None  # asyncio.Queue[EventEnvelope] in live mode

    def __init__(
        self,
        runtime: object | None = None,
        agent: object | None = None,
        arm_id: str = "arm0",
        perception_svc: object | None = None,
        skill_runner_svc: object | None = None,
        run_logger: RunLogger | None = None,
        tracker_name: str = "",
        video_source: object | None = None,
        live_backend: object | None = None,
        cognitive_stack: object | None = None,
        live_agent: object | None = None,
        live_agent_audio: object | None = None,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._agent = agent
        self._arm_id = arm_id
        self._perception_svc = perception_svc
        self._skill_runner_svc = skill_runner_svc
        self._tracker_name = tracker_name
        self._video_source = video_source
        self._live_backend = live_backend
        self._cognitive_stack = cognitive_stack
        self._live_agent = live_agent  # LiveAgentClient | None
        self._live_agent_audio = live_agent_audio  # AudioComponents | None
        self._panel_data = {**_EMPTY_DATA, "arm_id": arm_id} if runtime is not None else _DATA
        # Accept an externally-created logger (shared with the VLM fn) or
        # create one automatically when running live.
        if run_logger is not None:
            self._run_logger: RunLogger | None = run_logger
        else:
            self._run_logger = RunLogger(_RUNS_DIR, arm_id) if runtime is not None else None
        self._last_operator_msg: str | None = None
        self._agent_queue: asyncio.Queue[str] | None = None
        self._feed_viewer: object | None = None  # FeedViewer instance (lazy import)
        self._cmd_route_queue: object | None = None  # for command routing
        self._pending_commands: dict[str, object] = {}  # intercepted commands by id

    def _stamp_lease(self, cmd: object) -> object:
        """Stamp epoch + lease_token on a CommandEnvelope from the active lease."""
        if self._cognitive_stack is None:
            return cmd
        from dataclasses import replace as _dc_replace

        lm = self._cognitive_stack.switchboard.lease_manager
        return _dc_replace(
            cmd,
            epoch=lm.current_epoch,
            lease_token=lm.current_token,
        )

    async def on_mount(self) -> None:
        self.call_after_refresh(self.set_focus, None)
        if self._runtime:
            self.set_interval(2.0, self._poll_system_panel)
            self._event_queue = self._runtime.bus.subscribe(self._arm_id, maxsize=0)  # type: ignore[union-attr]
            self.run_worker(self._listen_events(), name="event_listener")
        # Cognitive stack: start switchboard (non-blocking)
        if self._cognitive_stack is not None:
            sb = self._cognitive_stack.switchboard
            await sb.start()
        # Start agent processor
        if self._runtime and self._agent:
            self._agent_queue = asyncio.Queue()
            self.run_worker(self._agent_processor_loop(), name="agent_processor")
        # Start sim-mode services
        if self._skill_runner_svc is not None:
            await self._skill_runner_svc.start()  # type: ignore[union-attr]
            # Intercept submit_command to capture commands for routing
            self._orig_submit = self._runtime.submit_command  # type: ignore[union-attr]
            self._runtime.submit_command = self._intercepted_submit  # type: ignore[union-attr,assignment]
            # Start command routing worker
            self._cmd_route_queue = self._runtime.bus.subscribe(self._arm_id, maxsize=0)  # type: ignore[union-attr]
            self.run_worker(self._route_commands(), name="cmd_router")
        if self._perception_svc is not None:
            await self._perception_svc.start()  # type: ignore[union-attr]
        # Cloud wait + initial scene analysis in background so UI renders immediately
        self.run_worker(self._startup_cloud_and_perception(), name="startup_backend")
        # Live backend: voice command polling + audio panel updates
        if self._live_backend is not None:
            self.run_worker(self._poll_voice_commands(), name="voice_cmd_poll")
            self.set_interval(0.5, self._poll_audio_panel)
        # Live agent: connect WS + start audio capture/playback
        if self._live_agent is not None:
            self.run_worker(self._start_live_agent(), name="live_agent_start")
            self.set_interval(0.5, self._poll_audio_panel)

    async def _startup_cloud_and_perception(self) -> None:
        """Background: wait for cloud backend, fall back to local, then DESCRIBE_SCENE."""
        import logging as _logging
        import time as _time

        log = _logging.getLogger(__name__)

        if self._cognitive_stack is not None:
            sb = self._cognitive_stack.switchboard
            cfg = self._cognitive_stack.config
            from halo.cognitive.config import BackendType

            if cfg.active == BackendType.CLOUD:
                deadline = _time.monotonic() + cfg.startup_cloud_wait_s
                cloud_ready = False
                cloud_be = sb.active_backend
                while _time.monotonic() < deadline:
                    try:
                        cloud_ready = await cloud_be.health_check()
                        if cloud_ready:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
                if not cloud_ready:
                    log.warning(
                        "Cloud not ready within %.0fs, falling back to local",
                        cfg.startup_cloud_wait_s,
                    )
                    await sb.switch_to(BackendType.LOCAL, reason="cloud startup timeout")

        # Initial scene analysis
        if self._perception_svc is not None:
            await self._perception_svc.request_refresh(reason="startup")  # type: ignore[union-attr]

    async def on_unmount(self) -> None:
        if self._cognitive_stack is not None:
            try:
                await self._cognitive_stack.switchboard.stop()
            except Exception:
                pass
        if self._feed_viewer is not None:
            self._feed_viewer.stop()  # type: ignore[union-attr]
            self._feed_viewer = None
        if self._perception_svc is not None:
            await self._perception_svc.stop()  # type: ignore[union-attr]
        if self._skill_runner_svc is not None:
            await self._skill_runner_svc.stop()  # type: ignore[union-attr]
        if self._cmd_route_queue is not None and self._runtime:
            self._runtime.bus.unsubscribe(self._arm_id, self._cmd_route_queue)  # type: ignore[union-attr]
            self._cmd_route_queue = None
        self._pending_commands.clear()
        if self._live_backend is not None and hasattr(self._live_backend, "aclose"):
            try:
                await self._live_backend.aclose()  # type: ignore[union-attr]
            except Exception:
                pass
        if self._live_agent is not None:
            await self._stop_live_agent()
        if self._run_logger:
            self._run_logger.close()
        if self._runtime and self._event_queue is not None:
            self._runtime.bus.unsubscribe(self._arm_id, self._event_queue)  # type: ignore[union-attr]

    def compose(self) -> ComposeResult:
        d = self._panel_data
        yield TitleBar(arm_id=d["arm_id"])
        with Vertical(id="body"):
            with Horizontal(id="main-row"):
                with Vertical(id="left-col"):
                    with Vertical(id="left-info"):
                        yield PlannerPanel(data=d, id="planner-panel")
                        yield TargetPerceptionPanel(data=d, tracker_name=self._tracker_name, id="perception-panel")
                    yield TalkPanel(data=d, id="talk-panel")
                    audio_panel = AudioPanel(id="audio-panel")
                    if self._live_backend is not None or self._live_agent is not None:
                        audio_panel.add_class("visible")
                    yield audio_panel
                with Vertical(id="right-col"):
                    with Vertical(id="right-info"):
                        yield SystemPanel(data=d, id="system-panel")
                        yield ServosPanel(data=d, id="servos-panel")
                        yield ControlServicePanel(data=d, id="control-panel")
                        yield EventsPanel(data=d, id="events-panel")
                    yield PanicPanel(id="panic-panel")
            yield HintBar()

    # ── Event handlers ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button
        if btn.id == "abort-btn":
            self.action_emergency_abort()
        elif btn.id == "reset-btn":
            self.action_reset_fuse()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "planner-input":
            await self.action_send_message()

    # ── Actions ──

    def action_emergency_abort(self) -> None:
        import time

        now = time.monotonic()
        if now - self._abort_cooldown < 3.0:
            return
        self._abort_cooldown = now
        self._fuse_blown = True
        self._pending_commands.clear()  # drop any in-flight commands accepted before abort
        self._last_operator_msg = None
        if self._agent_queue is not None:
            while not self._agent_queue.empty():
                try:
                    self._agent_queue.get_nowait()
                except Exception:
                    break
        self._update_fuse_display()
        self.notify(
            "FUSE BLOWN — all execution halted. Ctrl+R to reset.",
            severity="error",
            timeout=10,
            title="ABORT",
        )
        if self._runtime:
            self.run_worker(self._do_abort(), name="abort_worker")

    def action_reset_fuse(self) -> None:
        if not self._fuse_blown:
            self.notify("Fuse is not blown", severity="warning", timeout=3)
            return
        self._fuse_blown = False
        self._update_fuse_display()
        self.notify("Fuse reset — execution re-enabled", severity="information", timeout=5)

    def _update_fuse_display(self) -> None:
        from textual.css.query import NoMatches

        try:
            abort_btn = self.query_one("#abort-btn", Button)
            reset_btn = self.query_one("#reset-btn", Button)
            if self._fuse_blown:
                abort_btn.disabled = True
                abort_btn.label = "FUSE BLOWN"
                reset_btn.disabled = False
            else:
                abort_btn.disabled = False
                abort_btn.label = "ABORT NOW!"
                reset_btn.disabled = True
        except NoMatches:
            pass

    async def action_send_message(self) -> None:
        from datetime import datetime

        inp = self.query_one("#planner-input", Input)
        msg = inp.value.strip()
        if not msg:
            self.set_focus(None)
            return
        inp.value = ""
        # Append user message to history
        ts = datetime.now().strftime("%H:%M:%S")
        t = Text()
        t.append(ts, style="grey62")
        t.append("  ")
        t.append(msg, style="#b0bcd0")
        history = self.query_one("#prompt-history", VerticalScroll)
        await history.mount(Static(t, classes="history-item"))
        history.scroll_end(animate=False)
        # Queue agent call if live
        if self._agent_queue is not None:
            self._last_operator_msg = msg
            if hasattr(self._agent, "reset_loop_state"):
                self._agent.reset_loop_state()
            self._agent_queue.put_nowait(msg)
        self.set_focus(None)

    def action_cancel_input(self) -> None:
        inp = self.query_one("#planner-input", Input)
        inp.value = ""
        self.set_focus(None)

    def action_focus_input(self) -> None:
        self.query_one("#planner-input", Input).focus()

    def action_show_reasoning(self) -> None:
        self.push_screen(ReasoningScreen(self._last_reasoning), callback=lambda _: self.set_focus(None))

    def action_yank_reasoning(self) -> None:
        if not self._last_reasoning:
            self.notify("No reasoning to copy yet", severity="warning", timeout=3)
            return
        if _copy_text(self._last_reasoning):
            self.notify("Reasoning copied to clipboard", timeout=3)
        else:
            self.notify("Copy failed — check clipboard tool", severity="warning", timeout=3)

    def action_toggle_feed(self) -> None:
        if not self._runtime:
            self.notify("Feed viewer requires live mode", severity="warning", timeout=3)
            return

        # Already running — stop it
        if self._feed_viewer is not None and self._feed_viewer.is_running:  # type: ignore[union-attr]
            self._feed_viewer.stop()  # type: ignore[union-attr]
            self._feed_viewer = None
            self.notify("Feed viewer closed", timeout=3)
            return

        # Lazy import so the viewer extra is truly optional
        try:
            from halo.tui.feed_viewer import FeedViewer
        except ImportError:
            self.notify("opencv-python not installed — run: uv sync --extra viewer", severity="error", timeout=5)
            return

        viewer = FeedViewer(
            store=self._runtime.store,  # type: ignore[union-attr]
            arm_id=self._arm_id,
            video_source=self._video_source,
            skill_runner_svc=self._skill_runner_svc,
        )
        if viewer.start():
            self._feed_viewer = viewer
            self.notify("Feed viewer opened (Q/Esc in window to close)", timeout=3)
        else:
            self.notify("Cannot open feed viewer — headless OpenCV build?", severity="error", timeout=5)

    def action_show_legend(self) -> None:
        self.push_screen(LegendScreen(), callback=lambda _: self.set_focus(None))

    def action_toggle_mic(self) -> None:
        # Live agent audio path
        if self._live_agent_audio is not None:
            audio = self._live_agent_audio
            if audio.available and audio.capture is not None:  # type: ignore[union-attr]
                audio.capture.muted = not audio.capture.muted  # type: ignore[union-attr]
                status = "muted" if audio.capture.muted else "listening"  # type: ignore[union-attr]
                self.notify(f"Mic {status}", timeout=2)
                return
        # Legacy live backend audio path
        if self._live_backend is None:
            self.notify("Mic toggle requires live agent or live backend", severity="warning", timeout=3)
            return
        session_state = getattr(self._live_backend, "session_state", None)
        if session_state is None:
            return
        session = getattr(self._live_backend, "_session", None)
        if session is None:
            return
        capture = getattr(session, "_audio_capture", None)
        if capture is not None and hasattr(capture, "muted"):
            capture.muted = not capture.muted
            status = "muted" if capture.muted else "listening"
            self.notify(f"Mic {status}", timeout=2)

    # ── Live workers ──

    async def _poll_system_panel(self) -> None:
        """Refresh all live panels from the latest snapshot every 2 s."""
        try:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)  # type: ignore[union-attr]
            services = _derive_services(snap)
            panel_data = _snap_to_panel_data(snap, self._panel_data)
            backend = ""
            if self._cognitive_stack is not None:
                backend = self._cognitive_stack.switchboard.active_type.value
            self.query_one("#system-panel", SystemPanel).refresh_live(services, backend=backend)
            self.query_one("#planner-panel", PlannerPanel).refresh_live(panel_data)
            inp = self.query_one("#planner-input", Input)
            typing_active = self.focused is inp and bool(inp.value)
            if not typing_active:
                self.query_one("#perception-panel", TargetPerceptionPanel).refresh_live(panel_data)
            # Update joints panel from MuJocoVideoSource if available
            self._update_joints_panel()
        except Exception:
            pass

    # SO-101 6-DOF joint names (5 arm + 1 gripper).
    _SO101_JOINT_NAMES = ("shldr_pan", "shldr_lift", "elbow", "wrist_flex", "wrist_roll", "gripper")

    def _update_joints_panel(self) -> None:
        """Build servos list from MuJocoVideoSource qpos/qvel and refresh the panel."""
        if self._video_source is None:
            return
        qpos = getattr(self._video_source, "latest_qpos", None)
        qvel = getattr(self._video_source, "latest_qvel", None)
        if qpos is None:
            return
        n = min(len(self._SO101_JOINT_NAMES), len(qpos))
        servos = []
        for i in range(n):
            jid = f"J{i + 1}"
            vel = float(qvel[i]) if qvel is not None and i < len(qvel) else 0.0
            servos.append((jid, self._SO101_JOINT_NAMES[i], "OK", float(qpos[i]), vel))
        self.query_one("#servos-panel", ServosPanel).refresh_live(servos)

    async def _poll_voice_commands(self) -> None:
        """Poll live backend for voice-triggered commands and submit them."""
        try:
            while True:
                await asyncio.sleep(0.5)
                if self._live_backend is None or self._runtime is None:
                    continue
                # Gate voice draining by active leader — skip if cloud is not active
                if self._cognitive_stack is not None:
                    from halo.cognitive.config import BackendType

                    if self._cognitive_stack.switchboard.active_type != BackendType.CLOUD:
                        continue
                drain_fn = getattr(self._live_backend, "drain_pending_commands", None)
                if drain_fn is None:
                    continue
                commands = drain_fn()
                if not commands:
                    continue
                from dataclasses import replace as _dc_replace

                for cmd in commands:
                    cmd = _dc_replace(cmd, precondition_snapshot_id=None)
                    cmd = self._stamp_lease(cmd)
                    await self._runtime.submit_command(cmd)  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass

    async def _poll_audio_panel(self) -> None:
        """Refresh the AudioPanel from live agent or live backend session state."""
        try:
            panel = self.query_one("#audio-panel", AudioPanel)
        except Exception:
            return

        # Live agent path
        if self._live_agent is not None:
            client_state = self._live_agent.state  # type: ignore[union-attr]
            audio = self._live_agent_audio
            capture = audio.capture if audio and audio.available else None  # type: ignore[union-attr]
            playback = audio.playback if audio and audio.available else None  # type: ignore[union-attr]

            if not client_state.connected:
                mic_status = "Disconnected"
            elif capture is not None:
                mic_status = "Muted" if capture.muted else "Listening"
            else:
                mic_status = "Text-only"

            if not client_state.connected:
                speaker_status = "Disconnected"
            elif playback is not None and playback.state.output_level > 0.01:
                speaker_status = "Speaking"
            else:
                speaker_status = "Silent"

            # Flash "Interrupted" for ~2s after barge-in
            if time.monotonic() - client_state.last_interrupted_ts < 2.0:
                speaker_status = "Interrupted"

            mic_level = capture.state.input_level if capture is not None else 0.0
            spk_level = playback.state.output_level if playback is not None else 0.0

            panel.refresh_live(
                mic_status=mic_status,
                speaker_status=speaker_status,
                transcript_in=client_state.last_transcription_in,
                transcript_out=client_state.last_transcription_out,
                mic_level=mic_level,
                speaker_level=spk_level,
                transcript_in_finished=client_state.transcription_in_finished,
                transcript_out_finished=client_state.transcription_out_finished,
            )

            return

        # Legacy live backend path
        if self._live_backend is None:
            return
        session_state = getattr(self._live_backend, "session_state", None)
        if session_state is None:
            return
        session = getattr(self._live_backend, "_session", None)
        capture = getattr(session, "_audio_capture", None) if session else None
        if capture is not None and hasattr(capture, "muted"):
            mic_status = "Muted" if capture.muted else "Listening"
        elif session_state.connected:
            mic_status = "Text-only"
        else:
            mic_status = "Disconnected"

        speaker_status = "Speaking" if getattr(session_state, "turn_active", False) else "Silent"
        if not session_state.connected:
            speaker_status = "Disconnected"

        panel.refresh_live(
            mic_status=mic_status,
            speaker_status=speaker_status,
            transcript_in=getattr(session_state, "last_transcription_in", ""),
            transcript_out=getattr(session_state, "last_transcription_out", ""),
            transcript_in_finished=getattr(session_state, "transcription_in_finished", True),
            transcript_out_finished=getattr(session_state, "transcription_out_finished", True),
        )

    async def _handle_live_tool_call(self, call_id: str, name: str, args: dict) -> None:
        """Execute a proxy tool call from the Live Agent and send the result back."""
        import logging as _log

        _logger = _log.getLogger(__name__)
        try:
            if name == "describe_scene":
                if self._runtime is not None:
                    import time as _time
                    import uuid as _uuid

                    from halo.contracts.commands import CommandEnvelope, DescribeScenePayload
                    from halo.contracts.enums import CommandType

                    reason = args.get("reason", "")
                    cmd = CommandEnvelope(
                        command_id=str(_uuid.uuid4()),
                        arm_id=self._arm_id,
                        issued_at_ms=int(_time.time() * 1000),
                        type=CommandType.DESCRIBE_SCENE,
                        payload=DescribeScenePayload(reason=reason),
                        precondition_snapshot_id=None,
                    )
                    cmd = self._stamp_lease(cmd)  # type: ignore[assignment]
                    ack = await self._runtime.submit_command(cmd)
                    result = f"Scene analysis requested (status={ack.status.value})."
                else:
                    result = "Runtime not available."

            elif name == "submit_user_intent":
                intent = args.get("intent", "")
                if self._agent_queue is not None and intent:
                    self._last_operator_msg = intent
                    if hasattr(self._agent, "reset_loop_state"):
                        self._agent.reset_loop_state()
                    self._agent_queue.put_nowait(intent)
                    result = f"Intent forwarded to planner: {intent}"
                else:
                    result = "Planner agent not available."

            elif name == "abort":
                if self._runtime is not None:
                    snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)
                    if snap.skill is not None:
                        import time as _time
                        import uuid as _uuid

                        from halo.contracts.commands import AbortSkillPayload, CommandEnvelope
                        from halo.contracts.enums import CommandType

                        cmd = CommandEnvelope(
                            command_id=str(_uuid.uuid4()),
                            arm_id=self._arm_id,
                            issued_at_ms=int(_time.time() * 1000),
                            type=CommandType.ABORT_SKILL,
                            payload=AbortSkillPayload(
                                skill_run_id=snap.skill.skill_run_id,
                                reason="operator_voice_abort",
                            ),
                        )
                        cmd = self._stamp_lease(cmd)  # type: ignore[assignment]
                        if self._skill_runner_svc is not None:
                            self._skill_runner_svc._queue.clear()  # type: ignore[union-attr]
                        # Clear planner state so it doesn't re-plan the aborted task
                        self._last_operator_msg = None
                        if self._agent_queue is not None:
                            while not self._agent_queue.empty():
                                try:
                                    self._agent_queue.get_nowait()
                                except Exception:
                                    break
                        ack = await self._runtime.submit_command(cmd)
                        result = f"Aborted current skill and cleared queue (status={ack.status.value})."
                    else:
                        result = "No active skill to abort."
                else:
                    result = "Runtime not available."
            else:
                result = f"Unknown tool: {name}"

            _logger.info("Live agent tool %s → %s", name, result[:120])
        except Exception:
            _logger.exception("Live agent tool %s failed", name)
            result = f"Tool {name} failed with an error."

        await self._live_agent.send_tool_result(call_id, result)  # type: ignore[union-attr]

    async def _start_live_agent(self) -> None:
        """Connect LiveAgentClient and start audio capture/playback."""
        try:
            # Wire proxy-tool handler
            self._live_agent._on_tool_call = self._handle_live_tool_call  # type: ignore[union-attr]
            await self._live_agent.connect()  # type: ignore[union-attr]
            audio = self._live_agent_audio
            if audio and audio.available:  # type: ignore[union-attr]
                if audio.capture:  # type: ignore[union-attr]
                    audio.capture.muted = True  # start muted
                    audio.capture.start()  # type: ignore[union-attr]
                if audio.playback:  # type: ignore[union-attr]
                    audio.playback.start()  # type: ignore[union-attr]
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Failed to start live agent")

    async def _stop_live_agent(self) -> None:
        """Stop audio and disconnect LiveAgentClient."""
        audio = self._live_agent_audio
        if audio and audio.available:  # type: ignore[union-attr]
            if audio.capture:  # type: ignore[union-attr]
                audio.capture.stop()  # type: ignore[union-attr]
            if audio.playback:  # type: ignore[union-attr]
                audio.playback.stop()  # type: ignore[union-attr]
        if self._live_agent is not None:
            try:
                await self._live_agent.aclose()  # type: ignore[union-attr]
            except Exception:
                pass

    def _with_task_context(self, system_msg: str) -> str:
        """Append the last operator instruction so the agent keeps task context."""
        if self._last_operator_msg:
            return f"{system_msg}\nOperator task (act on it now): {self._last_operator_msg}"
        return system_msg

    _AGENT_WAKE_EVENTS = frozenset(
        {
            "SKILL_SUCCEEDED",
            "SKILL_FAILED",
            "SAFETY_REFLEX_TRIGGERED",
            "PERCEPTION_FAILURE",
            "SCENE_DESCRIBED",
            "TARGET_ACQUIRED",
            "COMMAND_REJECTED",
        }
    )

    _LIVE_AGENT_NARRATION_EVENTS = frozenset(
        {
            "SKILL_STARTED",
            "SKILL_SUCCEEDED",
            "SKILL_FAILED",
            "SAFETY_REFLEX_TRIGGERED",
            "COMMAND_REJECTED",
        }
    )

    async def _listen_events(self) -> None:
        """Forward EventBus events to EventsPanel, log to events.jsonl, and wake agent on urgent events.

        The agent reads actual event data from the snapshot's recent_events,
        not from the TUI. The TUI only sends a wake signal.
        """
        events_panel = self.query_one("#events-panel", EventsPanel)
        try:
            while True:
                evt = await self._event_queue.get()  # type: ignore[union-attr]
                # Persist to events.jsonl
                if self._run_logger:
                    try:
                        self._run_logger.log_event(evt)
                    except Exception as _log_err:
                        import logging as _lg

                        _lg.getLogger("halo.tui").debug(
                            "log_event error for %s: %s",
                            getattr(evt, "type", "?"),
                            _log_err,
                        )
                    # Also write compaction summary to run.jsonl
                    evt_type_str = getattr(evt.type, "value", str(evt.type))  # type: ignore[union-attr]
                    if evt_type_str == "SESSION_COMPACTED":
                        try:
                            data = getattr(evt, "data", {}) or {}
                            self._run_logger.log_compaction(
                                compacted_count=data.get("compacted_count", 0),
                                retained_count=data.get("retained_count", 0),
                                summary=data.get("summary", ""),
                                backend=data.get("backend", ""),
                            )
                        except Exception:
                            pass
                # Skip high-frequency FSM events from TUI panel (still logged above)
                evt_type_name = getattr(evt.type, "value", str(evt.type))  # type: ignore[union-attr]
                if evt_type_name not in ("PHASE_ENTER", "PHASE_EXIT"):
                    try:
                        await events_panel.append_event(evt)
                    except Exception:
                        pass  # DOM error must not kill the listener
                # Forward notable events to Live Agent for narration
                if self._live_agent is not None and evt_type_name in self._LIVE_AGENT_NARRATION_EVENTS:
                    summary = _format_event(evt)
                    asyncio.create_task(
                        self._live_agent.send_monitor_update("event", summary)  # type: ignore[union-attr]
                    )
                # Forward full scene description to Live Agent
                if self._live_agent is not None and evt_type_name == "SCENE_DESCRIBED":
                    evt_data = getattr(evt, "data", {}) or {}
                    scene_text = evt_data.get("scene", "")
                    if scene_text:
                        asyncio.create_task(
                            self._live_agent.send_monitor_update("scene_description", scene_text)  # type: ignore[union-attr]
                        )
                # Wake the agent — it reads event details from the snapshot
                # Skip waking on operator-initiated aborts to prevent replanning
                evt_data = getattr(evt, "data", {}) or {}
                is_operator_abort = evt_type_name == "SKILL_FAILED" and evt_data.get("reason") == "abort"
                if self._agent_queue is not None and evt_type_name in self._AGENT_WAKE_EVENTS and not is_operator_abort:
                    # Include event detail so the planner LLM can disambiguate
                    detail_parts = [f"[event: {evt_type_name}"]
                    if evt_data.get("skill_name"):
                        detail_parts.append(f" skill={evt_data['skill_name']}")
                    if evt_data.get("target_handle"):
                        detail_parts.append(f" target={evt_data['target_handle']}")
                    if evt_data.get("reason_code"):
                        detail_parts.append(f" reason={evt_data['reason_code']}")
                    wake_msg = "".join(detail_parts) + "]"
                    self._agent_queue.put_nowait(self._with_task_context(wake_msg))
        except asyncio.CancelledError:
            pass

    async def _agent_processor_loop(self) -> None:
        """Single-threaded agent call processor. Reads from _agent_queue,
        batches messages that arrived during inference, and calls the agent."""
        assert self._agent_queue is not None
        try:
            while True:
                # Wait for the first message
                msg = await self._agent_queue.get()
                # Brief yield so burst events can queue up
                await asyncio.sleep(0.05)
                # Drain any additional messages into one batch
                while not self._agent_queue.empty():
                    try:
                        extra = self._agent_queue.get_nowait()
                        msg = msg + "\n" + extra
                    except asyncio.QueueEmpty:
                        break
                await self._do_agent_call(msg)
        except asyncio.CancelledError:
            pass

    async def _do_agent_call(self, msg: str | None) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        history = self.query_one("#prompt-history", VerticalScroll)

        # Mount "[thinking…]" placeholder
        thinking_text = Text()
        thinking_text.append(ts, style="grey62")
        thinking_text.append("  ")
        thinking_text.append("[thinking…]", style="italic #9e9e9e")
        thinking_widget = Static(thinking_text, classes="history-item")
        await history.mount(thinking_widget)
        history.scroll_end(animate=False)

        try:
            import time as _time

            from halo.services.planner_service.snapshot_serializer import snapshot_to_dict

            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)  # type: ignore[union-attr]
            # Inject fuse context so planner knows not to issue commands
            if self._fuse_blown and msg is not None:
                fuse_ctx = "[SAFETY FUSE BLOWN — all commands blocked. Operator must reset before any skill can run.]"
                msg = fuse_ctx + "\n" + msg
            # Event wake signals (e.g. "[event: COMMAND_REJECTED]\nOperator task: ...")
            # are NOT operator commands — pass them as context only so they don't
            # reset loop detection.  Real operator messages never start with "[".
            # When batching combines events with a real operator message, extract
            # the operator part so it's properly framed as a NEW OPERATOR TASK.
            is_event_wake = msg is not None and msg.startswith("[")
            operator_cmd: str | None = None
            if is_event_wake and msg is not None:
                # Check if a real operator message got batched after event lines
                lines = msg.split("\n")
                non_event_lines = [
                    ln for ln in lines if not ln.startswith("[") and not ln.startswith("Operator task (act on it now):")
                ]
                if non_event_lines:
                    operator_cmd = "\n".join(non_event_lines).strip() or None
            elif not is_event_wake:
                operator_cmd = msg
            _t0 = _time.monotonic()
            commands = await self._agent.decide(snap, operator_cmd=operator_cmd)  # type: ignore[union-attr]
            inference_ms = int((_time.monotonic() - _t0) * 1000)
            reasoning = getattr(self._agent, "last_reasoning", "") or ""
            self._last_reasoning = reasoning

            # Submit commands — clear stale precondition_snapshot_id because
            # concurrent service ticks create new snapshots during LLM inference.
            from dataclasses import replace as _dc_replace

            acks = []
            for cmd in commands:
                cmd = _dc_replace(cmd, precondition_snapshot_id=None)
                ack = await self._runtime.submit_command(cmd)  # type: ignore[union-attr]
                acks.append((cmd, ack))

            # Log interaction
            if self._run_logger:
                self._run_logger.log_interaction(
                    arm_id=self._arm_id,
                    operator_msg=msg,
                    snapshot=snapshot_to_dict(snap),
                    commands=[{"id": c.command_id, "str": _format_cmd(c)} for c, _ in acks],
                    acks=[{"id": a.command_id, "status": a.status.value} for _, a in acks],
                    reasoning=reasoning,
                    inference_ms=inference_ms,
                    model=getattr(self._agent, "model_name", "") or "",
                    token_usage=getattr(self._agent, "last_token_usage", None) or {},
                )

            # Clear stale operator task when agent is idle with no commands —
            # prevents old task text from being re-injected into subsequent ticks.
            if not commands and snap.skill is None:
                self._last_operator_msg = None

            # Update thinking widget
            from halo.contracts.enums import CommandAckStatus

            accepted = [(c, a) for c, a in acks if a.status == CommandAckStatus.ACCEPTED]
            rejected = [(c, a) for c, a in acks if a.status != CommandAckStatus.ACCEPTED]
            result_text = Text()
            result_text.append(ts, style="grey62")
            result_text.append("  ")
            if rejected:
                descs = ", ".join(f"{_format_cmd(c)} [{a.reason or a.status.value}]" for c, a in rejected)
                result_text.append(f"✗ Rejected {len(rejected)} command(s): {descs}", style="bold red")
                if accepted:
                    result_text.append(" | ", style="grey62")
            if accepted:
                descs = ", ".join(_format_cmd(c) for c, _ in accepted)
                result_text.append(f"▶ Queued {len(accepted)} command(s): {descs}", style="bright_green")
            elif not rejected:
                snippet = (reasoning[:80] + "…") if len(reasoning) > 80 else reasoning
                result_text.append("▶ No commands", style="grey62")
                if snippet:
                    result_text.append(f" — {snippet}", style="#9e9e9e")
                result_text.append("  [R]", style="#4fc3f7")
            result_text.append(f"  ({inference_ms} ms)", style="#9e9e9e")
            thinking_widget.update(result_text)
            history.scroll_end(animate=False)

            # Stream planner output to Live Agent so it can narrate
            if self._live_agent is not None and (accepted or rejected):
                parts = []
                if accepted:
                    parts.append("Planner issued: " + ", ".join(_format_cmd(c) for c, _ in accepted))
                if rejected:
                    parts.append("Rejected: " + ", ".join(f"{_format_cmd(c)} ({a.reason})" for c, a in rejected))
                if reasoning:
                    parts.append(f"Reasoning: {reasoning[:200]}")
                planner_summary = ". ".join(parts)
                asyncio.create_task(self._live_agent.send_monitor_update("planner_decision", planner_summary))  # type: ignore[union-attr]

        except Exception as exc:
            if self._run_logger:
                self._run_logger.log_interaction(
                    arm_id=self._arm_id,
                    operator_msg=msg,
                    snapshot=None,
                    commands=[],
                    acks=[],
                    error=str(exc),
                    model=getattr(self._agent, "model_name", "") or "",
                )
            err_text = Text()
            err_text.append(ts, style="grey62")
            err_text.append("  ")
            err_text.append(f"✗ {exc}", style="bold red")
            thinking_widget.update(err_text)
            history.scroll_end(animate=False)
            self.notify(str(exc), severity="error", title="Agent error")

    # ── Command routing (teacher mode) ──

    async def _intercepted_submit(self, cmd) -> object:
        """Intercept submit_command to capture commands for routing to SkillRunner."""
        from halo.contracts.enums import CommandType

        if self._fuse_blown and cmd.type not in (CommandType.ABORT_SKILL, CommandType.DESCRIBE_SCENE):
            from halo.contracts.commands import CommandAck
            from halo.contracts.enums import CommandAckStatus

            self.notify("FUSE BLOWN — command blocked. Ctrl+R to reset.", severity="error", timeout=5)
            return CommandAck(command_id=cmd.command_id, status=CommandAckStatus.REJECTED, reason="fuse_blown")
        self._pending_commands[cmd.command_id] = cmd
        return await self._orig_submit(cmd)  # type: ignore[union-attr]

    async def _route_commands(self) -> None:
        """Route accepted START_SKILL/ABORT_SKILL commands to SkillRunnerService.

        Mirrors HeadlessRunner._route_commands() logic.
        """
        from halo.contracts.commands import StartSkillPayload
        from halo.contracts.enums import CommandType
        from halo.contracts.events import EventType

        try:
            while True:
                event = await self._cmd_route_queue.get()  # type: ignore[union-attr]

                if event.type == EventType.COMMAND_REJECTED:
                    cmd_id = event.data.get("command_id")
                    if cmd_id:
                        self._pending_commands.pop(cmd_id, None)
                    continue

                if event.type != EventType.COMMAND_ACCEPTED:
                    continue

                cmd_id = event.data.get("command_id")
                if not cmd_id:
                    continue
                cmd = self._pending_commands.pop(cmd_id, None)
                if cmd is None:
                    continue

                if cmd.type == CommandType.START_SKILL:
                    if self._fuse_blown:
                        continue  # drop — fuse blown after command was accepted
                    payload = cmd.payload
                    assert isinstance(payload, StartSkillPayload)
                    await self._skill_runner_svc.start_skill(  # type: ignore[union-attr]
                        skill_name=payload.skill_name,
                        skill_run_id=f"run-{cmd.command_id[:8]}",
                        target_handle=payload.target_handle,
                    )
                elif cmd.type == CommandType.ABORT_SKILL:
                    await self._skill_runner_svc.abort_skill()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass

    async def _do_abort(self) -> None:
        import time
        from uuid import uuid4

        from halo.contracts.commands import AbortSkillPayload, CommandEnvelope, CommandType

        snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)  # type: ignore[union-attr]
        if snap.skill is None:
            self.notify("No active skill to abort", severity="warning")
            return

        cmd = CommandEnvelope(
            command_id=str(uuid4()),
            arm_id=self._arm_id,
            issued_at_ms=int(time.time() * 1000),
            type=CommandType.ABORT_SKILL,
            payload=AbortSkillPayload(
                skill_run_id=snap.skill.skill_run_id,
                reason="operator_abort",
            ),
        )
        cmd = self._stamp_lease(cmd)
        # Clear skill queue directly — emergency abort stops everything
        if self._skill_runner_svc is not None:
            self._skill_runner_svc._queue.clear()  # type: ignore[union-attr]
        ack = await self._runtime.submit_command(cmd)  # type: ignore[union-attr]
        self.notify(
            f"ABORT_SKILL → {ack.status.value}",
            severity="information" if ack.status.value == "ACCEPTED" else "warning",
            timeout=5,
        )


def _take_screenshot(path: str = "halo_tui.svg") -> None:
    """Run headlessly, render one frame, save SVG screenshot, exit."""
    import asyncio

    async def _run() -> None:
        app = HALOApp()
        async with app.run_test(headless=True, size=(220, 60)) as pilot:
            await pilot.pause(0.3)  # let layout settle
            svg = app.export_screenshot()
        with open(path, "w") as f:
            f.write(svg)
        print(f"Screenshot saved: {path}")

    asyncio.run(_run())


def _run_live(args: list[str]) -> None:
    """Start the TUI wired to a real HALORuntime + PlannerAgent."""

    from halo.runtime.runtime import HALORuntime
    from halo.services.target_perception_service.service import TargetPerceptionService
    from halo.services.target_perception_service.tracker_fn import get_tracker_name, make_tracker_factory_fn
    from halo.services.target_perception_service.video_source import VideoSource

    # Parse CLI args — model defaults come from config dataclasses, not here
    arm_id = "arm0"
    model: str | None = None
    vlm_model: str | None = None
    base_url = "http://localhost:11434"
    source_type = "videoloop"
    cloud_url = ""
    sa_key_file: str | None = None
    sa_email: str | None = None

    for i, arg in enumerate(args):
        if arg == "--arm" and i + 1 < len(args):
            arm_id = args[i + 1]
        elif arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
        elif arg == "--vlm-model" and i + 1 < len(args):
            vlm_model = args[i + 1]
        elif arg == "--base-url" and i + 1 < len(args):
            base_url = args[i + 1]
        elif arg == "--source" and i + 1 < len(args):
            source_type = args[i + 1]
        elif arg == "--backend" and i + 1 < len(args):
            if args[i + 1] != "local":
                msg = "--backend is deprecated; use --cloud-url to connect to the cloud service"
                raise SystemExit(msg)
        elif arg == "--cloud-url" and i + 1 < len(args):
            cloud_url = args[i + 1]
        elif arg == "--sa-key-file" and i + 1 < len(args):
            sa_key_file = args[i + 1]
        elif arg == "--sa-email" and i + 1 < len(args):
            sa_email = args[i + 1]

    live_agent_enabled = "--live-agent" in args

    run_logger = RunLogger(_RUNS_DIR, arm_id)

    live_backend = None
    cognitive_stack = None

    # Build cognitive stack — all paths go through Switchboard
    from halo.cognitive import make_cognitive_stack
    from halo.cognitive.config import BackendType, CognitiveConfig, LocalConfig
    from halo.cognitive.lease import LeaseManager

    # Build local config with CLI overrides (or defaults from LocalConfig)
    local_kwargs: dict = {"base_url": base_url}
    if model:
        local_kwargs["planner_model"] = model
    if vlm_model:
        local_kwargs["vlm_model"] = vlm_model
    local_cfg = LocalConfig(**local_kwargs)

    if cloud_url:
        from halo.cognitive.config import RemoteCloudConfig
        from halo.cognitive.remote_backend import RemoteCognitiveBackend

        remote = RemoteCognitiveBackend(
            RemoteCloudConfig(service_url=cloud_url, sa_key_file=sa_key_file, sa_email=sa_email),
            arm_id=arm_id,
            run_logger=run_logger,
        )
        lease_mgr = LeaseManager()
        runtime = HALORuntime(lease_manager=lease_mgr)
        runtime.register_arm(arm_id)
        stack = make_cognitive_stack(
            config=CognitiveConfig(active=BackendType.CLOUD, local=local_cfg, enable_failover=True),
            cloud_backend=remote,
            lease_mgr=lease_mgr,
            bus=runtime.bus,
            snapshot_fn=runtime.get_latest_runtime_snapshot,
            run_logger=run_logger,
            arm_id=arm_id,
        )
    else:
        runtime = HALORuntime()
        runtime.register_arm(arm_id)
        stack = make_cognitive_stack(
            config=CognitiveConfig(active=BackendType.LOCAL, local=local_cfg, enable_failover=False),
            bus=runtime.bus,
            snapshot_fn=runtime.get_latest_runtime_snapshot,
            run_logger=run_logger,
            arm_id=arm_id,
        )

    agent = stack.switchboard
    vlm_fn = stack.switchboard.vlm_scene
    cognitive_stack = stack

    if source_type == "mujoco":
        from halo.bridge.config import SimBridgeConfig
        from halo.bridge.sim_source import SimSource

        sim_cfg = SimBridgeConfig(managed=True, log_file=str(run_logger.run_dir / "sim.log"))
        video_source = SimSource(config=sim_cfg)
    else:
        video_source = VideoSource()
    video_source.start()
    capture_fn = video_source.make_capture_fn(arm_id)
    tracker_factory_fn = make_tracker_factory_fn()

    perception_svc = TargetPerceptionService(
        arm_id=arm_id,
        runtime=runtime,
        vlm_fn=vlm_fn,
        capture_fn=capture_fn,
        tracker_factory_fn=tracker_factory_fn,
        run_logger=run_logger,
    )

    # Sim-mode services (only when MuJoCo source is active)
    skill_runner_svc = None
    if source_type == "mujoco":
        import asyncio as _asyncio

        from halo.services.skill_runner_service.config import SkillRunnerConfig
        from halo.services.skill_runner_service.service import SkillRunnerService

        sim_client = video_source.client

        async def start_pick_fn(arm_id_: str, target_body: str) -> dict:
            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(None, sim_client.start_pick, target_body)

        async def start_place_fn(arm_id_: str, target_body: str, held_body: str) -> dict:
            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(None, sim_client.start_place, target_body, held_body)

        async def abort_pick_fn() -> dict:
            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(None, sim_client.abort_pick)

        def sim_phase_fn() -> tuple[int, bool, str | None]:
            return video_source.latest_phase_id, video_source.latest_done, video_source.latest_error

        skill_runner_svc = SkillRunnerService(
            arm_id=arm_id,
            runtime=runtime,
            config=SkillRunnerConfig(),
            start_pick_fn=start_pick_fn,
            start_place_fn=start_place_fn,
            abort_pick_fn=abort_pick_fn,
            sim_phase_fn=sim_phase_fn,
        )

    # Live Agent + audio
    live_agent = None
    live_agent_audio = None
    if live_agent_enabled and cloud_url:
        from halo.cognitive.audio_io import make_audio_components
        from halo.cognitive.live_agent_client import LiveAgentClient

        # Use IAM auth for https:// Cloud Run URLs, skip for local http://
        _la_auth_fn = None
        if cloud_url.startswith("https://"):
            from halo.cognitive.remote_backend import make_id_token_fn

            _la_auth_fn = make_id_token_fn(audience=cloud_url, sa_key_file=sa_key_file, sa_email=sa_email)

        live_agent = LiveAgentClient(url=cloud_url, arm_id=arm_id, auth_token_fn=_la_auth_fn)
        live_agent_audio = make_audio_components(on_audio=live_agent.send_audio)
        if live_agent_audio.available and live_agent_audio.playback:
            live_agent.set_audio_callbacks(
                on_audio_out=live_agent_audio.playback.enqueue,
                on_interrupt=live_agent_audio.playback.clear,
            )
    elif live_agent_enabled and not cloud_url:
        print("WARNING: --live-agent requires --cloud-url; live agent disabled")

    # Redirect all logging to the run log directory so warnings/errors
    # don't corrupt the Textual TUI display (which owns stdout/stderr).
    import logging as _logging
    import sys as _sys

    _log_file = run_logger.run_dir / "tui.log" if run_logger.run_dir else _RUNS_DIR / "tui.log"
    _file_handler = _logging.FileHandler(_log_file)
    _file_handler.setFormatter(_logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _logging.root.handlers = [_file_handler]
    _logging.root.setLevel(_logging.DEBUG)

    # Silence noisy third-party DEBUG loggers that flood the log with
    # per-frame audio chunks, HTTP connection details, etc.
    for _noisy in ("websockets", "httpcore", "httpx", "LiteLLM"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)

    # Strip StreamHandlers that libraries (litellm, httpx, google.*) attach
    # directly to their own loggers — those bypass root and write to the
    # terminal, corrupting the Textual TUI.
    for _name in list(_logging.Logger.manager.loggerDict):
        _lgr = _logging.getLogger(_name)
        for _h in list(_lgr.handlers):
            if isinstance(_h, _logging.StreamHandler) and not isinstance(_h, _logging.FileHandler):
                _lgr.removeHandler(_h)

    # Redirect stdout/stderr to the log file to catch stray print() calls
    # from third-party libraries (litellm debug prints, etc.).
    _log_stream = open(_log_file, "a")  # noqa: SIM115
    _sys.stdout = _log_stream  # type: ignore[assignment]
    _sys.stderr = _log_stream  # type: ignore[assignment]

    HALOApp(
        runtime=runtime,
        agent=agent,
        arm_id=arm_id,
        perception_svc=perception_svc,
        skill_runner_svc=skill_runner_svc,
        run_logger=run_logger,
        tracker_name=get_tracker_name(),
        video_source=video_source,
        live_backend=live_backend,
        cognitive_stack=cognitive_stack,
        live_agent=live_agent,
        live_agent_audio=live_agent_audio,
    ).run()

    # After TUI exits, suppress all output — stray log messages from asyncio
    # cleanup, health-check threads, etc. are terminal noise.
    import logging as _logging
    import os
    import sys

    _logging.disable(_logging.CRITICAL)
    _devnull = open(os.devnull, "w")  # noqa: SIM115
    sys.stdout = _devnull
    sys.stderr = _devnull

    video_source.stop()


def main() -> None:
    import sys

    args = sys.argv[1:]
    if "--screenshot" in args:
        idx = args.index("--screenshot")
        path = args[idx + 1] if idx + 1 < len(args) else "halo_tui.svg"
        _take_screenshot(path)
    elif "--live" in args:
        _run_live(args)
    else:
        HALOApp().run()


if __name__ == "__main__":
    main()
