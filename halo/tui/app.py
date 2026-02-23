"""HALO TUI — static mockup (v0).

Run with:
    uv run python -m halo.tui.app
    uv run python -m halo.tui.app --live
    uv run python -m halo.tui.app --live --arm arm0 --model llama3.2:3b --base-url http://localhost:11434
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static
from rich.text import Text

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
    perc_distance_m=0.09,
    perc_confidence=0.86,
    perc_obs_age_ms=45,
    perc_failure="OK",
    perc_hint_valid=True,
    # ── Control Service state ──
    ctrl_status="RUNNING",
    ctrl_safety="OK",
    ctrl_reflex=None,
    ctrl_action_xyz=(0.12, -0.05, 0.08),
    ctrl_gripper=0.0,
    servos=[
        ("J1", "base_yaw",    "OK", 42, 0.35),
        ("J2", "shoulder",    "OK", 45, 0.55),
        ("J3", "elbow",       "OK", 46, 0.55),
        ("J4", "wrist_pitch", "OK", 44, 0.50),
        ("J5", "wrist_yaw",   "OK", 42, 0.25),
        ("J6", "wrist_roll",  "OK", 41, 0.30),
    ],
    # Prompt history — shown newest-at-bottom inside the Talk panel
    prompt_history=[
        ("14:31:44", "What objects are visible on the table?"),
        ("14:31:50", "Retry the grasp once."),
        ("14:32:05", "Use cube-2 instead, then continue with the bin."),
    ],
    # Row-major order: [top-left, top-right, bottom-left, bottom-right]
    suggestions=[
        "Pause after this step.",
        "Back off 5 cm and try again.",
        "Retry the grasp once.",
        "Stop and describe the scene.",
    ],
    services=[
        ("TargetPerception", "TRACKING", "bright_green"),
        ("SkillRunner",      "RUNNING",  "bright_green"),
        ("ControlService",   "RUNNING",  "bright_green"),
        ("Safety",           "OK",       "yellow"),
    ],
    target_info=[
        ("Target",     "cube-1"),
        ("Distance",   "9 cm"),
        ("Confidence", "86%"),
    ],
    events=[
        ("14:32:08", "PHASE_ENTER PREGRASP_ALIGN"),
        ("14:32:08", "PERCEPTION_RECOVERED"),
        ("14:32:06", "SKILL_STARTED run-9"),
    ],
)

_EMPTY_DATA = dict(
    arm_id="arm0",
    skill_name=None, skill_run_id=None, skill_phase=None,
    act_status=None, act_buffer_ms=None, act_buffer_low=False,
    outcome_state=None, outcome_reason=None, elapsed_ms=None,
    perc_tracking=None, perc_target=None, perc_distance_m=None,
    perc_confidence=None, perc_obs_age_ms=None, perc_failure=None, perc_hint_valid=None,
    ctrl_status=None, ctrl_safety=None, ctrl_reflex=None,
    ctrl_action_xyz=None, ctrl_gripper=None,
    servos=[
        ("J1", "base_yaw",    "NC", None, None),
        ("J2", "shoulder",    "NC", None, None),
        ("J3", "elbow",       "NC", None, None),
        ("J4", "wrist_pitch", "NC", None, None),
        ("J5", "wrist_yaw",   "NC", None, None),
        ("J6", "wrist_roll",  "NC", None, None),
    ],
    prompt_history=[],
    suggestions=_DATA["suggestions"],  # keep as useful operator shortcuts
    services=[
        ("TargetPerception", "—", "#9e9e9e"),
        ("SkillRunner",      "—", "#9e9e9e"),
        ("ControlService",   "—", "#9e9e9e"),
        ("Safety",           "—", "#9e9e9e"),
    ],
    target_info=[],
    events=[],
)

_LEGEND = [
    ("Tab / Shift+Tab", "Navigate between input and buttons"),
    ("T",               "Focus the message input directly"),
    ("Enter",       "Send message to planner"),
    ("Esc",         "Clear input and return to monitoring mode"),
    ("R",           "Show full planner reasoning (Ctrl+Y to copy inside)"),
    ("Y",           "Yank — copy last reasoning to clipboard instantly"),
    ("A",           "Emergency abort — always fires (even while typing)"),
    ("Ctrl+Q",      "Quit"),
    ("?",           "Show / hide this legend"),
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
        OverrideTargetPayload,
        RequestPerceptionRefreshPayload,
        StartSkillPayload,
    )
    p = cmd.payload  # type: ignore[attr-defined]
    if isinstance(p, StartSkillPayload):
        return f"START_SKILL({p.skill_name.value}, {p.target_handle})"
    if isinstance(p, AbortSkillPayload):
        return f"ABORT_SKILL({p.skill_run_id[-4:]}, {p.reason})"
    if isinstance(p, OverrideTargetPayload):
        return f"OVERRIDE_TARGET({p.target_handle})"
    if isinstance(p, RequestPerceptionRefreshPayload):
        return f"REFRESH_PERCEPTION({p.mode})"
    return str(cmd.type)  # type: ignore[attr-defined]


def _derive_services(snap: object) -> tuple[list[tuple[str, str, str]], str]:
    """Return (service_rows, target_info_str) from a PlannerSnapshot."""
    rows: list[tuple[str, str, str]] = []

    perc = getattr(snap, "perception", None)
    if perc is not None:
        ts = getattr(perc, "tracking_status", None)
        status = str(ts.value) if ts else "UNKNOWN"
        color = ("bright_green" if status == "TRACKING"
                 else "yellow" if status in ("RELOCALIZING", "REACQUIRING")
                 else "red")
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

    target = getattr(snap, "target", None)
    target_info: list[tuple[str, str]] = []
    if target is not None:
        handle = getattr(target, "handle", "")
        dist_cm = int(getattr(target, "distance_m", 0) * 100)
        conf = int(getattr(target, "confidence", 0) * 100)
        target_info = [
            ("Target",     handle or "—"),
            ("Distance",   f"{dist_cm} cm"),
            ("Confidence", f"{conf}%"),
        ]

    return rows, target_info


# ── Panel widgets ─────────────────────────────────────────────────

class PlannerPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Skill Runner"

    def compose(self) -> ComposeResult:
        no_data = self._data["skill_name"] is None
        # Skill + run id
        t = Text()
        t.append("Skill:    ", style="bold white")
        if no_data:
            t.append("—", style="#9e9e9e")
        else:
            t.append(self._data["skill_name"], style="bold #4fc3f7")
            t.append(f"  {self._data['skill_run_id']}", style="#9e9e9e")
        yield Static(t)
        # Phase
        t2 = Text()
        t2.append("Phase:    ", style="bold white")
        t2.append("—" if no_data else self._data["skill_phase"], style="#9e9e9e" if no_data else "bold white")
        yield Static(t2)
        # ACT buffer
        t3 = Text()
        t3.append("ACT:      ", style="bold white")
        if no_data:
            t3.append("—", style="#9e9e9e")
        else:
            buf = self._data["act_buffer_ms"]
            low = self._data["act_buffer_low"]
            buf_color = "yellow" if low else "bright_green"
            t3.append(self._data["act_status"], style=f"bold {buf_color}")
            if buf is not None:
                t3.append(f"  buffer: {buf} ms", style="#9e9e9e")
            if low:
                t3.append("  !", style="bold yellow")
        yield Static(t3)
        # Outcome
        t4 = Text()
        t4.append("Outcome:  ", style="bold white")
        if no_data:
            t4.append("—", style="#9e9e9e")
        else:
            outcome = self._data["outcome_state"]
            outcome_color = (
                "bright_green" if outcome == "SUCCESS"
                else "red" if outcome == "FAILURE"
                else "#b0bcd0"
            )
            t4.append(outcome, style=f"bold {outcome_color}")
            if self._data["outcome_reason"]:
                t4.append(f"  ({self._data['outcome_reason']})", style="yellow")
        yield Static(t4)
        # Elapsed
        t5 = Text()
        t5.append("Elapsed:  ", style="bold white")
        if no_data:
            t5.append("—", style="#9e9e9e")
        else:
            elapsed_s = (self._data["elapsed_ms"] or 0) / 1000
            t5.append(f"{elapsed_s:.1f} s", style="#9e9e9e")
        yield Static(t5)


class TargetPerceptionPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Target Perception"

    def compose(self) -> ComposeResult:
        def row(label: str, value: str, value_style: str) -> Static:
            t = Text()
            t.append(f"{label:<12}", style="bold white")
            t.append(value, style=value_style)
            return Static(t)

        tracking = self._data["perc_tracking"]
        no_data = tracking is None

        # Status
        if no_data:
            yield row("Status:", "—", "#9e9e9e")
        else:
            track_color = (
                "bright_green" if tracking == "TRACKING"
                else "yellow" if tracking == "RELOCALIZING"
                else "red"
            )
            t = Text()
            t.append(f"{'Status:':<12}", style="bold white")
            t.append(tracking, style=f"bold {track_color}")
            if self._data["perc_hint_valid"] is False:
                t.append("  invalid", style="bold red")
            yield Static(t)

        # Target
        target = self._data["perc_target"]
        yield row("Target:", target or "—", "#4fc3f7" if target else "#9e9e9e")

        # Distance
        dist = self._data["perc_distance_m"]
        yield row("Distance:", f"{dist * 100:.0f} cm" if dist is not None else "—",
                  "white" if dist is not None else "#9e9e9e")

        # Confidence
        conf = self._data["perc_confidence"]
        yield row("Confidence:", f"{conf * 100:.0f}%" if conf is not None else "—",
                  "white" if conf is not None else "#9e9e9e")

        # Obs age
        age = self._data["perc_obs_age_ms"]
        yield row("Obs age:", f"{age} ms" if age is not None else "—", "#9e9e9e")

        # Failure code
        failure = self._data["perc_failure"]
        fail_color = "bright_green" if failure == "OK" else ("yellow" if failure else "#9e9e9e")
        yield row("Code:", failure or "—", fail_color)


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
        self.border_title = "Servos (6DOF)"

    def compose(self) -> ComposeResult:
        for jid, name, status, temp, load in self._data["servos"]:
            t = Text(no_wrap=True)
            t.append(f"{jid} ", style="bold white")
            t.append(f"{name:<13}", style="white")
            if status == "NC":
                t.append("NC   ", style="#9e9e9e")
                t.append("not connected", style="#9e9e9e")
            else:
                t.append(f"{status:<5}", style="bold bright_green")
                t.append(f"{temp}°C  ", style="white")
                t.append_text(_bar(load))
            yield Static(t)


class TalkPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Talk to Planner"
        # scroll history to bottom after layout settles
        self.call_after_refresh(
            lambda: self.query_one("#prompt-history", VerticalScroll).scroll_end(animate=False)
        )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="prompt-history"):
            for ts, msg in self._data["prompt_history"]:
                t = Text()
                t.append(ts, style="grey62")
                t.append("  ")
                t.append(msg, style="#b0bcd0")
                yield Static(t, classes="history-item")
        yield Input(placeholder="Type a command…", id="planner-input")
        sugg = self._data["suggestions"]
        yield Horizontal(
            Button(f"▶ {sugg[0]}", name=sugg[0], classes="suggestion"),
            Button(f"▶ {sugg[1]}", name=sugg[1], classes="suggestion"),
            id="sugg-row-1",
        )
        yield Horizontal(
            Button(f"▶ {sugg[2]}", name=sugg[2], classes="suggestion"),
            Button(f"▶ {sugg[3]}", name=sugg[3], classes="suggestion"),
            id="sugg-row-2",
        )


class SystemPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "System"

    @staticmethod
    def _service_row(name: str, status: str, color: str) -> Text:
        t = Text()
        t.append("● ", style=color)
        t.append(f"{name:<16}  ", style="white")
        t.append(status, style=f"bold {color}")
        return t

    _TARGET_LABELS = ("Target", "Distance", "Confidence")

    @classmethod
    def _target_rows(cls, target_info: list[tuple[str, str]]) -> list[Static]:
        values = dict(target_info)
        rows = []
        for label in cls._TARGET_LABELS:
            value = values.get(label)
            t = Text()
            t.append("  ")  # match "● " column
            t.append(f"{label:<16}  ", style="bold white")
            t.append(value if value else "—", style="white" if value else "#9e9e9e")
            rows.append(Static(t))
        return rows

    def compose(self) -> ComposeResult:
        for name, status, color in self._data["services"]:
            yield Static(self._service_row(name, status, color))
        for row in self._target_rows(self._data["target_info"]):
            yield row

    async def refresh_live(self, services: list[tuple[str, str, str]], target_info: list[tuple[str, str]]) -> None:
        """Replace content with live service statuses."""
        await self.query("Static").remove()
        for name, status, color in services:
            await self.mount(Static(self._service_row(name, status, color)))
        for row in self._target_rows(target_info):
            await self.mount(row)


class EventsPanel(Container):
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


class PanicPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Panic"

    def compose(self) -> ComposeResult:
        with Horizontal(id="abort-row"):
            yield Button("ABORT NOW!", id="abort-btn")
        yield Static("(Hold 'A' to stop immediately)", id="abort-hint")


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
            ("Tab",   "navigate"),
            ("T",     "type"),
            ("Enter", "send"),
            ("Esc",   "cancel"),
            ("A",     "abort"),
            ("Ctrl+Q","quit"),
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
        Binding("ctrl+y", "copy_all", "copy all", show=False),
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
            yield Static(" Ctrl+Y copy all  ·  R or Esc close", id="reasoning-close-hint")


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
    PanicPanel {
        border: solid #263050;
        border-title-color: #4fc3f7;
        border-title-style: bold;
        padding: 0 1;
        height: auto;
    }

    /* info containers shrink first; Talk + Panic hold their space */
    #left-info            { height: 1fr; min-height: 0; overflow: hidden hidden; }
    #right-info           { height: 1fr; min-height: 0; overflow: hidden hidden; }
    TalkPanel             { height: 2fr; min-height: 14; }
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

    #sugg-row-1, #sugg-row-2 {
        height: 1;
        margin: 0;
    }

    .suggestion {
        width: 1fr;
        height: 1;
        min-width: 0;
        background: transparent;
        color: #4fc3f7;
        border: none;
        text-align: left;
        padding: 0 1;
    }

    .suggestion:hover {
        background: #1e2a40;
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
        Binding("a", "emergency_abort", "ABORT", priority=True, show=False),
        Binding("t", "focus_input", "type", show=False),
        Binding("enter", "send_message", "send", show=False),
        Binding("escape", "cancel_input", "cancel", show=False),
        Binding("r", "show_reasoning", "reasoning", show=False),
        Binding("y", "yank_reasoning", "yank", show=False),
        Binding("question_mark", "show_legend", "legend", show=False),
    ]

    _abort_cooldown: float = 0.0  # monotonic time of last abort
    _last_reasoning: str = ""

    def __init__(
        self,
        runtime: object | None = None,
        agent: object | None = None,
        arm_id: str = "arm0",
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._agent = agent
        self._arm_id = arm_id
        self._panel_data = (
            {**_EMPTY_DATA, "arm_id": arm_id} if runtime is not None else _DATA
        )
        self._run_logger = RunLogger(_RUNS_DIR, arm_id) if runtime is not None else None

    def on_mount(self) -> None:
        self.call_after_refresh(self.set_focus, None)
        if self._runtime:
            self.set_interval(2.0, self._poll_system_panel)

    def on_unmount(self) -> None:
        if self._run_logger:
            self._run_logger.close()

    def compose(self) -> ComposeResult:
        d = self._panel_data
        yield TitleBar(arm_id=d["arm_id"])
        with Vertical(id="body"):
            with Horizontal(id="main-row"):
                with Vertical(id="left-col"):
                    with Vertical(id="left-info"):
                        yield PlannerPanel(data=d, id="planner-panel")
                        yield TargetPerceptionPanel(data=d, id="perception-panel")
                    yield TalkPanel(data=d, id="talk-panel")
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
        elif "suggestion" in btn.classes and btn.name:
            inp = self.query_one("#planner-input", Input)
            inp.value = btn.name
            inp.focus()
            inp.cursor_position = len(inp.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "planner-input":
            self.action_send_message()

    # ── Actions ──

    def action_emergency_abort(self) -> None:
        import time
        now = time.monotonic()
        if now - self._abort_cooldown < 3.0:
            return
        self._abort_cooldown = now
        self.notify(
            "⚠ EMERGENCY STOP triggered!",
            severity="error",
            timeout=10,
            title="ABORT",
        )
        if self._runtime:
            self.run_worker(self._do_abort(), name="abort_worker")

    def action_send_message(self) -> None:
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
        history.mount(Static(t, classes="history-item"))
        history.scroll_end(animate=False)
        # Launch agent call if live
        if self._runtime and self._agent:
            self.run_worker(
                self._do_agent_call(msg),
                group="planner",
                exclusive=True,
            )
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

    def action_show_legend(self) -> None:
        self.push_screen(LegendScreen(), callback=lambda _: self.set_focus(None))

    # ── Live workers ──

    async def _poll_system_panel(self) -> None:
        """Refresh SystemPanel with live service statuses every 2 s."""
        try:
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)  # type: ignore[union-attr]
            services, target_info = _derive_services(snap)
            await self.query_one("#system-panel", SystemPanel).refresh_live(services, target_info)
        except Exception:
            pass

    async def _do_agent_call(self, msg: str) -> None:
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
            from halo.services.planner_service.snapshot_serializer import snapshot_to_dict
            snap = await self._runtime.get_latest_runtime_snapshot(self._arm_id)  # type: ignore[union-attr]
            commands = await self._agent.decide(snap, operator_cmd=msg)  # type: ignore[union-attr]
            reasoning = getattr(self._agent, "last_reasoning", "") or ""
            self._last_reasoning = reasoning

            # Submit commands
            acks = []
            for cmd in commands:
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
                )

            # Update thinking widget
            result_text = Text()
            result_text.append(ts, style="grey62")
            result_text.append("  ")
            if acks:
                descs = ", ".join(_format_cmd(c) for c, _ in acks)
                result_text.append(f"▶ Queued {len(acks)} command(s): {descs}", style="bright_green")
            else:
                snippet = (reasoning[:80] + "…") if len(reasoning) > 80 else reasoning
                result_text.append("▶ No commands", style="grey62")
                if snippet:
                    result_text.append(f" — {snippet}", style="#9e9e9e")
                result_text.append("  [R]", style="#4fc3f7")
            thinking_widget.update(result_text)
            history.scroll_end(animate=False)

        except Exception as exc:
            if self._run_logger:
                self._run_logger.log_interaction(
                    arm_id=self._arm_id,
                    operator_msg=msg,
                    snapshot=None,
                    commands=[],
                    acks=[],
                    error=str(exc),
                )
            err_text = Text()
            err_text.append(ts, style="grey62")
            err_text.append("  ")
            err_text.append(f"✗ {exc}", style="bold red")
            thinking_widget.update(err_text)
            history.scroll_end(animate=False)
            self.notify(str(exc), severity="error", title="Agent error")

    async def _do_abort(self) -> None:
        from uuid import uuid4
        import time
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
    from pathlib import Path
    from halo.runtime.runtime import HALORuntime
    from halo.services.planner_service.agent import PlannerAgent

    # Parse --arm, --model, --base-url from args
    arm_id = "arm0"
    model = "gpt-oss:20B"
    base_url = "http://localhost:11434"

    for i, arg in enumerate(args):
        if arg == "--arm" and i + 1 < len(args):
            arm_id = args[i + 1]
        elif arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
        elif arg == "--base-url" and i + 1 < len(args):
            base_url = args[i + 1]

    runtime = HALORuntime()
    runtime.register_arm(arm_id)
    prompts_dir = Path(__file__).parents[2] / "configs" / "planner"
    agent = PlannerAgent(model, base_url, prompts_dir)
    HALOApp(runtime=runtime, agent=agent, arm_id=arm_id).run()


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
