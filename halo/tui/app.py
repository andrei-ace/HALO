"""HALO TUI — static mockup (v0).

Run with:
    uv run python -m halo.tui.app
"""

from __future__ import annotations

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static
from rich.text import Text


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
    # Planner commands issued + acks (command_id abbreviated to last 4 chars)
    actions=[
        ("14:32:09", "START_SKILL(pick, cube-1)",          "cmd", "7a21"),
        ("14:32:09", "ACCEPTED",                            "ack", "7a21"),
        ("14:32:01", "REQUEST_PERCEPTION_REFRESH(fast)",   "cmd", "7a20"),
        ("14:32:01", "ACCEPTED",                            "ack", "7a20"),
    ],
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
    target_info="Target: cube-1, distance: 9 cm, confidence: 86%",
    events=[
        ("14:32:08", "PHASE_ENTER PREGRASP_ALIGN"),
        ("14:32:08", "PERCEPTION_RECOVERED"),
        ("14:32:06", "SKILL_STARTED run-9"),
    ],
)

_LEGEND = [
    ("click / Tab", "Focus the message input (enter typing mode)"),
    ("Enter",       "Send message to planner"),
    ("Esc",         "Clear input and return to monitoring mode"),
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


# ── Panel widgets ─────────────────────────────────────────────────

class PlannerPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Skill Runner"

    def compose(self) -> ComposeResult:
        # Skill + run id
        t = Text()
        t.append("Skill:    ", style="bold white")
        t.append(_DATA["skill_name"], style="bold #4fc3f7")
        t.append(f"  {_DATA['skill_run_id']}", style="#9e9e9e")
        yield Static(t)
        # Phase
        t2 = Text()
        t2.append("Phase:    ", style="bold white")
        t2.append(_DATA["skill_phase"], style="bold white")
        yield Static(t2)
        # ACT buffer
        buf = _DATA["act_buffer_ms"]
        low = _DATA["act_buffer_low"]
        buf_color = "yellow" if low else "bright_green"
        t3 = Text()
        t3.append("ACT:      ", style="bold white")
        t3.append(_DATA["act_status"], style=f"bold {buf_color}")
        t3.append(f"  buffer: {buf} ms", style="#9e9e9e")
        if low:
            t3.append("  !", style="bold yellow")
        yield Static(t3)
        # Outcome
        outcome = _DATA["outcome_state"]
        outcome_color = (
            "bright_green" if outcome == "SUCCESS"
            else "red" if outcome == "FAILURE"
            else "#b0bcd0"
        )
        t4 = Text()
        t4.append("Outcome:  ", style="bold white")
        t4.append(outcome, style=f"bold {outcome_color}")
        if _DATA["outcome_reason"]:
            t4.append(f"  ({_DATA['outcome_reason']})", style="yellow")
        yield Static(t4)
        # Elapsed
        elapsed_s = _DATA["elapsed_ms"] / 1000
        t5 = Text()
        t5.append("Elapsed:  ", style="bold white")
        t5.append(f"{elapsed_s:.1f} s", style="#9e9e9e")
        yield Static(t5)


class ActionsPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Planner Actions"

    def compose(self) -> ComposeResult:
        for ts, desc, kind, cmd_id in _DATA["actions"]:
            t = Text()
            t.append(ts, style="grey62")
            t.append("  ")
            if kind == "cmd":
                t.append(desc, style="white")
                t.append(f"  →{cmd_id}", style="#4fc3f7")
            else:  # ack
                color = "bright_green" if desc == "ACCEPTED" else "red"
                t.append(f"  {cmd_id} ", style="#4fc3f7")
                t.append(desc, style=f"bold {color}")
            yield Static(t)


class ServosPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Servos (6DOF)"

    def compose(self) -> ComposeResult:
        for jid, name, _status, temp, load in _DATA["servos"]:
            t = Text(no_wrap=True)
            t.append(f"{jid} ", style="bold white")
            t.append(f"{name:<13}", style="white")
            t.append("OK   ", style="bold bright_green")
            t.append(f"{temp}°C  ", style="white")
            t.append_text(_bar(load))
            yield Static(t)


class TalkPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Talk to Planner"
        # scroll history to bottom after layout settles
        self.call_after_refresh(
            lambda: self.query_one("#prompt-history", VerticalScroll).scroll_end(animate=False)
        )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="prompt-history"):
            for ts, msg in _DATA["prompt_history"]:
                t = Text()
                t.append(ts, style="grey62")
                t.append("  ")
                t.append(msg, style="#b0bcd0")
                yield Static(t, classes="history-item")
        yield Input(placeholder="Type a command…", id="planner-input")
        sugg = _DATA["suggestions"]
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
    def on_mount(self) -> None:
        self.border_title = "System"

    def compose(self) -> ComposeResult:
        for name, status, color in _DATA["services"]:
            t = Text()
            t.append("● ", style=color)
            t.append(f"{name}  ", style="white")
            t.append(status, style=f"bold {color}")
            yield Static(t)
        yield Static("")
        yield Static(_DATA["target_info"])


class EventsPanel(Container):
    def on_mount(self) -> None:
        self.border_title = "Events"

    def compose(self) -> ComposeResult:
        for ts, desc in _DATA["events"]:
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
    def render(self) -> Text:
        t = Text(justify="center")
        t.append(f"HALO  —  {_DATA['arm_id']}", style="bold white")
        return t


class HintBar(Static):
    """Single-line keybinding hint that spans the full width."""
    def render(self) -> Text:
        t = Text(justify="center")
        t.append("[ ? ] legend", style="#4fc3f7")
        for key, desc in (
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

    def on_key(self, event: events.Key) -> None:
        if event.key in ("question_mark", "escape"):
            self.dismiss()


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
    ActionsPanel,
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

    TalkPanel   { height: 1fr; min-height: 12; }
    EventsPanel { height: 1fr; min-height: 8;  }
    PanicPanel  { min-height: 8; }

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
        height: 1;
        margin: 0;
        background: #111827;
        color: white;
        border: none;
        border-top: solid #263050;
        border-bottom: solid #263050;
        padding: 0 1;
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
    """

    BINDINGS = [
        Binding("a", "emergency_abort", "ABORT", priority=True, show=False),
        Binding("enter", "send_message", "send", show=False),
        Binding("escape", "cancel_input", "cancel", show=False),
        Binding("question_mark", "show_legend", "legend", show=False),
    ]

    def on_mount(self) -> None:
        self.call_after_refresh(self.set_focus, None)

    def compose(self) -> ComposeResult:
        yield TitleBar()
        with Vertical(id="body"):
            with Horizontal(id="main-row"):
                with Vertical(id="left-col"):
                    yield PlannerPanel()
                    yield ActionsPanel()
                    yield ServosPanel()
                    yield TalkPanel()
                with Vertical(id="right-col"):
                    yield SystemPanel()
                    yield EventsPanel()
                    yield PanicPanel()
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
        self.notify("EMERGENCY STOP triggered!", severity="error", timeout=5)

    def action_send_message(self) -> None:
        from datetime import datetime
        inp = self.query_one("#planner-input", Input)
        msg = inp.value.strip()
        if msg:
            self.notify(f"→ Planner: {msg!r}", timeout=3)
            inp.value = ""
            # append to history
            ts = datetime.now().strftime("%H:%M:%S")
            t = Text()
            t.append(ts, style="grey62")
            t.append("  ")
            t.append(msg, style="#b0bcd0")
            history = self.query_one("#prompt-history", VerticalScroll)
            history.mount(Static(t, classes="history-item"))
            history.scroll_end(animate=False)
        self.set_focus(None)

    def action_cancel_input(self) -> None:
        inp = self.query_one("#planner-input", Input)
        inp.value = ""
        self.set_focus(None)

    def action_show_legend(self) -> None:
        self.push_screen(LegendScreen(), callback=lambda _: self.set_focus(None))


def _take_screenshot(path: str = "halo_tui.svg") -> None:
    """Run headlessly, render one frame, save SVG screenshot, exit."""
    import asyncio

    async def _run() -> None:
        app = HALOApp()
        async with app.run_test(headless=True, size=(209, 53)) as pilot:
            await pilot.pause(0.3)  # let layout settle
            svg = app.export_screenshot()
        with open(path, "w") as f:
            f.write(svg)
        print(f"Screenshot saved: {path}")

    asyncio.run(_run())


def main() -> None:
    import sys
    if "--screenshot" in sys.argv:
        idx = sys.argv.index("--screenshot")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "halo_tui.svg"
        _take_screenshot(path)
    else:
        HALOApp().run()


if __name__ == "__main__":
    main()
