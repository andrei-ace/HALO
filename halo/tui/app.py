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

_EMPTY_DATA = dict(
    arm_id="arm0",
    skill_name="—", skill_run_id="—", skill_phase="—",
    act_status="—", act_buffer_ms=0, act_buffer_low=False,
    outcome_state="—", outcome_reason=None, elapsed_ms=0,
    actions=[],
    servos=[],
    prompt_history=[],
    suggestions=_DATA["suggestions"],  # keep as useful operator shortcuts
    services=[],
    target_info="",
    events=[],
)

_LEGEND = [
    ("Tab / Shift+Tab", "Navigate between input and buttons"),
    ("T",               "Focus the message input directly"),
    ("Enter",       "Send message to planner"),
    ("Esc",         "Clear input and return to monitoring mode"),
    ("R",           "Show full planner reasoning for last response"),
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


# ── Panel widgets ─────────────────────────────────────────────────

class PlannerPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Skill Runner"

    def compose(self) -> ComposeResult:
        # Skill + run id
        t = Text()
        t.append("Skill:    ", style="bold white")
        t.append(self._data["skill_name"], style="bold #4fc3f7")
        t.append(f"  {self._data['skill_run_id']}", style="#9e9e9e")
        yield Static(t)
        # Phase
        t2 = Text()
        t2.append("Phase:    ", style="bold white")
        t2.append(self._data["skill_phase"], style="bold white")
        yield Static(t2)
        # ACT buffer
        buf = self._data["act_buffer_ms"]
        low = self._data["act_buffer_low"]
        buf_color = "yellow" if low else "bright_green"
        t3 = Text()
        t3.append("ACT:      ", style="bold white")
        t3.append(self._data["act_status"], style=f"bold {buf_color}")
        t3.append(f"  buffer: {buf} ms", style="#9e9e9e")
        if low:
            t3.append("  !", style="bold yellow")
        yield Static(t3)
        # Outcome
        outcome = self._data["outcome_state"]
        outcome_color = (
            "bright_green" if outcome == "SUCCESS"
            else "red" if outcome == "FAILURE"
            else "#b0bcd0"
        )
        t4 = Text()
        t4.append("Outcome:  ", style="bold white")
        t4.append(outcome, style=f"bold {outcome_color}")
        if self._data["outcome_reason"]:
            t4.append(f"  ({self._data['outcome_reason']})", style="yellow")
        yield Static(t4)
        # Elapsed
        elapsed_s = self._data["elapsed_ms"] / 1000
        t5 = Text()
        t5.append("Elapsed:  ", style="bold white")
        t5.append(f"{elapsed_s:.1f} s", style="#9e9e9e")
        yield Static(t5)


class ActionsPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Planner Actions"

    def compose(self) -> ComposeResult:
        for ts, desc, kind, cmd_id in self._data["actions"]:
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

    def append_cmd(self, ts: str, desc: str, short_id: str) -> None:
        t = Text()
        t.append(ts, style="grey62")
        t.append("  ")
        t.append(desc, style="white")
        t.append(f"  →{short_id}", style="#4fc3f7")
        self.mount(Static(t))

    def append_ack(self, ts: str, status: str, short_id: str) -> None:
        t = Text()
        t.append(ts, style="grey62")
        t.append("  ")
        color = "bright_green" if status == "ACCEPTED" else "red"
        t.append(f"  {short_id} ", style="#4fc3f7")
        t.append(status, style=f"bold {color}")
        self.mount(Static(t))


class ServosPanel(Container):
    def __init__(self, data: dict = _DATA, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def on_mount(self) -> None:
        self.border_title = "Servos (6DOF)"

    def compose(self) -> ComposeResult:
        for jid, name, _status, temp, load in self._data["servos"]:
            t = Text(no_wrap=True)
            t.append(f"{jid} ", style="bold white")
            t.append(f"{name:<13}", style="white")
            t.append("OK   ", style="bold bright_green")
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

    def compose(self) -> ComposeResult:
        for name, status, color in self._data["services"]:
            t = Text()
            t.append("● ", style=color)
            t.append(f"{name}  ", style="white")
            t.append(status, style=f"bold {color}")
            yield Static(t)
        yield Static("")
        yield Static(self._data["target_info"])


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
    ]

    def __init__(self, reasoning: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._reasoning = reasoning

    def action_close(self) -> None:
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Container(id="reasoning-box"):
            yield Static(" Planner reasoning (last response)", id="reasoning-heading")
            yield Static("")
            if self._reasoning:
                # Word-wrap long lines by splitting on newlines first
                for line in self._reasoning.splitlines():
                    yield Static(line or " ", classes="reasoning-line")
            else:
                yield Static("  (no reasoning captured yet)", classes="reasoning-line")
            yield Static("")
            yield Static(" Press R or Esc to close", id="reasoning-close-hint")


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

    PlannerPanel { height: 1fr; min-height: 9;  }
    ActionsPanel { height: 1fr; min-height: 8;  }
    TalkPanel    { height: 2fr; min-height: 14; }
    EventsPanel  { height: 1fr; min-height: 8;  }
    PanicPanel   { min-height: 8; }

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
        width: 80;
        height: auto;
        max-height: 40;
        background: #141d30;
        border: solid #4fc3f7;
        padding: 1 0;
        overflow-y: auto;
    }

    #reasoning-heading {
        color: #4fc3f7;
        text-style: bold;
        margin-bottom: 0;
    }

    .reasoning-line {
        padding: 0 2;
        color: #b0bcd0;
    }

    #reasoning-close-hint {
        color: #8a8a8a;
    }
    """

    BINDINGS = [
        Binding("a", "emergency_abort", "ABORT", priority=True, show=False),
        Binding("t", "focus_input", "type", show=False),
        Binding("enter", "send_message", "send", show=False),
        Binding("escape", "cancel_input", "cancel", show=False),
        Binding("r", "show_reasoning", "reasoning", show=False),
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

    def on_unmount(self) -> None:
        if self._run_logger:
            self._run_logger.close()

    def compose(self) -> ComposeResult:
        d = self._panel_data
        yield TitleBar(arm_id=d["arm_id"])
        with Vertical(id="body"):
            with Horizontal(id="main-row"):
                with Vertical(id="left-col"):
                    yield PlannerPanel(data=d, id="planner-panel")
                    yield ActionsPanel(data=d, id="actions-panel")
                    yield TalkPanel(data=d, id="talk-panel")
                with Vertical(id="right-col"):
                    yield SystemPanel(data=d, id="system-panel")
                    yield ServosPanel(data=d, id="servos-panel")
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

    def action_show_legend(self) -> None:
        self.push_screen(LegendScreen(), callback=lambda _: self.set_focus(None))

    # ── Live workers ──

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

            # Append to ActionsPanel
            actions_panel = self.query_one("#actions-panel", ActionsPanel)
            for cmd, ack in acks:
                cmd_ts = datetime.now().strftime("%H:%M:%S")
                short_id = cmd.command_id[-4:]
                actions_panel.append_cmd(cmd_ts, _format_cmd(cmd), short_id)
                actions_panel.append_ack(cmd_ts, ack.status.value, short_id)

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
        from datetime import datetime
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

        actions_panel = self.query_one("#actions-panel", ActionsPanel)
        ts = datetime.now().strftime("%H:%M:%S")
        short_id = cmd.command_id[-4:]
        actions_panel.append_cmd(ts, _format_cmd(cmd), short_id)
        actions_panel.append_ack(ts, ack.status.value, short_id)


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
