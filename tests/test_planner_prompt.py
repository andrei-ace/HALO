from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parents[1] / "configs" / "planner" / "system_prompt.md"


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_prompt_requires_operator_task_for_action() -> None:
    prompt = _prompt_text()

    assert "NEVER act without an operator task" in prompt
    assert "--- NEW OPERATOR TASK ---" in prompt
    assert "still-pending operator instruction" in prompt


def test_prompt_marks_event_only_ticks_as_non_actionable() -> None:
    prompt = _prompt_text()

    assert "An event-only message is never a task." in prompt
    assert "If the current message contains only event lines like `[event: SCENE_DESCRIBED]`" in prompt
    assert "Do not invent a task from it." in prompt


def test_prompt_includes_scene_only_regression_example() -> None:
    prompt = _prompt_text()

    assert "Visible affordances are not instructions." in prompt
    assert 'Seeing a cube and a tray does NOT mean "move the cube to the tray"' in prompt
    assert "If the current message is only `[event: SCENE_DESCRIBED]`" in prompt
    assert "green_cube_01" in prompt
    assert "beige_tray_01" in prompt
    assert "Scene received, awaiting operator instruction." in prompt


def test_prompt_says_operator_requested_abort_must_not_resume_task() -> None:
    prompt = _prompt_text()

    assert "Operator-requested abort clears authorization." in prompt
    assert "If a skill or task was aborted because the operator asked to stop" in prompt
    assert "do NOT retry or resume that action" in prompt
    assert "After an operator-requested abort, stay stopped." in prompt
    assert "Do not re-queue it from follow-up events, retries, or prior task context" in prompt
