# configs/

Runtime prompt templates and Mermaid FSM definitions consumed by HALO services.

## Structure

| Path | Consumer | Purpose |
|---|---|---|
| `planner/system_prompt.md` | `PlannerService` (`agent.py → _load_prompts()`) | Core planner agent instructions (role, tools, constraints) |
| `perception/scene_analysis.md` | `TargetPerceptionService` (`ollama_vlm_fn.py`) | VLM prompt for object detection / scene description |
| `skills/<name>/default.mmd` | `SkillRunnerService` (`mermaid_parser.py → build_default_registry()`) | Mermaid stateDiagram-v2 defining FSM topology (nodes, edges, guards) |
| `skills/<name>/system_prompt.md` | `PlannerService` (`agent.py → _load_prompts()`) | Skill-specific reference appended to planner context (pick, place only) |
| `live_agent/system_prompt.md` | `LiveAgentSession` (`live_agent.py`) | Conversational voice/text assistant system prompt |

## How prompts are loaded

- **Planner**: `agent.py` reads `configs/planner/system_prompt.md`, then globs `configs/skills/*/system_prompt.md` and appends each under a "Skill Reference" header.
- **Perception**: `ollama_vlm_fn.py` reads `configs/perception/scene_analysis.md` as the VLM system prompt.
- **FSM definitions**: `build_default_registry()` in `definitions.py` parses each `configs/skills/<name>/default.mmd` via `parse_mermaid_fsm()` to build immutable `FsmGraph` objects.
- **Live agent**: `LiveAgentSession` reads `configs/live_agent/system_prompt.md` at session start.

## Adding a new skill

1. Create `configs/skills/<name>/default.mmd` — Mermaid stateDiagram-v2 with `[*] --> FIRST_STATE` and `LAST_STATE --> [*]`.
2. Add the skill name to `SkillName` enum in `halo/contracts/enums.py`.
3. Add phase IDs to `PhaseId` enum for each FSM state.
4. Register in `build_default_registry()` (`halo/services/skill_runner_service/definitions.py`).
5. Write state handlers implementing `StateHandler` protocol (`halo/services/skill_runner_service/handlers.py`).
6. (Optional) Add `configs/skills/<name>/system_prompt.md` if the planner needs skill-specific reference text.
