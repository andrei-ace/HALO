# PlannerService

Event-driven LLM orchestrator (30 s watchdog). Fetches the latest runtime snapshot, calls a LangGraph ReAct agent to decide commands, and submits them back to the runtime.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `PlannerServiceConfig` — watchdog interval, max commands per tick |
| `snapshot_serializer.py` | `snapshot_to_dict()` — PlannerSnapshot → plain dict for LLM context |
| `tools.py` | `build_tools(ctx)` — 5 LangChain `@tool` functions closed over `AgentContext` |
| `agent.py` | `PlannerAgent` — LangGraph ReAct agent with ChatOllama, middleware, loop detection |
| `service.py` | `PlannerService` — event loop, tick, command submission |

## Key Types

```python
DecideFn = Callable[[PlannerSnapshot], Awaitable[list[CommandEnvelope]]]

PlannerService(arm_id, runtime, decide_fn, config=PlannerServiceConfig())
PlannerAgent(model_name="gpt-oss:20b", base_url="http://localhost:11434", prompts_dir=...)
make_decide_fn(...)  # convenience factory → PlannerAgent.decide
```

## Tools (5 LangChain @tool functions)

| Tool | Command | Precondition |
|------|---------|-------------|
| `start_skill(skill_name, target_handle, options)` | START_SKILL | snapshot_id |
| `abort_skill(skill_run_id, reason)` | ABORT_SKILL | snapshot_id |
| `override_target(skill_run_id, target_handle)` | OVERRIDE_TARGET | snapshot_id |
| `describe_scene(reason)` | DESCRIBE_SCENE | None (stateless) |
| `track_object(target_handle)` | TRACK_OBJECT | None (stateless) |

**One-tool-per-tick invariant**: each tool can fire at most once per tick. Second call is rejected.

## PlannerAgent Design

- **LLM**: ChatOllama (local Ollama, `gpt-oss:20b` default)
- **Checkpointer**: InMemorySaver (conversation persists across ticks in memory)
- **Middleware** (`_deprecate_old_snapshots`): replaces all but the latest "Current robot state:" message with a deprecation notice — enforces exactly-one-snapshot invariant
- **Loop detection** (MAX_LOOP_RETRIES=4): tracks consecutive identical commands across ticks; rejects batch on streak >= 4
- `decide(snap, operator_cmd=None)`: main entry point. Formats snapshot as JSON HumanMessage, optionally appends operator instruction, invokes agent, returns accumulated commands
- `reset_loop_state()`: called when operator sends new instruction

## Service Event Loop

- `_drain_events()`: reads from EventBus queue; sets `_urgent_event` on: SKILL_SUCCEEDED, SKILL_FAILED, SAFETY_REFLEX_TRIGGERED, PERCEPTION_FAILURE, SCENE_DESCRIBED, TARGET_ACQUIRED, COMMAND_REJECTED
- `_run_loop()`: waits on urgent event OR watchdog timeout (30 s) → calls `tick()`
- `tick()`: get snapshot → decide_fn → submit up to `max_commands_per_tick` → return list[CommandAck]
- **Ticks are serialized** — decide_fn is awaited before the next event is processed
- `start()` issues an initial DESCRIBE_SCENE command for scene acquisition

## snapshot_to_dict()

Converts PlannerSnapshot to plain dict. Key conversions:
- `PhaseId` → `.name` string
- `StrEnum` values → `.value`
- Reason code tuples → `list[str]`
- Outcome set to `None` when `snap.skill is None` (avoids stale LLM context)

## Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `watchdog_interval_s` | 30.0 | Max silence before forced tick |
| `max_commands_per_tick` | 5 | Prevent command flooding |

## Prompts

Loaded from `halo/configs/planner/`:
- `system_prompt.md` — core agent instructions
- `skills/pick.md` — PICK skill reference
- `skills/place.md` — PLACE skill reference

## Integration

- **Reads**: latest snapshot via `runtime.get_latest_runtime_snapshot(arm_id)`
- **Writes**: commands via `runtime.submit_command(cmd)` → CommandAck
- **Subscribes to**: EventBus (urgent events wake the loop)
- **Consumed by**: SkillRunnerService + TargetPerceptionService execute the commands

## Testing

`tick()` is directly callable. Tests inject a mock `decide_fn` and verify command submission + ack handling. Integration tests (`integration/`) exercise the full LangGraph agent against Ollama.
