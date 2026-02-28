"""RunReport — post-run metrics computed from EventRecorder data.

Provides structured analysis of a HALO test run: per-skill timing,
safety events, perception stats, and control throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halo.contracts.enums import PhaseId
from halo.contracts.events import EventType
from halo.testing.event_recorder import EventRecorder, RecordedEvent


@dataclass
class PhaseTimingEntry:
    """Timing for a single phase within a skill run."""

    phase_id: int
    phase_name: str
    enter_at: float  # monotonic timestamp
    exit_at: float | None = None  # None if phase was still active at end

    @property
    def duration_s(self) -> float | None:
        if self.exit_at is None:
            return None
        return self.exit_at - self.enter_at


@dataclass
class SkillMetrics:
    """Metrics for a single skill execution."""

    skill_run_id: str | None
    outcome: str  # "succeeded", "failed", "in_progress"
    start_at: float | None = None
    end_at: float | None = None
    phase_timings: list[PhaseTimingEntry] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        if self.start_at is None or self.end_at is None:
            return None
        return self.end_at - self.start_at


@dataclass
class SafetyMetrics:
    """Aggregate safety metrics for the run."""

    reflex_count: int = 0
    recovery_count: int = 0
    reflex_events: list[RecordedEvent] = field(default_factory=list)


@dataclass
class PerceptionMetrics:
    """Aggregate perception metrics for the run."""

    target_acquisitions: int = 0
    perception_failures: int = 0
    perception_recoveries: int = 0
    scene_describes: int = 0
    vlm_calls: int = 0  # scene_describes is a proxy for VLM calls


@dataclass
class ControlMetrics:
    """Aggregate control metrics for the run."""

    command_accepted: int = 0
    command_rejected: int = 0


@dataclass
class RunReport:
    """Top-level run report aggregating all metrics."""

    total_events: int = 0
    duration_s: float = 0.0
    event_types: dict[str, int] = field(default_factory=dict)
    skills: list[SkillMetrics] = field(default_factory=list)
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)
    perception: PerceptionMetrics = field(default_factory=PerceptionMetrics)
    control: ControlMetrics = field(default_factory=ControlMetrics)
    event_timeline: list[tuple[float, str]] = field(default_factory=list)


def compute_run_report(recorder: EventRecorder) -> RunReport:
    """Compute a RunReport from recorded events."""
    events = recorder.all_events
    if not events:
        return RunReport()

    report = RunReport(
        total_events=len(events),
        duration_s=events[-1].recorded_at - events[0].recorded_at if len(events) > 1 else 0.0,
    )

    # Event type counts
    for rec in events:
        name = rec.event.type.value
        report.event_types[name] = report.event_types.get(name, 0) + 1

    # Event timeline
    t0 = events[0].recorded_at
    report.event_timeline = [(rec.recorded_at - t0, rec.event.type.value) for rec in events]

    # Skill metrics
    current_skill: SkillMetrics | None = None
    # Phase tracking
    current_phase_entry: PhaseTimingEntry | None = None

    for rec in events:
        evt = rec.event

        if evt.type == EventType.SKILL_STARTED:
            current_skill = SkillMetrics(
                skill_run_id=evt.data.get("skill_run_id"),
                outcome="in_progress",
                start_at=rec.recorded_at,
            )
            report.skills.append(current_skill)

        elif evt.type == EventType.SKILL_SUCCEEDED:
            if current_skill is not None:
                current_skill.outcome = "succeeded"
                current_skill.end_at = rec.recorded_at

        elif evt.type == EventType.SKILL_FAILED:
            if current_skill is not None:
                current_skill.outcome = "failed"
                current_skill.end_at = rec.recorded_at

        elif evt.type == EventType.PHASE_ENTER:
            phase_id = evt.data.get("phase_id", -1)
            try:
                phase_name = PhaseId(phase_id).name
            except ValueError:
                phase_name = f"UNKNOWN_{phase_id}"
            current_phase_entry = PhaseTimingEntry(
                phase_id=phase_id,
                phase_name=phase_name,
                enter_at=rec.recorded_at,
            )
            if current_skill is not None:
                current_skill.phase_timings.append(current_phase_entry)

        elif evt.type == EventType.PHASE_EXIT:
            if current_phase_entry is not None:
                current_phase_entry.exit_at = rec.recorded_at
                current_phase_entry = None

        # Safety
        elif evt.type == EventType.SAFETY_REFLEX_TRIGGERED:
            report.safety.reflex_count += 1
            report.safety.reflex_events.append(rec)
        elif evt.type == EventType.SAFETY_RECOVERED:
            report.safety.recovery_count += 1

        # Perception
        elif evt.type == EventType.TARGET_ACQUIRED:
            report.perception.target_acquisitions += 1
        elif evt.type == EventType.PERCEPTION_FAILURE:
            report.perception.perception_failures += 1
        elif evt.type == EventType.PERCEPTION_RECOVERED:
            report.perception.perception_recoveries += 1
        elif evt.type == EventType.SCENE_DESCRIBED:
            report.perception.scene_describes += 1
            report.perception.vlm_calls += 1

        # Control
        elif evt.type == EventType.COMMAND_ACCEPTED:
            report.control.command_accepted += 1
        elif evt.type == EventType.COMMAND_REJECTED:
            report.control.command_rejected += 1

    return report
