"""HALO testing infrastructure — EventRecorder, state seeders, mock factories, HeadlessRunner."""

from halo.testing.event_recorder import EventRecorder, RecordedEvent
from halo.testing.metrics import RunReport, compute_run_report
from halo.testing.runner import HeadlessRunner, RunnerConfig
from halo.testing.state_seeder import make_act, make_perception, make_skill, make_target, seed_store

__all__ = [
    "EventRecorder",
    "HeadlessRunner",
    "RecordedEvent",
    "RunReport",
    "RunnerConfig",
    "compute_run_report",
    "make_act",
    "make_perception",
    "make_skill",
    "make_target",
    "seed_store",
]
