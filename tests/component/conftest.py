"""Shared fixtures for component tests.

Each component test wires ONE real service with mocked deps at
fast-integration latency.
"""

import pytest

from halo.testing.mock_fns import LatencyProfile


@pytest.fixture
def latency() -> LatencyProfile:
    return LatencyProfile.fast_integration()
