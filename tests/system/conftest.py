"""Shared fixtures for system tests.

System tests wire ALL services together with mocked external deps.
"""

import pytest

from halo.testing.mock_fns import LatencyProfile


@pytest.fixture
def latency() -> LatencyProfile:
    return LatencyProfile.fast_integration()
