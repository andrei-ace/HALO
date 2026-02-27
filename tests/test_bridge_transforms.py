"""Tests for bridge coordinate frame transforms."""

import math

import pytest

from halo.bridge.transforms import world_to_ee_frame


def test_identity_quaternion_is_passthrough():
    """Identity quaternion [1,0,0,0] should return the input unchanged."""
    v = [1.0, 2.0, 3.0]
    result = world_to_ee_frame(v, [1.0, 0.0, 0.0, 0.0])
    assert result == pytest.approx(v)


def test_90deg_yaw_rotation():
    """90-degree yaw (rotation about Z): q = [cos45, 0, 0, sin45].

    R maps body→world.  R^T maps world→body.
    With +90° yaw: world +X maps to EE -Y.
    """
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    q = [c, 0.0, 0.0, s]

    v_world = [1.0, 0.0, 0.0]
    result = world_to_ee_frame(v_world, q)

    assert result[0] == pytest.approx(0.0, abs=1e-10)
    assert result[1] == pytest.approx(-1.0, abs=1e-10)
    assert result[2] == pytest.approx(0.0, abs=1e-10)


def test_180deg_yaw_rotation():
    """180-degree yaw: world +X maps to EE -X."""
    q = [0.0, 0.0, 0.0, 1.0]  # 180° about Z
    result = world_to_ee_frame([1.0, 0.0, 0.0], q)
    assert result[0] == pytest.approx(-1.0, abs=1e-10)
    assert result[1] == pytest.approx(0.0, abs=1e-10)
    assert result[2] == pytest.approx(0.0, abs=1e-10)


def test_preserves_magnitude():
    """Rotation should not change vector magnitude."""
    v = [3.0, 4.0, 0.0]
    # Exact unit quaternion for 90° about Y
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    q = [c, 0.0, s, 0.0]
    result = world_to_ee_frame(v, q)
    original_mag = math.sqrt(sum(d * d for d in v))
    result_mag = math.sqrt(sum(d * d for d in result))
    assert result_mag == pytest.approx(original_mag, abs=1e-10)


def test_zero_vector():
    """Zero vector stays zero regardless of quaternion."""
    result = world_to_ee_frame([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    assert result == pytest.approx([0.0, 0.0, 0.0])
