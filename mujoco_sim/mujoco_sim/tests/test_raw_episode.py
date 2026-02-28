"""Tests for dataset module: Timestep, RawEpisode, HDF5 roundtrip.

No robosuite dependency — uses synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_sim.dataset import EpisodeMetadata, RawEpisode, Timestep, episode_path, read_episode, write_episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NQ = 9  # Panda: 7 joints + 2 gripper fingers
NV = 9
SCENE_H, SCENE_W = 48, 64  # small for tests
WRIST_H, WRIST_W = 24, 32


def _make_timestep(
    i: int,
    *,
    with_object: bool = True,
    with_contacts: bool = False,
    with_phase: bool = False,
) -> Timestep:
    """Create a synthetic timestep with deterministic values."""
    rng = np.random.RandomState(i)
    ts = Timestep(
        rgb_scene=rng.randint(0, 256, (SCENE_H, SCENE_W, 3), dtype=np.uint8),
        rgb_wrist=rng.randint(0, 256, (WRIST_H, WRIST_W, 3), dtype=np.uint8),
        qpos=rng.randn(NQ).astype(np.float64),
        qvel=rng.randn(NV).astype(np.float64),
        gripper=float(rng.rand()),
        ee_pose=rng.randn(7).astype(np.float64),
        action=rng.randn(7).astype(np.float64),
        phase_id=i % 10 if with_phase else None,
        object_pose=rng.randn(7).astype(np.float64) if with_object else None,
        contacts=rng.randn(rng.randint(1, 5)).astype(np.float64) if with_contacts else None,
    )
    return ts


def _make_episode(n: int = 10, **kwargs) -> RawEpisode:
    """Build a RawEpisode with *n* synthetic timesteps."""
    meta = EpisodeMetadata(seed=42, env_name="Lift", robot="Panda", control_freq=20)
    ep = RawEpisode(metadata=meta)
    for i in range(n):
        ep.append(_make_timestep(i, **kwargs))
    return ep


# ---------------------------------------------------------------------------
# Timestep tests
# ---------------------------------------------------------------------------


class TestTimestep:
    def test_creation(self):
        ts = _make_timestep(0)
        assert ts.rgb_scene.shape == (SCENE_H, SCENE_W, 3)
        assert ts.rgb_wrist.shape == (WRIST_H, WRIST_W, 3)
        assert ts.qpos.shape == (NQ,)
        assert ts.qvel.shape == (NV,)
        assert ts.ee_pose.shape == (7,)
        assert ts.action.shape == (7,)
        assert ts.object_pose.shape == (7,)

    def test_optional_fields_none(self):
        ts = _make_timestep(0, with_object=False, with_contacts=False)
        assert ts.phase_id is None
        assert ts.object_pose is None
        assert ts.contacts is None

    def test_phase_id_set(self):
        ts = _make_timestep(3, with_phase=True)
        assert ts.phase_id == 3


# ---------------------------------------------------------------------------
# RawEpisode tests
# ---------------------------------------------------------------------------


class TestRawEpisode:
    def test_append_and_len(self):
        ep = _make_episode(5)
        assert len(ep) == 5

    def test_getitem_single(self):
        ep = _make_episode(5)
        ts = ep[2]
        assert isinstance(ts, Timestep)
        expected = _make_timestep(2)
        np.testing.assert_array_equal(ts.rgb_scene, expected.rgb_scene)

    def test_getitem_slice(self):
        ep = _make_episode(5)
        sliced = ep[1:3]
        assert isinstance(sliced, list)
        assert len(sliced) == 2

    def test_bulk_rgb_scenes(self):
        ep = _make_episode(5)
        arr = ep.rgb_scenes
        assert arr.shape == (5, SCENE_H, SCENE_W, 3)
        assert arr.dtype == np.uint8

    def test_bulk_rgb_wrists(self):
        ep = _make_episode(5)
        arr = ep.rgb_wrists
        assert arr.shape == (5, WRIST_H, WRIST_W, 3)

    def test_bulk_qpos(self):
        ep = _make_episode(5)
        assert ep.qpos_array.shape == (5, NQ)

    def test_bulk_qvel(self):
        ep = _make_episode(5)
        assert ep.qvel_array.shape == (5, NV)

    def test_bulk_gripper(self):
        ep = _make_episode(5)
        assert ep.gripper_array.shape == (5,)

    def test_bulk_ee_poses(self):
        ep = _make_episode(5)
        assert ep.ee_poses.shape == (5, 7)

    def test_bulk_actions(self):
        ep = _make_episode(5)
        assert ep.actions.shape == (5, 7)

    def test_bulk_object_poses_present(self):
        ep = _make_episode(5, with_object=True)
        assert ep.object_poses is not None
        assert ep.object_poses.shape == (5, 7)

    def test_bulk_object_poses_absent(self):
        ep = _make_episode(5, with_object=False)
        assert ep.object_poses is None

    def test_bulk_phase_ids_present(self):
        ep = _make_episode(5, with_phase=True)
        assert ep.phase_ids is not None
        assert ep.phase_ids.shape == (5,)
        assert ep.phase_ids.dtype == np.int32

    def test_bulk_phase_ids_absent(self):
        ep = _make_episode(5)
        assert ep.phase_ids is None

    def test_metadata_defaults(self):
        ep = RawEpisode()
        assert ep.metadata.seed is None
        assert ep.metadata.env_name == "Lift"


# ---------------------------------------------------------------------------
# HDF5 roundtrip tests
# ---------------------------------------------------------------------------


class TestHDF5Roundtrip:
    def test_roundtrip_basic(self, tmp_path):
        """Write + read should produce identical data."""
        ep = _make_episode(10)
        out = write_episode(ep, tmp_path / "test.hdf5")
        assert out.exists()

        loaded = read_episode(out)
        assert len(loaded) == len(ep)

        # Metadata
        assert loaded.metadata.seed == ep.metadata.seed
        assert loaded.metadata.env_name == ep.metadata.env_name
        assert loaded.metadata.robot == ep.metadata.robot
        assert loaded.metadata.control_freq == ep.metadata.control_freq

        # Spot-check first and last timestep
        for idx in [0, -1]:
            orig = ep[idx]
            rt = loaded[idx]
            np.testing.assert_array_equal(rt.rgb_scene, orig.rgb_scene)
            np.testing.assert_array_equal(rt.rgb_wrist, orig.rgb_wrist)
            np.testing.assert_allclose(rt.qpos, orig.qpos)
            np.testing.assert_allclose(rt.qvel, orig.qvel)
            assert rt.gripper == pytest.approx(orig.gripper)
            np.testing.assert_allclose(rt.ee_pose, orig.ee_pose)
            np.testing.assert_allclose(rt.action, orig.action)
            np.testing.assert_allclose(rt.object_pose, orig.object_pose)

    def test_roundtrip_no_object_pose(self, tmp_path):
        """Episodes without object_pose should roundtrip cleanly."""
        ep = _make_episode(5, with_object=False)
        out = write_episode(ep, tmp_path / "no_obj.hdf5")
        loaded = read_episode(out)

        assert loaded[0].object_pose is None
        assert loaded.object_poses is None

    def test_roundtrip_with_contacts(self, tmp_path):
        """Contacts (variable-length) survive roundtrip."""
        ep = _make_episode(5, with_contacts=True)
        out = write_episode(ep, tmp_path / "contacts.hdf5")
        loaded = read_episode(out)

        for i in range(len(ep)):
            orig_c = ep[i].contacts
            loaded_c = loaded[i].contacts
            if orig_c is not None:
                assert loaded_c is not None
                np.testing.assert_allclose(loaded_c, orig_c)

    def test_roundtrip_seed_none(self, tmp_path):
        """seed=None roundtrips as None (stored as -1)."""
        ep = RawEpisode(metadata=EpisodeMetadata(seed=None))
        ep.append(_make_timestep(0))
        out = write_episode(ep, tmp_path / "no_seed.hdf5")
        loaded = read_episode(out)
        assert loaded.metadata.seed is None

    def test_roundtrip_extra_metadata(self, tmp_path):
        """Extra metadata keys survive roundtrip."""
        meta = EpisodeMetadata(seed=7, extra={"teacher": "pick", "version": "1"})
        ep = RawEpisode(metadata=meta)
        ep.append(_make_timestep(0))
        out = write_episode(ep, tmp_path / "extra.hdf5")
        loaded = read_episode(out)
        assert loaded.metadata.extra["teacher"] == "pick"
        assert loaded.metadata.extra["version"] == "1"

    def test_gzip_compression(self, tmp_path):
        """Image datasets should use gzip compression."""
        import h5py

        ep = _make_episode(5)
        out = write_episode(ep, tmp_path / "gzip.hdf5")
        with h5py.File(out, "r") as f:
            assert f["obs/rgb_scene"].compression == "gzip"
            assert f["obs/rgb_wrist"].compression == "gzip"
            # Non-image datasets should not be compressed
            assert f["obs/qpos"].compression is None

    def test_roundtrip_with_phase_ids(self, tmp_path):
        """Phase IDs survive roundtrip."""
        ep = _make_episode(5, with_phase=True)
        out = write_episode(ep, tmp_path / "phases.hdf5")
        loaded = read_episode(out)

        assert loaded.phase_ids is not None
        np.testing.assert_array_equal(loaded.phase_ids, ep.phase_ids)
        for i in range(len(ep)):
            assert loaded[i].phase_id == ep[i].phase_id

    def test_roundtrip_without_phase_ids(self, tmp_path):
        """Episodes without phase_ids roundtrip as None."""
        ep = _make_episode(5)
        out = write_episode(ep, tmp_path / "no_phases.hdf5")
        loaded = read_episode(out)
        assert loaded.phase_ids is None
        assert loaded[0].phase_id is None

    def test_bulk_arrays_match_after_roundtrip(self, tmp_path):
        """Bulk accessors on loaded episode match original."""
        ep = _make_episode(8)
        out = write_episode(ep, tmp_path / "bulk.hdf5")
        loaded = read_episode(out)

        np.testing.assert_array_equal(loaded.rgb_scenes, ep.rgb_scenes)
        np.testing.assert_array_equal(loaded.rgb_wrists, ep.rgb_wrists)
        np.testing.assert_allclose(loaded.qpos_array, ep.qpos_array)
        np.testing.assert_allclose(loaded.qvel_array, ep.qvel_array)
        np.testing.assert_allclose(loaded.gripper_array, ep.gripper_array)
        np.testing.assert_allclose(loaded.ee_poses, ep.ee_poses)
        np.testing.assert_allclose(loaded.actions, ep.actions)
        np.testing.assert_allclose(loaded.object_poses, ep.object_poses)


# ---------------------------------------------------------------------------
# episode_path helper
# ---------------------------------------------------------------------------


class TestEpisodePath:
    def test_format(self):
        p = episode_path("/data/episodes", 42)
        assert str(p) == "/data/episodes/ep_000042.hdf5"

    def test_zero(self):
        p = episode_path("/data", 0)
        assert p.name == "ep_000000.hdf5"
