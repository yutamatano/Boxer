"""Measured IMU tilt must not become a fabricated world position."""

import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from boxernet.boxernet import BoxerNet
from loaders.neon_loader import NeonLoader
from test_neon_loader import make_geometry, recording
from utils.neon_gravity import camera_gravity
from utils.neon_vio import scene_to_imu


def write_imu(root, times, rotations):
    from pupil_labs.neon_recording.timeseries.imu.imu_timeseries import IMUTimeseries

    dtype = [("timestamp_ns", "int64")] + [
        (name, "float32") for name in IMUTimeseries.FALLBACK_DTYPE.names
    ]
    rows = np.zeros(len(times), dtype=dtype)
    rows["timestamp_ns"] = times
    (root / "imu.dtype").write_text(repr(dtype))
    q = rotations.as_quat()
    for axis, values in zip("xyzw", q.T):
        rows[f"quaternion_{axis}"] = values
    for axis, values in zip("xyz", rotations.inv().apply([0., 0., 1.]).T):
        rows[f"accel_{axis}"] = values
    rows.tofile(root / "imu ps1.raw")
    np.asarray(times, dtype="<i8").tofile(root / "imu ps1.time")


def test_slerp_keeps_nanosecond_epochs_and_handles_quaternion_sign():
    epoch = 1_790_000_000_000_000_123
    times = np.array([epoch, epoch + 2], dtype=np.int64)
    q = Rotation.from_euler("x", [[0], [90]], degrees=True).as_quat()
    q[1] *= -1
    result = camera_gravity(np.array([epoch + 1], dtype=np.int64), times, q)
    expected = Rotation.from_euler("x", 45, degrees=True).inv().apply([0., 0., -1.])
    np.testing.assert_allclose(result[0], expected @ scene_to_imu()[:3, :3], atol=1e-12)


def test_no_extrapolation_no_interpolation_across_invalid_samples_or_long_gaps():
    times = np.array([0, 10_000_000, 20_000_000, 200_000_000], dtype=np.int64)
    q = np.tile([0., 0., 0., 1.], (4, 1))
    q[1] = np.nan
    targets = np.array([-1, 0, 5_000_000, 15_000_000, 100_000_000, 200_000_001], dtype=np.int64)
    g = camera_gravity(targets, times, q)
    assert np.isfinite(g).all(axis=1).tolist() == [False, True, False, False, False, False]
    with pytest.raises(ValueError, match="increasing int64"):
        camera_gravity(targets, times.astype(float), q)


def test_gravity_is_independent_of_magnetic_heading():
    time = np.array([0], dtype=np.int64)
    tilt = Rotation.from_euler("xy", [25, -15], degrees=True)
    heading = Rotation.from_euler("z", 120, degrees=True)
    first = camera_gravity(time, time, tilt.as_quat()[None])
    second = camera_gravity(time, time, (heading * tilt).as_quat()[None])
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_native_imu_tilts_only_local_voxel_not_camera_or_world_flags(recording, tmp_path):
    root, times = recording
    rotations = Rotation.from_euler("x", [[20]] * 5, degrees=True)
    write_imu(root, times, rotations)
    geometry = tmp_path / "geometry.npz"
    make_geometry(geometry, times[:2])
    loader = NeonLoader(root, geometry_path=geometry, resize=32)
    try:
        assert loader.local_gravity_valid.tolist() == [False, False, True, True, True]
        assert "T_world_voxel0" not in loader.load(0)
        datum = loader.load(2)
        assert datum["has_world_pose"] is False
        torch.testing.assert_close(datum["T_world_rig0"].matrix, loader.local_camera_pose().matrix)
        g_local = datum["T_world_rig0"].R.numpy() @ loader.local_gravity_camera[2]
        np.testing.assert_allclose(datum["T_world_voxel0"].R.numpy()[:, 2], -g_local, atol=1e-6)
        np.testing.assert_allclose(datum["T_world_voxel0"].t.numpy(), 0)

        # Exercise the actual model input path without allocating DINO/weights.
        model = BoxerNet.__new__(BoxerNet)
        torch.nn.Module.__init__(model)
        model.device = "cpu"
        datum["bb2d"] = torch.tensor([[1., 20., 2., 25.]])
        inputs = model.prepare_inputs(datum)
        torch.testing.assert_close(inputs["T_world_voxel0"].matrix[0], datum["T_world_voxel0"].matrix)
        loader.write_metadata(tmp_path, "gravity")
        meta = json.loads((tmp_path / "gravity_input.json").read_text())
        assert meta["world_pose_frames"] == 2
        assert meta["local_gravity"]["imu_frames"] == 3
        assert meta["local_gravity"]["upright_fallback_frames"] == 0
    finally:
        loader.close()


def test_prepared_clock_offset_and_calibrated_extrinsics_are_used(recording, tmp_path):
    root, times = recording
    rotations = Rotation.from_euler("y", [[10]] * 5, degrees=True)
    write_imu(root, times - 2000, rotations)
    transform = scene_to_imu()
    transform[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps({
        "recording": str(root), "image_clock": "aux", "imu_offset_ms_applied": .001,
        "T_imu_camera": transform.tolist(),
    }))
    loader = NeonLoader(root, resize=32, imu_config_path=config)
    try:
        assert loader.local_gravity_valid.all()
        expected = rotations.inv().apply([0., 0., -1.]) @ transform[:3, :3]
        np.testing.assert_allclose(loader.local_gravity_camera, expected, atol=1e-7)
        assert loader.imu_time_source == "aux" and loader.imu_offset_ms == .001
    finally:
        loader.close()
    config.write_text(json.dumps({"recording": str(tmp_path / "other")}))
    with pytest.raises(ValueError, match="another recording"):
        NeonLoader(root, imu_config_path=config)
