"""VIO boundary contracts: exact epochs, units, calibration and lost-frame gaps."""

import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from convert_neon_trajectory import convert
from prepare_neon_vio import rotation_sync_check
from utils.neon_vio import (
    ORB_SLAM3_COMMIT, check_times, imu_in_si, noise_parameters,
    rectification, relative_seconds, scene_to_imu, split_valid_runs,
    validate_transform, write_orb_settings,
)


def test_epoch_subtraction_preserves_ns():
    origin = 1_790_000_000_000_000_123
    stamps = np.array([origin, origin + 1, origin + 100_000_003], dtype=np.int64)
    np.testing.assert_allclose(relative_seconds(stamps, origin), [0, 1e-9, 0.100000003], atol=1e-17)
    with pytest.raises(ValueError):
        relative_seconds(stamps.astype(float), origin)
    with pytest.raises(ValueError):
        check_times(stamps[[0, 0, 2]], "images")


def test_si_imu_keeps_gravity():
    imu = SimpleNamespace(angular_velocity=[[180, 0, -90]], acceleration=[[0, 0, 1]])
    gyro, accel = imu_in_si(imu)
    np.testing.assert_allclose(gyro, [[np.pi, 0, -np.pi / 2]])
    np.testing.assert_allclose(accel, [[0, 0, 9.80665]])


def test_calibration_and_dotted_yaml_roundtrip(tmp_path):
    K = np.array([[100, 0, 80], [0, 110, 60], [0, 0, 1]], dtype=float)
    scaled, maps, height = rectification(K, np.zeros(8), (160, 120), 80)
    assert height == 60 and maps[0].shape == (60, 80)
    np.testing.assert_allclose(scaled[:2, 2], [39.75, 29.75])
    Tbc = validate_transform(scene_to_imu())
    np.testing.assert_allclose(Tbc[:3, 3], [0, -0.0013, -0.00662])
    path = tmp_path / "camera.yaml"
    write_orb_settings(path, {"File.version": "1.0", "Camera1.fx": 50.0,
                              "Camera.width": 80, "IMU.Frequency": 104.3}, Tbc)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    assert storage.getNode("File.version").string() == "1.0"
    assert storage.getNode("Camera1.fx").isReal()
    assert storage.getNode("Camera.width").isInt()
    stored_transform = storage.getNode("IMU.T_b_c1").mat()
    assert stored_transform.dtype == np.float32
    np.testing.assert_allclose(stored_transform, Tbc, atol=1e-7)
    storage.release()
    with pytest.raises(ValueError):
        validate_transform(np.diag([-1, 1, 1, 1]))


def test_uncalibrated_noise_requires_opt_in(tmp_path):
    with pytest.raises(ValueError, match="experimental-noise"):
        noise_parameters(None, False)
    values, source = noise_parameters(None, True)
    assert "uncalibrated" in source
    values["NoiseGyro"] = -1
    path = tmp_path / "noise.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        noise_parameters(path, False)


def test_runs_split_on_loss_source_gap_and_map_change():
    runs = split_valid_runs([0, 1, 2, 3, 5, 6, 7], [0, 0, 0, 0, 0, 1, 1],
                            [True, True, False, True, True, True, True])
    assert [r.tolist() for r in runs] == [[0, 1], [3], [4], [5, 6]]


def test_rotation_diagnostic_convention_and_offset():
    times = np.linspace(0, 5, 501)
    omega = np.column_stack([np.zeros_like(times), np.zeros_like(times), 0.3 + 0.3 * times])
    Tbc = scene_to_imu()
    pairs = []
    offset = 0.04
    for t0 in np.arange(0.3, 3.8, 0.3):
        t1 = t0 + 0.2
        angle = 0.3 * (t1 - t0) + 0.15 * ((t1 - offset)**2 - (t0 - offset)**2)
        delta = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        pairs.append((t0, t1, Tbc[:3, :3].T @ delta.T @ Tbc[:3, :3]))
    result = rotation_sync_check(pairs, times, omega, Tbc)
    assert result["additional_imu_offset_ms_candidate"] == pytest.approx(40)
    assert result["median_rotation_residual_deg"] < 1e-6
    assert result["applied_automatically"] is False
    assert rotation_sync_check([], times, omega, Tbc)["status"] == "inconclusive"


def write_rows(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def trajectory_fixture(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(json.dumps({
        "orb_slam3_commit": ORB_SLAM3_COMMIT, "pose_frame": "scene_camera",
        "imu_noise_source": "synthetic", "extrinsics_source": "synthetic",
    }))
    times = 1_790_000_000_000_000_123 + np.arange(8, dtype=np.int64) * 100_000_003
    seconds = relative_seconds(times, int(times[0]))
    write_rows(prepared / "frames.csv", ["frame_id", "time_ns", "slam_time_s"],
               [[i + 10, int(t), format(s, ".17g")] for i, (t, s) in enumerate(zip(times, seconds))])
    trajectory = tmp_path / "trajectory.csv"
    states = tmp_path / "states.csv"
    # Camera has a nonidentity orientation. Conversion must preserve it and Z.
    q = Rotation.from_euler("x", 40, degrees=True).as_quat().tolist()
    write_rows(trajectory, ["slam_time_s", "map_id", "metric", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
               [[format(s, ".17g"), 0 if i < 6 else 1, 1, 100 + i, 200, 300 + i] + q
                for i, s in enumerate(seconds)])
    write_rows(states, ["frame_id", "slam_time_s", "state", "imu_initialized", "map_id"],
               [[i + 10, format(s, ".17g"), 3 if i == 3 else 2, 0 if i == 0 else 1, 0 if i < 6 else 1]
                for i, s in enumerate(seconds)])
    return prepared, trajectory, states, times, q


def test_conversion_excludes_uninitialized_and_lost_preserves_axes(trajectory_fixture, tmp_path):
    prepared, trajectory, states, times, q = trajectory_fixture
    out = tmp_path / "converted"
    report = convert(prepared, trajectory, states, out, min_frames=2)
    assert [(s["start_n"], s["max_n"]) for s in report["segments"]] == [(11, 2), (14, 2), (16, 2)]
    assert report["metric_tracked_frames"] == 6 and report["interpolated_frames"] == 0
    assert report["tracking_state_counts"] == {"2": 7, "3": 1}
    assert report["runtime_map_count"] == 2
    with np.load(out / report["segments"][0]["file"]) as geo:
        np.testing.assert_array_equal(geo["timestamps_ns"], times[1:3])
        np.testing.assert_allclose(geo["T_world_camera"][0, :3, 3], [0, 0, 0])
        np.testing.assert_allclose(geo["T_world_camera"][1, :3, 3], [1, 0, 1])
        np.testing.assert_allclose(geo["T_world_camera"][0, :3, :3], Rotation.from_quat(q).as_matrix())
        assert "points_world" not in geo


def test_conversion_refuses_duplicate_and_nonmetric(trajectory_fixture, tmp_path):
    prepared, trajectory, states, _, _ = trajectory_fixture
    data = trajectory.read_text()
    trajectory.write_text(data + data.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        convert(prepared, trajectory, states, tmp_path / "duplicate", 2)
    trajectory.write_text(data)
    state_text = states.read_text()
    states.write_text("\n".join([state_text.splitlines()[0]] +
                      [line.replace(",2,1,", ",2,0,") for line in state_text.splitlines()[1:]]) + "\n")
    with pytest.raises(ValueError, match="No continuous metric"):
        convert(prepared, trajectory, states, tmp_path / "nonmetric", 2)
    assert json.loads((tmp_path / "nonmetric/report.json").read_text())["segments"] == []
