"""Coordinate, clock and missing-data contracts for Neon gaze alignment."""

import csv

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils.neon_gaze import GAZE_FIELDS, GazeCsvWriter, NeonGaze, draw_gaze
from utils.tag_aligner import AlignedPoses, POSE_FIELDS, relative_seconds_to_ns


EPOCH = 1_790_000_000_000_000_123
K = np.array([[100., 0., 80.], [0., 110., 60.], [0., 0., 1.]])


def write_poses(path, intervals, translation=(1000., 2000., 3000.), quat=(0., 0., 0., 1.)):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(POSE_FIELDS)
        for start, end in intervals:
            writer.writerow([start, end, *translation, *quat])


def sample(gaze, time=EPOCH, **overrides):
    args = dict(frame_id=7, time_ns=time, K=K, distortion=np.zeros(5), K_out=K,
                raw_size=(160, 120), out_size=(160, 120))
    args.update(overrides)
    return gaze.sample(**args)


def test_epoch_precision_and_half_open_pose_gaps(tmp_path):
    path = tmp_path / "aligned_poses.csv"
    write_poses(path, [("0.000000001", "0.100000003"), ("0.2", "0.3")])
    poses = AlignedPoses(path, EPOCH, 0.001)
    assert relative_seconds_to_ns("0.000000001", EPOCH) == EPOCH + 1
    times = np.array([EPOCH, EPOCH+1, EPOCH+100000002, EPOCH+100000003,
                      EPOCH+199999999, EPOCH+200000000, EPOCH+300000000], dtype=np.int64)
    np.testing.assert_array_equal(poses.match(times), [-1, 0, 0, -1, -1, 1, -1])
    np.testing.assert_allclose(poses.matrices[0, :3, 3], [1, 3, -2])


def test_pose_rotation_is_camera_to_world_and_scale_is_only_applied_once(tmp_path):
    path = tmp_path / "aligned_poses.csv"
    write_poses(path, [(0, 1)], quat=Rotation.from_euler("y", 90, degrees=True).as_quat())
    pose = AlignedPoses(path, EPOCH, 0.001).matrices[0]
    gaze = NeonGaze(np.array([EPOCH]), [[80, 60]])
    row = sample(gaze, T_world_camera=pose)
    assert row["status"] == "world_ray"
    np.testing.assert_allclose([row[f"origin_world_{a}"] for a in "xyz"], [1, 3, -2])
    np.testing.assert_allclose([row[f"direction_world_{a}"] for a in "xyz"], [1, 0, 0], atol=1e-14)


def test_distortion_resize_and_direction_roundtrip():
    distortion = np.array([0.1, -0.02, 0.001, -0.003, 0.001])
    camera_point = np.array([0.2, -0.1, 1.])
    pixel = cv2.projectPoints(camera_point[None], np.zeros(3), np.zeros(3), K, distortion)[0]
    gaze = NeonGaze(np.array([EPOCH]), pixel.reshape(1, 2))
    resized = K.copy()
    resized[:2] *= 0.5
    resized[:2, 2] -= 0.25
    row = sample(gaze, distortion=distortion, K_out=resized, out_size=(80, 60))
    projected = resized @ camera_point
    np.testing.assert_allclose([row["image_x"], row["image_y"]], projected[:2], atol=1e-6)
    np.testing.assert_allclose([row[f"direction_camera_{a}"] for a in "xyz"],
                               camera_point / np.linalg.norm(camera_point), atol=1e-6)
    assert row["origin_world_x"] == ""
    assert row["status"] == "camera_only"


def test_nearest_sample_bound_tie_and_no_carry_forward():
    gaze = NeonGaze(np.array([EPOCH, EPOCH+20_000_000]), [[80, 60], [90, 70]])
    row = sample(gaze, EPOCH+10_000_000)
    assert row["gaze_time_ns"] == EPOCH  # ties select the earlier sample
    assert row["gaze_delta_ns"] == -10_000_000
    assert sample(gaze, EPOCH+30_000_001)["status"] == "missing_gaze"
    assert sample(gaze, EPOCH-10_000_001)["status"] == "missing_gaze"


@pytest.mark.parametrize("pixel,status", [([np.nan, 60], "invalid_gaze"),
                                          ([np.inf, 60], "invalid_gaze"),
                                          ([-1, 60], "outside_scene")])
def test_invalid_gaze_is_not_a_world_measurement(pixel, status):
    row = sample(NeonGaze(np.array([EPOCH]), [pixel]), T_world_camera=np.eye(4))
    assert row["status"] == status
    assert row["direction_world_x"] == ""
    assert not row["in_image"]


def test_csv_preserves_ns_and_missing_rows(tmp_path):
    path = tmp_path / "gaze.csv"
    gaze = NeonGaze(np.array([EPOCH]), [[80, 60]])
    writer = GazeCsvWriter(path)
    try:
        writer.write(sample(gaze))
        writer.write(sample(gaze, EPOCH+100_000_000))
    finally:
        writer.close()
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == GAZE_FIELDS
    assert int(rows[0]["time_ns"]) == EPOCH
    assert int(rows[0]["gaze_time_ns"]) == EPOCH
    assert rows[1]["gaze_time_ns"] == ""
    assert rows[1]["status"] == "missing_gaze"


def test_overlay_only_draws_valid_image_samples():
    row = sample(NeonGaze(np.array([EPOCH]), [[80, 60]]))
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    assert draw_gaze(image.copy(), row).any()
    assert not image.any()
    row["in_image"] = False
    assert not draw_gaze(image.copy(), row).any()


@pytest.mark.parametrize("intervals", [[(0, 0)], [(1, 2), (0, 1)], [(0, 2), (1, 3)]])
def test_invalid_pose_intervals_rejected(tmp_path, intervals):
    path = tmp_path / "poses.csv"
    write_poses(path, intervals)
    with pytest.raises(ValueError, match="intervals"):
        AlignedPoses(path, EPOCH, 1)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "bad"])
def test_invalid_relative_time_rejected(value):
    with pytest.raises(ValueError):
        relative_seconds_to_ns(value, EPOCH)


def test_invalid_scale_rotation_and_gaze_clock_rejected(tmp_path):
    path = tmp_path / "poses.csv"
    write_poses(path, [(0, 1)])
    for scale in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="meters_per_unit"):
            AlignedPoses(path, EPOCH, scale)
    write_poses(path, [(0, 1)], quat=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="quaternions"):
        AlignedPoses(path, EPOCH, 1)
    with pytest.raises(ValueError, match="increasing int64"):
        NeonGaze(np.array([EPOCH, EPOCH]), [[80, 60], [80, 60]])
    with pytest.raises(ValueError, match="increasing int64"):
        NeonGaze(np.array([float(EPOCH)]), [[80, 60]])
