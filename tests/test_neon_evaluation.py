"""Accuracy evaluation must preserve scale and reject approximate clock matches."""

import csv
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evaluate_neon_geometry import evaluate, reference_error


def test_reference_alignment_does_not_hide_scale_error(tmp_path):
    stamps = 1_790_000_000_000_000_001 + np.arange(3, dtype=np.int64) * 100_000_000
    estimated = np.tile(np.eye(4), (3, 1, 1))
    estimated[:, 0, 3] = [0, 1, 2]
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    reference = estimated.copy()
    reference[:, :3, :3] = rotation
    reference[:, :3, 3] = estimated[:, :3, 3] @ rotation.T + [10, 20, 30]
    path = tmp_path / "reference.npz"
    np.savez(path, timestamps_ns=stamps, T_world_camera=reference)
    assert reference_error(stamps, estimated, path)["position_rmse_m"] < 1e-12
    scaled = estimated.copy()
    scaled[:, :3, 3] *= 2
    assert reference_error(stamps, scaled, path)["position_rmse_m"] == pytest.approx(
        np.sqrt(5 / 3)
    )
    with pytest.raises(ValueError, match="exact source timestamp"):
        reference_error(stamps + 1, estimated, path)


def test_internal_imu_check_does_not_claim_reference_accuracy(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    manifest = {
        "T_imu_camera": np.eye(4).tolist(),
        "K_rectified": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "image_size": [100, 100],
        "imu_offset_ms_applied": 0,
        "imu_noise_source": "synthetic",
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    stamps = 1_790_000_000_000_000_003 + np.arange(3, dtype=np.int64) * 1_000_000_000
    with (prepared / "frames.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_id", "time_ns", "slam_time_s"])
        writer.writerows([[10 + i, int(t), i] for i, t in enumerate(stamps)])
    imu = np.zeros((23, 7))
    imu[:, 0] = np.linspace(-0.1, 2.1, 23)
    np.savetxt(
        prepared / "imu.csv",
        imu,
        delimiter=",",
        header="time,gx,gy,gz,ax,ay,az",
        comments="",
    )
    geometry = tmp_path / "geometry.npz"
    np.savez(
        geometry,
        timestamps_ns=stamps,
        T_world_camera=np.tile(np.eye(4), (3, 1, 1)),
        points_world=[[0, 0, 4], [0, 0, -1]],
    )
    report = evaluate(prepared, geometry, tmp_path / "evaluation.json")
    assert report["gyro_rotation_residual_deg"]["max"] == 0
    assert report["projected_points_in_image"]["p50"] == 1
    assert "not measured" in report["reference_accuracy"]["status"]
    assert "not independent accuracy" in report["gyro_comparison"]
