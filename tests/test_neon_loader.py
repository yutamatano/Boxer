"""Neon input contracts using a small synthetic native recording (no private data)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxernet.boxernet import sdp_to_patches
from loaders.neon_loader import NeonLoader

Calibration = pytest.importorskip("pupil_labs.neon_recording.calib").Calibration


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "neon"
    root.mkdir()
    (root / "info.json").write_text(json.dumps({"data_format_version": "2.6"}))
    calibration = np.zeros(1, dtype=Calibration.dtype)
    calibration["version"] = 1
    calibration["scene_camera_matrix"] = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    calibration.tofile(root / "calibration.bin")
    video = root / "Neon Scene Camera v1 ps1.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (10 + i * 20, 30, 200)
        writer.write(frame)
    writer.release()
    # Preserve individual nanoseconds beyond float64's exact-integer range.
    times = 1_790_000_000_000_000_123 + np.arange(5, dtype=np.int64) * 100_000_003
    times.tofile(video.with_suffix(".time"))
    (times - 1000).tofile(video.with_suffix(".time_aux"))
    return root, times


def make_geometry(path, times, points=None):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    pose[:3, 3] = [2, 3, 1]
    data = {
        "timestamps_ns": times,
        "T_world_camera": np.repeat(pose[None], len(times), axis=0),
    }
    if points is not None:
        data["points_world"] = points
    np.savez(path, **data)
    return data


def test_native_frames_resize_color_and_exact_timestamps(recording, tmp_path):
    root, times = recording
    loader = NeonLoader(root, start_n=1, skip_n=2, max_n=2, resize=32)
    try:
        loader._init_prefetch()
        frames = list(loader)
        assert len(frames) == 2
        assert [d["frame_id"] for d in frames] == [1, 3]
        assert [d["time_ns0"] for d in frames] == times[[1, 3]].tolist()
        frame = frames[0]
        assert frame["img0"].shape == (1, 3, 32, 32)
        # Red in the source BGR video must stay red in the RGB tensor.
        assert frame["img0"][0, 0].mean() > frame["img0"][0, 2].mean() + 0.4
        assert frame["sdp_w"].shape == (0, 3)
        assert frame["observed_points"] is None
        assert not loader.has_world_poses
        assert frame["cam0"].f[0].item() == pytest.approx(25)
        assert frame["cam0"].c[0].item() == pytest.approx(15.75)
        patches = sdp_to_patches(
            frame["sdp_w"][None], frame["cam0"][None],
            frame["T_world_rig0"][None], 32, 32, 16,
        )
        assert torch.equal(patches, torch.full((1, 1, 2, 2), -1.0))
        loader.write_metadata(tmp_path, "test")
        saved = np.loadtxt(
            tmp_path / "test_frames.csv", dtype=np.int64, delimiter=",", skiprows=1
        )
        np.testing.assert_array_equal(saved[:, 1], times[[1, 3]])
        np.testing.assert_array_equal(saved[:, 2], times[[1, 3]] - 1000)
        meta = json.loads((tmp_path / "test_input.json").read_text())
        assert meta["coordinate_frame"] == "per_frame_assumed_upright"
        assert meta["point_cloud"] == "off"
    finally:
        loader.close()


def test_external_world_geometry_projects_metric_depth(recording, tmp_path):
    root, times = recording
    path = tmp_path / "geometry.npz"
    # Camera origin (2,3,1), forward is world +Y. This point is 4 m ahead.
    points = np.tile([[[2, 7, 1], [np.nan, np.nan, np.nan]]], (5, 1, 1))
    make_geometry(path, times, points)
    loader = NeonLoader(root, max_n=1, resize=32, point_cloud="on", geometry_path=path)
    try:
        datum = next(loader)
        assert loader.has_world_poses and loader.use_points
        assert datum["sdp_w"].shape == (1, 3)
        assert datum["observed_points"] is not None
        patches = sdp_to_patches(
            datum["sdp_w"][None], datum["cam0"][None],
            datum["T_world_rig0"][None], 32, 32, 16,
        )
        assert patches.max().item() == pytest.approx(4)
        assert (patches > 0).sum().item() == 1
    finally:
        loader.close()
    # Same external file, but point input disabled; poses remain available.
    loader = NeonLoader(root, max_n=1, point_cloud="off", geometry_path=path)
    try:
        datum = next(loader)
        assert loader.has_world_poses and not loader.use_points
        assert datum["sdp_w"].shape == (0, 3)
        assert datum["observed_points"] is None
    finally:
        loader.close()


def test_static_map_is_not_current_observation(recording, tmp_path):
    root, times = recording
    path = tmp_path / "map.npz"
    make_geometry(path, times, np.array([[2, 7, 1]], dtype=np.float32))
    loader = NeonLoader(root, max_n=1, geometry_path=path)
    try:
        datum = next(loader)
        assert datum["sdp_w"].shape == (1, 3)
        assert datum["observed_points"] is None
    finally:
        loader.close()


def test_required_points_fail_instead_of_silent_monocular_fallback(recording):
    root, _ = recording
    with pytest.raises(ValueError, match="requires --neon_geometry"):
        NeonLoader(root, point_cloud="on")


@pytest.mark.parametrize("problem", ["clock", "float_time", "pose", "points"])
def test_invalid_geometry_rejected(recording, tmp_path, problem):
    root, times = recording
    path = tmp_path / "invalid.npz"
    data = make_geometry(path, times)
    if problem == "clock":
        data["timestamps_ns"] = times + 1
    elif problem == "float_time":
        data["timestamps_ns"] = times.astype(np.float64)
    elif problem == "pose":
        data["T_world_camera"][0, 0, 0] = 2
    else:
        data["points_world"] = np.ones((5, 3, 4))
    np.savez(path, **data)
    with pytest.raises(ValueError):
        NeonLoader(root, geometry_path=path)


def test_missing_aux_does_not_change_primary_clock(recording):
    root, times = recording
    (root / "Neon Scene Camera v1 ps1.time_aux").unlink()
    loader = NeonLoader(root, max_n=1)
    try:
        assert next(loader)["time_ns0"] == int(times[0])
        assert (loader.aux_timestamps_ns == -1).all()
    finally:
        loader.close()


def test_truncated_timestamps_rejected(recording):
    root, times = recording
    # The official reader can truncate its stream to the video length; reject
    # such count mismatches rather than attaching the wrong times to images.
    np.append(times, times[-1] + 100_000_000).astype(np.int64).tofile(
        root / "Neon Scene Camera v1 ps1.time"
    )
    with pytest.raises(ValueError, match="Timestamp count mismatch"):
        NeonLoader(root)


@pytest.mark.parametrize("flag", ["--track", "--fuse"])
def test_cli_requires_trajectory_before_loading_models(recording, flag):
    root, _ = recording
    result = subprocess.run(
        [sys.executable, "run_boxer.py", "--input", str(root), flag],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires --neon_geometry" in result.stderr


def test_distortion_and_resize_use_matching_camera(recording):
    root, _ = recording
    calibration = np.fromfile(root / "calibration.bin", dtype=Calibration.dtype)
    calibration["scene_distortion_coefficients"][0, 0] = 0.2
    calibration.tofile(root / "calibration.bin")
    loader = NeonLoader(root, max_n=1, resize=32)
    try:
        datum = next(loader)
        # A known camera-space ray projects to the rectified location, while the
        # remap samples its distorted source location from the native video.
        point = np.array([[[0.3, 0.2, 1.0]]], dtype=np.float64)
        uv, _ = cv2.projectPoints(
            point, np.zeros(3), np.zeros(3), loader.K, loader.distortion
        )
        rectified, valid = datum["cam0"][None].project(torch.from_numpy(point).float())
        assert valid.all()
        expected = cv2.undistortPoints(
            uv, loader.K, loader.distortion, P=loader.K
        ).reshape(2)
        expected = (expected + 0.5) * [0.5, 32 / 48] - 0.5
        np.testing.assert_allclose(rectified.numpy()[0, 0], expected, atol=1e-4)
        # Check the image remap against an independently distorted output ray.
        normalized = (
            np.array([24, 22]) - datum["cam0"].c.numpy()
        ) / datum["cam0"].f.numpy()
        source, _ = cv2.projectPoints(
            np.array([[*normalized, 1.0]]), np.zeros(3), np.zeros(3),
            loader.K, loader.distortion,
        )
        np.testing.assert_allclose(
            [loader._maps[0][22, 24], loader._maps[1][22, 24]],
            source.reshape(2), atol=1e-4,
        )
    finally:
        loader.close()


def test_multipart_boundary_retains_frame_time_pairing(recording):
    import shutil

    root, times = recording
    shutil.copyfile(
        root / "Neon Scene Camera v1 ps1.mp4", root / "Neon Scene Camera v1 ps2.mp4"
    )
    part2 = times + 500_000_015
    part2.tofile(root / "Neon Scene Camera v1 ps2.time")
    (part2 - 1000).tofile(root / "Neon Scene Camera v1 ps2.time_aux")
    loader = NeonLoader(root, start_n=4, max_n=3)
    try:
        frames = list(loader)
        assert [d["frame_id"] for d in frames] == [4, 5, 6]
        assert [d["time_ns0"] for d in frames] == [times[4], part2[0], part2[1]]
    finally:
        loader.close()
