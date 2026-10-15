"""Replay uses source IDs, exact clocks and only registered world geometry."""

import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_neon_loader import make_geometry
from test_neon_loader import recording as native_recording_fixture

from utils.neon_viewer import build_neon_context, resolve_neon_recording
from utils.tw.obb import ObbTW
from utils.viewer_3d import SequenceOBBViewer, TrackerViewer, resolve_input

recording = native_recording_fixture


@pytest.fixture
def replay(recording, tmp_path):
    root, times = recording
    geometry = tmp_path / "geometry.npz"
    make_geometry(geometry, times[[1, 3]], np.array([[2, 7, 1]], dtype=np.float32))
    log = tmp_path / "results"
    log.mkdir()
    (log / "boxer_input.json").write_text(
        json.dumps(
            {
                "dataset": "neon",
                "has_world_poses": True,
                "coordinate_frame": "world_z_up",
                "image_size": 32,
            }
        )
    )
    (log / "boxer_frames.csv").write_text(
        f"frame_id,time_ns\n1,{times[1]}\n3,{times[3]}\n"
    )
    return root, times, geometry, log


def test_resolve_native_recording_and_parent(recording):
    root, _ = recording
    assert resolve_neon_recording(root) == root
    assert resolve_neon_recording(root.parent) == root
    assert resolve_input(str(root.parent)) == (str(root), "neon", root.name)
    other = root.parent / "other"
    other.mkdir()
    (other / "Neon Scene Camera v1 ps1.mp4").touch()
    with pytest.raises(ValueError, match="Multiple Neon"):
        resolve_neon_recording(root.parent)


def test_replay_exact_ids_empty_frames_images_and_static_points(replay):
    root, times, geometry, log = replay
    detection = ObbTW(torch.zeros(1, 165))
    ctx, timeline = build_neon_context(
        root, geometry, log, "boxer", {int(times[1]): detection}
    )
    try:
        np.testing.assert_array_equal(ctx["source_frame_ids"], [1, 3])
        np.testing.assert_array_equal(ctx["pose_ts"], times[[1, 3]])
        assert ctx["pose_ts"].dtype == np.int64
        assert list(timeline) == list(times[[1, 3]])
        assert timeline[int(times[1])] is detection
        assert len(timeline[int(times[3])]) == 0
        frame = ctx["loader"].load(1)
        assert frame["frame_id"] == 3 and frame["time_ns0"] == int(times[3])
        assert tuple(frame["img0"].shape) == (1, 3, 32, 32)
        assert ctx["calibs"][1].f[0].item() == pytest.approx(25)
        assert len(ctx["uid_to_p3"]) == 1
        assert ctx["time_to_uids_slaml"] == ctx["time_to_uids_slamr"] == {}
        viewer = SimpleNamespace(
            _data_source="neon",
            traj=ctx["traj"],
            calibs=ctx["calibs"],
            pose_ts=ctx["pose_ts"],
            calib_ts=ctx["calib_ts"],
            _rgb_timestamps=ctx["rgb_timestamps"],
            _loader=ctx["loader"],
            _rgb_lru_cache=OrderedDict(),
            _rgb_lru_max_items=2,
        )
        cam, pose = SequenceOBBViewer._get_cam_and_pose(viewer, int(times[3]))
        assert cam is not None and pose is not None
        assert SequenceOBBViewer._get_cam_and_pose(viewer, int(times[3]) + 1) == (
            None,
            None,
        )
        assert (
            SequenceOBBViewer._load_rgb_for_timestamp(viewer, int(times[3]) + 1) is None
        )
        image = SequenceOBBViewer._load_rgb_for_timestamp(viewer, int(times[3]))
        assert image.shape == (1200, 1200, 3)
        assert viewer._rgb_vrs_w == viewer._rgb_vrs_h == 32
        assert list(viewer._rgb_lru_cache) == [int(times[3])]
    finally:
        ctx["loader"].close()


@pytest.mark.parametrize(
    "problem",
    ["missing_geometry", "local", "clock", "pose_gap", "extra_detection", "bad_ids"],
)
def test_invalid_replay_is_rejected(replay, problem):
    root, times, geometry, log = replay
    detections = {}
    if problem == "missing_geometry":
        geometry = None
    elif problem == "local":
        path = log / "boxer_input.json"
        meta = json.loads(path.read_text())
        meta["coordinate_frame"] = "per_frame_assumed_upright"
        path.write_text(json.dumps(meta))
    elif problem == "clock":
        (log / "boxer_frames.csv").write_text(f"frame_id,time_ns\n1,{times[1] + 1}\n")
    elif problem == "pose_gap":
        make_geometry(geometry, times[[1]])
    elif problem == "extra_detection":
        detections[int(times[2])] = ObbTW(torch.zeros(0, 165))
    else:
        (log / "boxer_frames.csv").write_text(
            f"frame_id,time_ns\n3,{times[3]}\n1,{times[1]}\n"
        )
    with pytest.raises(ValueError):
        build_neon_context(root, geometry, log, "boxer", detections)


def test_per_frame_points_keep_observation_times(replay):
    root, times, geometry, log = replay
    points = np.array([[[2, 7, 1]], [[4, 7, 1]]], dtype=np.float32)
    make_geometry(geometry, times[[1, 3]], points)
    ctx, _ = build_neon_context(root, geometry, log, "boxer", {})
    try:
        assert ctx["time_to_uids_slaml"] == {int(times[1]): [0], int(times[3]): [1]}
        viewer = SimpleNamespace(
            _data_source="neon",
            _point_positions=points[:, 0],
            _time_to_uids_slaml=ctx["time_to_uids_slaml"],
            _time_to_uids_slamr={},
            _all_obs_timestamps=times[[1, 3]],
            _uid_to_idx={0: 0, 1: 1},
        )
        np.testing.assert_array_equal(
            TrackerViewer._get_observed_points(viewer, int(times[3])).numpy(), points[1]
        )
        assert TrackerViewer._get_observed_points(viewer, int(times[3]) + 1) is None
    finally:
        ctx["loader"].close()
