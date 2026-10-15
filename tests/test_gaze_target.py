"""Gaze/OBB intersections, coordinate invariance and per-frame runner output."""

import csv
import json
import sys

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from test_neon_loader import make_geometry, recording
from utils.gaze_target import estimate_gaze_target, ray_obb_intersections
from utils.neon_gaze import NeonGaze
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor


EPOCH = 1_790_000_000_000_000_123


def boxes(centres, probabilities=None):
    poses = np.tile(np.eye(4), (len(centres), 1, 1))
    poses[:, :3, 3] = centres
    return ObbTW.from_lmc(
        bb3_object=torch.tensor([[-1., 1., -1., 1., -1., 1.]] * len(centres)),
        T_world_object=PoseTW.from_matrix(torch.tensor(poses, dtype=torch.float32)),
        prob=torch.tensor(probabilities or [0.9] * len(centres))[:, None],
        text=torch.stack([string2tensor(pad_string(f"object_{i}", 128))
                          for i in range(len(centres))]),
    )


def gaze():
    K = np.array([[100., 0., 80.], [0., 100., 60.], [0., 0., 1.]])
    return NeonGaze(np.array([EPOCH]), [[80., 60.]]).sample(
        7, EPOCH, K, np.zeros(5), K, (160, 120), (160, 120),
    )


def test_nearest_forward_surface_and_no_mutation():
    obbs = boxes([[0, 0, 8], [0, 0, 4], [0, 0, -4]], [0.99, 0.6, 0.9])
    before = obbs.clone()
    result = estimate_gaze_target(gaze(), obbs, np.eye(4), "per_frame_assumed_upright")
    assert result["detection_index"] == 1
    assert result["intersection_count"] == 2
    assert result["label"] == "object_1"
    assert result["distance_m"] == pytest.approx(3.)
    assert result["position_camera_z"] == pytest.approx(3.)
    assert result["box_confidence"] == pytest.approx(0.6)
    assert result["coordinate_frame"] == "per_frame_assumed_upright"
    assert not result["point_cloud_used"]
    assert torch.equal(before._data, obbs._data)


def test_rigid_transform_invariance_and_world_coordinates():
    obbs = boxes([[0, 0, 5]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [25, -15, 70], degrees=True).as_matrix()
    transform[:3, 3] = [10, -3, 2]
    moved = obbs.clone()
    moved.set_T_world_object(PoseTW.from_matrix(
        torch.tensor(transform, dtype=torch.float32) @ obbs.T_world_object.matrix
    ))
    result = estimate_gaze_target(gaze(), moved, transform, "world_z_up", True)
    camera = [result[f"position_camera_{a}"] for a in "xyz"]
    frame = [result[f"position_frame_{a}"] for a in "xyz"]
    np.testing.assert_allclose(camera, [0, 0, 4], atol=1e-6)
    np.testing.assert_allclose(frame, transform[:3, 3] + 4 * transform[:3, 2], atol=1e-6)
    assert result["point_cloud_used"]


def test_rotated_box_nonunit_ray_and_noncentred_bounds():
    pose = np.eye(4)[None]
    pose[0, :3, :3] = Rotation.from_euler("y", 45, degrees=True).as_matrix()
    pose[0, :3, 3] = [0, 0, 5]
    bounds = np.array([[-1., 1., -1., 1., -1., 1.]])
    distance, _ = ray_obb_intersections([0, 0, 0], [0, 0, 10], bounds, pose)
    assert distance[0] == pytest.approx(5 - np.sqrt(2))
    # Object origin need not coincide with its bounding-box centre.
    distance, _ = ray_obb_intersections(
        [0, 0, 0], [0, 0, 1], [[-1, 1, -1, 1, 4, 6]], np.eye(4)[None]
    )
    assert distance[0] == 4


@pytest.mark.parametrize("origin,direction,expected", [
    ([0, 0, 0], [0, 0, 1], 4),
    ([2, 0, 0], [0, 0, 1], np.inf),  # parallel outside one slab
    ([1, 0, 0], [0, 0, 1], 4),  # tangent boundary, no divide-by-zero
    ([0, 0, 0], [0, 0, -1], np.inf),
    ([0, 0, 5], [0, 0, 1], 1),  # inside uses the exit surface
])
def test_parallel_behind_inside_and_boundary_rays(origin, direction, expected):
    obbs = boxes([[0, 0, 5]])
    distances, inside = ray_obb_intersections(
        origin, direction, obbs.bb3_object.numpy(), obbs.T_world_object.matrix.numpy()
    )
    assert distances[0] == expected
    assert bool(inside[0]) == (origin == [0, 0, 5])


def test_invalid_boxes_are_ignored_and_zero_ray_rejected():
    obbs = boxes([[0, 0, 3], [0, 0, 5], [0, 0, 7]])
    bounds = obbs.bb3_object.numpy().copy()
    bounds[0] = np.nan
    poses = obbs.T_world_object.matrix.numpy().copy()
    poses[1, :3, :3] *= 2  # a similarity is not a rigid pose
    distances, _ = ray_obb_intersections([0, 0, 0], [0, 0, 1], bounds, poses)
    np.testing.assert_array_equal(distances, [np.inf, np.inf, 6.])
    with pytest.raises(ValueError, match="nonzero"):
        ray_obb_intersections([0, 0, 0], [0, 0, 0], bounds, poses)


def test_missing_gaze_boxes_and_intersections_leave_positions_empty():
    obbs = boxes([[0, 0, 5]])
    row = gaze()
    assert estimate_gaze_target(row, obbs, np.eye(4), "world_z_up")["status"] == "estimated"
    missing = dict(row, status="missing_gaze")
    cases = [(missing, obbs, "missing_gaze"), (row, None, "no_boxes"),
             (row, boxes([[10, 0, 5]]), "no_intersection"),
             (dict(row, in_image=False), obbs, "outside_image")]
    for sample, candidates, status in cases:
        target = estimate_gaze_target(sample, candidates, np.eye(4), "world_z_up")
        assert target["status"] == status
        assert target["position_frame_x"] == target["distance_m"] == ""


@pytest.mark.parametrize("point_mode", ["off", "auto", "on"])
def test_runner_writes_every_frame_with_or_without_points(recording, tmp_path, monkeypatch, point_mode):
    import run_boxer
    import owl.owl_wrapper

    root, times = recording
    pixels = np.tile(np.array([[32, 24]], dtype="<f4"), (5, 1))
    pixels[3] = np.nan
    pixels.tofile(root / "gaze ps1.raw")
    times.tofile(root / "gaze ps1.time")
    geometry = tmp_path / "geometry.npz"
    make_geometry(geometry, times, np.array([[2., 7., 1.]], dtype=np.float32))

    class Detector:
        def __init__(self, *args, **kwargs):
            self.frame = -1

        def forward(self, image, **kwargs):
            self.frame += 1
            n = 0 if self.frame == 1 else 1
            return torch.tensor([[1., 30., 1., 30.]]).repeat(n, 1), torch.ones(n), [0]*n, None

    class Model(torch.nn.Module):
        hw = 32

        def forward(self, datum):
            # Known geometry verifies integration, independent of model accuracy.
            x = 10 if datum["frame_id"] == 2 else 0
            obbs = boxes([[x, 0, 5]])
            obbs.set_T_world_object(datum["T_world_rig0"] @ obbs.T_world_object)
            if datum["frame_id"] == 4:
                obbs.set_prob(torch.tensor([[0.1]]))
            if point_mode == "off":
                assert datum["sdp_w"].numel() == 0
            else:
                assert datum["sdp_w"].numel() == 3
            return {"obbs_pr_w": obbs.unsqueeze(0)}

    monkeypatch.setattr(owl.owl_wrapper, "OwlWrapper", Detector)
    monkeypatch.setattr(run_boxer.BoxerNet, "load_from_checkpoint", lambda *a, **k: Model())
    out = tmp_path / "result"
    args = ["run_boxer.py", "--input", str(root), "--start_n", "0", "--max_n", "5",
            "--force_cpu", "--skip_viz", "--labels", "chair", "--gaze_targets",
            "--point_cloud", point_mode, "--output_dir", str(out)]
    # Test the actual no-trajectory path when no point cloud is requested.
    if point_mode != "off":
        args += ["--neon_geometry", str(geometry)]
    monkeypatch.setattr(sys, "argv", args)
    run_boxer.main()
    with (out / "neon/boxer_gaze_targets.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(r["time_ns"]) for r in rows] == times.tolist()
    assert [r["status"] for r in rows] == [
        "estimated", "no_boxes", "no_intersection", "invalid_gaze", "no_boxes",
    ]
    assert float(rows[0]["position_camera_z"]) == pytest.approx(4.)
    assert rows[0]["coordinate_frame"] == (
        "per_frame_assumed_upright" if point_mode == "off" else "world_z_up"
    )
    assert rows[0]["point_cloud_used"] == str(point_mode != "off")
    metadata = json.loads((out / "neon/boxer_input.json").read_text())
    assert metadata["gaze_targets"]["temporal_fixation_detection"] is False
    # no_csv must still allow estimation while emitting no CSV/metadata files.
    no_csv_out = tmp_path / "no_csv"
    args[args.index("--output_dir") + 1] = str(no_csv_out)
    monkeypatch.setattr(sys, "argv", args + ["--no_csv"])
    run_boxer.main()
    assert list((no_csv_out / "neon").iterdir()) == []
