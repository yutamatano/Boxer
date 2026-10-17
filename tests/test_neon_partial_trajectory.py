"""All-frame Neon detection/replay with only registered frames in the world."""

import csv
import json
import sys

import numpy as np
import pytest
import torch
from test_gaze_target import boxes
from test_neon_loader import make_geometry
from test_neon_loader import recording as native_recording_fixture

from loaders.neon_loader import NeonLoader
from utils.file_io import read_obb_csv, save_bb2d_csv
from utils.gaze_target import GazeTargetCsvWriter, estimate_gaze_target
from utils.neon_gaze import GazeCsvWriter
from utils.neon_viewer import build_neon_context

recording = native_recording_fixture


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_partial_geometry_keeps_all_frames_and_only_registered_points(
    recording, tmp_path, mode
):
    root, times = recording
    geometry = tmp_path / "partial.npz"
    make_geometry(geometry, times[1:3], np.array([[2.0, 7.0, 1.0]], dtype=np.float32))
    loader = NeonLoader(root, resize=32, geometry_path=geometry, point_cloud=mode)
    try:
        frames = list(loader)
        assert [d["frame_id"] for d in frames] == list(range(5))
        assert [d["time_ns0"] for d in frames] == times.tolist()
        assert [d["has_world_pose"] for d in frames] == [
            False,
            True,
            True,
            False,
            False,
        ]
        assert [d["sdp_w"].shape[0] for d in frames] == (
            [0] * 5 if mode == "off" else [0, 1, 1, 0, 0]
        )
        for d in frames:
            assert d["coordinate_frame"] == (
                "world_z_up" if d["has_world_pose"] else "per_frame_assumed_upright"
            )
            if not d["has_world_pose"]:
                assert torch.equal(d["T_world_rig0"].t, torch.zeros(3))
        loader.write_metadata(tmp_path, "boxer")
    finally:
        loader.close()
    metadata = json.loads((tmp_path / "boxer_input.json").read_text())
    assert metadata["coordinate_frame"] == "mixed_world_and_per_frame"
    assert metadata["world_pose_frames"] == 2
    # The result remembers its geometry: replay needs no frame-range selection.
    context, timeline = build_neon_context(root, None, tmp_path, "boxer", {})
    try:
        assert list(timeline) == times.tolist()
        assert all(len(obbs) == 0 for obbs in timeline.values())
        assert context["world_pose_by_time"] == dict(
            zip(times.tolist(), [False, True, True, False, False])
        )
    finally:
        context["loader"].close()


def test_replay_rejects_changed_pose_coverage(recording, tmp_path):
    root, times = recording
    geometry = tmp_path / "partial.npz"
    make_geometry(geometry, times[1:3])
    loader = NeonLoader(root, resize=32, geometry_path=geometry)
    loader.write_metadata(tmp_path, "boxer")
    loader.close()
    make_geometry(geometry, times[2:4])
    with pytest.raises(ValueError, match="pose status"):
        build_neon_context(root, None, tmp_path, "boxer", {})


def test_internal_gaps_keep_point_times_and_do_not_age_tracks(recording, tmp_path):
    from types import SimpleNamespace

    from utils.track_3d_boxes import BoundingBox3DTracker
    from utils.viewer_3d import TrackerViewer

    root, times = recording
    geometry = tmp_path / "gaps.npz"
    points = np.array([[[2.0, 7.0, 1.0]], [[3.0, 7.0, 1.0]], [[4.0, 7.0, 1.0]]])
    make_geometry(geometry, times[[0, 2, 4]], points)
    loader = NeonLoader(root, resize=32, geometry_path=geometry)
    try:
        frames = list(loader)
        assert [d["sdp_w"].shape[0] for d in frames] == [1, 0, 1, 0, 1]
        loader.write_metadata(tmp_path, "boxer")
    finally:
        loader.close()
    context, _ = build_neon_context(root, None, tmp_path, "boxer", {})
    try:
        assert list(context["time_to_uids_slaml"]) == times[[0, 2, 4]].tolist()
        tracker = BoundingBox3DTracker(
            min_hits=1, merge_iou_threshold=0.0, force_cpu=True
        )
        viewer = SimpleNamespace(
            freeze_tracker=False,
            current_frame_idx=0,
            sorted_timestamps=times.tolist(),
            _neon_world_pose_by_time=context["world_pose_by_time"],
            _neon_last_tracked_ts=None,
            _neon_world_update_idx=-1,
            tracker=tracker,
            obs_trail_secs=0.0,
            timed_obbs={int(ts): boxes([[2.0, 7.0, 1.0]]) for ts in times},
            _filter_frame_obbs=lambda obbs: obbs,
            _get_cam_and_pose=lambda ts: (frames[0]["cam0"], frames[0]["T_world_rig0"]),
            _get_observed_points_trail=lambda *a, **k: None,
        )
        last_support = 0
        for i in range(5):
            TrackerViewer._step_neon_tracker(viewer, i)
            viewer.current_frame_idx = i
            track = tracker.get_all_tracks()[0]
            if i % 2 == 0:
                last_support += 1
            assert track.support_count == last_support
            assert track.missed_count == 0
        assert viewer._neon_world_update_idx == 2
        assert track.last_seen_frame == 2
    finally:
        context["loader"].close()


@pytest.fixture
def mixed_replay(recording, tmp_path):
    root, times = recording
    np.tile(np.array([[32.0, 24.0]], dtype="<f4"), (5, 1)).tofile(root / "gaze ps1.raw")
    times.tofile(root / "gaze ps1.time")
    geometry = tmp_path / "partial.npz"
    make_geometry(geometry, times[1:3])
    log = tmp_path / "results"
    log.mkdir()
    loader = NeonLoader(root, resize=32, geometry_path=geometry, gaze=True)
    gaze_writer = GazeCsvWriter(log / "boxer_gaze.csv")
    target_writer = GazeTargetCsvWriter(log / "boxer_gaze_targets.csv")
    detections = {}
    try:
        for i, datum in enumerate(loader):
            pose = datum["T_world_rig0"].matrix.numpy()
            obbs = boxes([pose[:3, 3] + 4 * pose[:3, 2]])
            detections[int(times[i])] = obbs
            gaze_writer.write(datum["gaze"])
            target_writer.write(
                estimate_gaze_target(
                    datum["gaze"],
                    obbs,
                    pose,
                    datum["coordinate_frame"],
                )
            )
            save_bb2d_csv(
                log / "owl_2dbbs.csv",
                i,
                np.array([[4, 4, 28, 28]]),
                [0.9],
                ["object_0"],
                time_ns=int(times[i]),
                img_width=32,
                img_height=32,
                append=i > 0,
            )
        loader.write_metadata(log, "boxer", gaze_targets=True)
    finally:
        gaze_writer.close()
        target_writer.close()
        loader.close()
    return root, times, log, detections


def test_headless_mixed_replay_holds_world_and_projects_local_boxes(
    mixed_replay, monkeypatch
):
    import moderngl_window as mglw
    from moderngl_window.context.headless import Window
    from moderngl_window.timers.clock import Timer

    from utils import imgui_compat as imgui
    from utils.viewer_3d import TrackerViewer

    monkeypatch.setenv("MESA_SHADER_CACHE_DISABLE", "true")
    monkeypatch.setattr("utils.fuse_3d_boxes._load_cached_text_embeddings", lambda: {})
    window = Window(size=(1600, 900), backend="egl", samples=0)
    mglw.activate_context(window=window)
    root, times, log, detections = mixed_replay
    context, timeline = build_neon_context(root, None, log, "boxer", detections)
    viewer = None
    try:
        viewer = TrackerViewer(
            timeline,
            str(log),
            seq_ctx=context,
            force_cpu=True,
            bb2d_csv_path=str(log / "owl_2dbbs.csv"),
            ctx=window.ctx,
            wnd=window,
            timer=Timer(),
        )
        imgui.get_io().set_ini_filename("")
        viewer.tracker_min_hits = 1
        viewer.tracker_merge_iou = 0.0
        viewer._reset_tracker()
        viewer.follow_view = True
        assert len(viewer._traj_all_segments) == 1  # only the 1 -> 2 world edge
        assert viewer.total_frames == 5
        state_at_two = None
        for i in range(5):
            viewer._step_to_frame(i)
            window.use()
            viewer.on_render(float(i), 1.0 / 30)
            viewer.ctx.finish()
            assert len(viewer._bb2d_current_boxes) == 1
            assert len(viewer._rgb_projected_raw_lines) == 1
            assert np.isfinite(viewer.get_camera_matrices()[2]).all()
            local = i not in (1, 2)
            assert viewer._current_frame_local == local
            assert viewer._gaze_overlay.has_ray == (not local)
            assert viewer._gaze_overlay.has_target == (not local)
            tracks = viewer.tracker.get_all_tracks()
            state = [(t.track_id, t.support_count, t.missed_count) for t in tracks]
            if i == 0:
                assert not tracks and viewer.tracked_all_instance_count == 0
            elif i == 2:
                state_at_two = state
                assert tracks[0].support_count == 2
            elif i > 2:
                assert state == state_at_two
                assert viewer.tracked_all_instance_count > 0
            if local:
                # Changing projection/overlay settings must not reintroduce
                # world tracks or a fake camera into a pose-less frame.
                viewer._rebuild_rgb_projections()
                viewer._build_frustum_geometry(*viewer._get_cam_and_pose(int(times[i])))
                viewer._update_trajectory_tail(int(times[i]))
                assert len(viewer._rgb_projected_tracked_visible_lines) == 1
                assert viewer.cached_instance_count == 0
                assert viewer.frustum_instance_count == viewer.obs_point_count == 0
                assert len(viewer.frame_obb_sets.raw) == 0
        # Seeking backwards into a local frame clears future world geometry.
        viewer._step_to_frame(0)
        assert not viewer.tracker.get_all_tracks()
        assert viewer.tracked_all_instance_count == 0
        # A valid frame with no detections still advances world tracking once.
        timeline[int(times[1])] = detections[int(times[1])][:0]
        viewer._step_to_frame(1)
        assert viewer._neon_world_update_idx == 0
        assert not viewer.tracker.get_all_tracks()
        viewer._step_to_frame(2)
        assert viewer._neon_world_update_idx == 1
        assert viewer.tracker.get_all_tracks()[0].support_count == 1
        # An empty local frame clears both image projections, preserving tracks.
        timeline[int(times[3])] = detections[int(times[3])][:0]
        viewer._step_to_frame(3)
        assert not viewer._rgb_projected_raw_lines
        assert not viewer._rgb_projected_tracked_visible_lines
        assert viewer.tracker.get_all_tracks()[0].support_count == 1
    finally:
        if viewer is not None:
            viewer.on_close()
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context()
        context["loader"].close()
        window.destroy()


@pytest.mark.parametrize("mode", ["auto", "off", "on", "local"])
def test_runner_exports_all_detections_but_tracks_only_world(
    recording, tmp_path, monkeypatch, mode
):
    import owl.owl_wrapper
    import run_boxer
    from utils.track_3d_boxes import BoundingBox3DTracker

    root, times = recording
    geometry = tmp_path / "partial.npz"
    make_geometry(geometry, times[1:3], np.array([[2.0, 7.0, 1.0]], dtype=np.float32))
    visited = []

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return torch.tensor([[1.0, 30.0, 1.0, 30.0]]), torch.ones(1), [0], None

    class Model(torch.nn.Module):
        hw = 32

        def forward(self, datum):
            visited.append(datum["frame_id"])
            obbs = boxes([[0.0, 0.0, 4.0]])
            obbs.set_T_world_object(datum["T_world_rig0"] @ obbs.T_world_object)
            return {"obbs_pr_w": obbs.unsqueeze(0)}

    def tracker(**kwargs):
        kwargs.update(min_hits=1, merge_iou_threshold=0.0)
        return BoundingBox3DTracker(**kwargs)

    monkeypatch.setattr(owl.owl_wrapper, "OwlWrapper", Detector)
    monkeypatch.setattr(
        run_boxer.BoxerNet, "load_from_checkpoint", lambda *a, **k: Model()
    )
    monkeypatch.setattr("utils.track_3d_boxes.BoundingBox3DTracker", tracker)
    out = tmp_path / "results"
    args = [
        "run_boxer.py",
        "--input",
        str(root),
        "--start_n",
        "0",
        "--max_n",
        "5",
        "--force_cpu",
        "--skip_viz",
        "--track",
        "--labels",
        "chair",
        "--point_cloud",
        "off" if mode == "local" else mode,
        "--output_dir",
        str(out),
    ]
    if mode != "local":
        args += ["--neon_geometry", str(geometry)]
    monkeypatch.setattr(sys, "argv", args)
    run_boxer.main()
    assert visited == list(range(5))
    log = out / "neon"
    assert list(read_obb_csv(log / "boxer_3dbbs.csv")) == times.tolist()
    if mode != "local":
        assert list(read_obb_csv(log / "boxer_3dbbs_world.csv")) == times[1:3].tolist()
        # A new track is tentative until its second observation (existing lifecycle).
        assert (
            list(read_obb_csv(log / "boxer_3dbbs_tracked.csv")) == times[2:3].tolist()
        )
    else:
        assert not (log / "boxer_3dbbs_world.csv").exists()
        assert not (log / "boxer_3dbbs_tracked.csv").exists()
    with (log / "owl_2dbbs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame_id"]) for row in rows] == list(range(5))
    context, timeline = build_neon_context(
        root, None, log, "boxer", read_obb_csv(log / "boxer_3dbbs.csv")
    )
    try:
        assert list(timeline) == times.tolist()
    finally:
        context["loader"].close()
    if mode != "local":
        fused_inputs = []
        monkeypatch.setattr(
            "utils.fuse_3d_boxes.fuse_obbs_from_csv",
            lambda path: fused_inputs.append(list(read_obb_csv(path))),
        )
        args.remove("--track")
        monkeypatch.setattr(sys, "argv", args + ["--cache3d", "--fuse"])
        run_boxer.main()
        assert fused_inputs == [times[1:3].tolist()]
