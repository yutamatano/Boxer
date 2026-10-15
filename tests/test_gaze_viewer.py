"""Exact-time replay, coordinate validation, and gaze disappearing on missing frames."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from loaders.neon_loader import NeonLoader
from test_gaze_target import boxes
from test_neon_loader import make_geometry, recording
from utils.gaze_target import GazeTargetCsvWriter, estimate_gaze_target
from utils.gaze_viewer import GazeOverlay, load_gaze_frames
from utils.neon_gaze import GazeCsvWriter
from utils.neon_viewer import build_neon_context
from utils.viewer_3d import TrackerViewer


@pytest.fixture(params=[False, True], ids=["local", "world"])
def exported(recording, tmp_path, request):
    root, times = recording
    # Frame 0/1 have gaze; frame 2 has invalid gaze; 3/4 are outside tolerance.
    np.array([[32., 24.], [32., 24.], [np.nan, 24.]], dtype="<f4").tofile(root / "gaze ps1.raw")
    (times[:3] + 1).tofile(root / "gaze ps1.time")
    geometry = tmp_path / "geometry.npz" if request.param else None
    if geometry:
        make_geometry(geometry, times)
    log = tmp_path / "results"
    log.mkdir()
    loader = NeonLoader(root, resize=32, gaze=True, geometry_path=geometry)
    poses = []
    gaze_writer = GazeCsvWriter(log / "boxer_gaze.csv")
    target_writer = GazeTargetCsvWriter(log / "boxer_gaze_targets.csv")
    try:
        for i in range(5):
            datum = loader.load(i)
            pose = datum["T_world_rig0"].matrix.numpy()
            poses.append(pose)
            centre = pose[:3, 3] + pose[:3, 2] * 4
            obbs = boxes([centre]) if i == 0 else None
            gaze_writer.write(datum["gaze"])
            target_writer.write(estimate_gaze_target(
                datum["gaze"], obbs, pose, loader.coordinate_frame,
            ))
        loader.write_metadata(log, "boxer", gaze_targets=True)
    finally:
        gaze_writer.close()
        target_writer.close()
        loader.close()
    return root, times, geometry, log, np.array(poses)


def test_replay_coordinates_and_missing_frames(exported):
    root, times, geometry, log, poses = exported
    ctx, timeline = build_neon_context(root, geometry, log, "boxer", {}, point_cloud="off")
    try:
        assert len(timeline) == 5  # includes frames with zero detections
        assert ctx["frame_local"] == (geometry is None)
        assert not ctx["uid_to_p3"]
        frames = ctx["gaze_frames"]
        hit = frames[int(times[0])]
        np.testing.assert_allclose(hit["origin"], poses[0, :3, 3])
        np.testing.assert_allclose(hit["direction"], poses[0, :3, 2])
        np.testing.assert_allclose(hit["target"], poses[0, :3, 3] + 3 * poses[0, :3, 2])
        assert hit["distance_m"] == pytest.approx(3.)
        assert frames[int(times[1])]["target"] is None
        assert frames[int(times[1])]["direction"] is not None
        assert frames[int(times[2])]["direction"] is None
        assert frames[int(times[3])]["direction"] is None
        assert int(times[0]) + 1 not in frames
    finally:
        ctx["loader"].close()


def rewrite_csv(path, mutate):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("problem", ["clock", "id", "duplicate", "nan", "coordinate",
                                      "sample", "target", "distance", "no_gaze"])
def test_inconsistent_exports_rejected(exported, problem):
    root, times, geometry, log, _ = exported
    gaze = log / "boxer_gaze.csv"
    target = log / "boxer_gaze_targets.csv"
    if problem == "clock":
        rewrite_csv(gaze, lambda r: r[0].update(time_ns=str(int(times[0]) + 1)))
    elif problem == "id":
        rewrite_csv(gaze, lambda r: r[0].update(frame_id="1"))
    elif problem == "duplicate":
        rewrite_csv(gaze, lambda r: r.append(r[0]))
    elif problem == "nan":
        rewrite_csv(gaze, lambda r: r[0].update(direction_camera_x="nan"))
    elif problem == "coordinate":
        rewrite_csv(target, lambda r: r[0].update(coordinate_frame="other_map"))
    elif problem == "sample":
        rewrite_csv(target, lambda r: r[0].update(gaze_time_ns=str(int(times[0]) + 2)))
    elif problem == "target":
        rewrite_csv(target, lambda r: r[0].update(position_frame_x="100"))
    elif problem == "distance":
        rewrite_csv(target, lambda r: r[0].update(distance_m="-3"))
    else:
        gaze.unlink()
    with pytest.raises(ValueError):
        build_neon_context(root, geometry, log, "boxer", {})


def test_wrong_map_or_geometry_added_to_local_is_rejected(exported, tmp_path):
    root, times, geometry, log, _ = exported
    other = tmp_path / "other_map.npz"
    data = make_geometry(other, times)
    data["T_world_camera"][:, 0, 3] += 5
    np.savez(other, **data)
    with pytest.raises(ValueError, match="geometry|world geometry"):
        build_neon_context(root, other, log, "boxer", {})


def test_gaze_only_and_sparse_rows_never_reuse_previous_frame(exported):
    root, times, geometry, log, _ = exported
    (log / "boxer_gaze_targets.csv").unlink()
    rewrite_csv(log / "boxer_gaze.csv", lambda rows: rows.pop(1))
    ctx, _ = build_neon_context(root, geometry, log, "boxer", {})
    try:
        assert ctx["gaze_frames"][int(times[0])]["target"] is None
        assert int(times[1]) not in ctx["gaze_frames"]
    finally:
        ctx["loader"].close()


def test_local_replay_does_not_update_tracker_or_stack_frames():
    visited = []
    viewer = SimpleNamespace(
        total_frames=3, current_frame_idx=0, _frame_local=True,
        sorted_timestamps=[101, 202, 303], _gaze_frames={},
        _rebuild_current_view=lambda: visited.append(viewer.current_frame_idx),
        _load_rgb_for_timestamp=lambda ts: None,
    )
    # No tracker object: any attempted cross-frame update would fail this test.
    for index in [1, 2, 0, 2]:
        TrackerViewer._step_to_frame(viewer, index)
    assert visited == [1, 2, 0, 2]


def test_optional_gaze_does_not_break_old_outputs(tmp_path):
    assert load_gaze_frames(tmp_path, "boxer", [0], [101], [np.eye(4)], "world_z_up") == {}


def test_point_cloud_modes_in_replay(recording, tmp_path):
    root, times = recording
    geometry = tmp_path / "geometry.npz"
    make_geometry(geometry, times, np.array([[1, 2, 3]], dtype=np.float32))
    loader = NeonLoader(root, resize=32, geometry_path=geometry)
    try:
        loader.write_metadata(tmp_path, "boxer")
    finally:
        loader.close()
    for mode in ("off", "auto", "on"):
        ctx, _ = build_neon_context(root, geometry, tmp_path, "boxer", {}, point_cloud=mode)
        try:
            assert bool(ctx["uid_to_p3"]) == (mode != "off")
        finally:
            ctx["loader"].close()


def test_gpu_buffers_are_bounded_and_missing_gaze_clears_marker():
    # Shader/viewport integration is also exercised by the real headless smoke run.
    class Buffer:
        def write(self, data):
            self.data = np.frombuffer(data, dtype=np.float32).copy()

    overlay = GazeOverlay.__new__(GazeOverlay)
    overlay.line_vbo, overlay.point_vbo = Buffer(), Buffer()
    hit = dict(origin=np.array([1., 2., 3.]), direction=np.array([0., 1., 0.]),
               target=np.array([1., 5., 3.]))
    ray_only = dict(hit, target=None)
    for _ in range(1000):
        overlay.update(hit)
        assert overlay.has_target
        np.testing.assert_array_equal(overlay.line_vbo.data[3:6], hit["target"])
        overlay.update(ray_only, 2.)
        assert overlay.has_ray and not overlay.has_target
        np.testing.assert_array_equal(overlay.line_vbo.data[3:6], [1, 4, 3])
        overlay.update(None)
        assert not overlay.has_ray and not overlay.has_target
    assert overlay.line_vbo.data.nbytes == 40
    assert overlay.point_vbo.data.nbytes == 24


def test_headless_viewer_renders_hits_clears_misses_and_seeks(exported, monkeypatch):
    import moderngl_window as mglw
    from moderngl_window.context.headless import Window
    from moderngl_window.timers.clock import Timer
    from utils import imgui_compat as imgui

    monkeypatch.setenv("MESA_SHADER_CACHE_DISABLE", "true")
    # Synthetic replay tests rendering; no text model/inference download is needed.
    monkeypatch.setattr("utils.fuse_3d_boxes._load_cached_text_embeddings", lambda: {})
    try:
        window = Window(size=(1600, 900), backend="egl", samples=0)
    except Exception as exc:
        pytest.skip(f"EGL OpenGL context unavailable: {exc}")
    mglw.activate_context(window=window)
    root, times, geometry, log, poses = exported
    context, timeline = build_neon_context(root, geometry, log, "boxer", {})
    viewer = None
    try:
        viewer = TrackerViewer(timeline, str(log), seq_ctx=context, force_cpu=True,
                               ctx=window.ctx, wnd=window, timer=Timer())
        imgui.get_io().set_ini_filename("")
        viewer.camera_distance = 8.
        viewer.camera_target = poses[0, :3, 3] + poses[0, :3, 2] * 1.5
        overlay = viewer._gaze_overlay
        resource_ids = (overlay.line_vbo.glo, overlay.point_vbo.glo,
                        overlay.line_vao.glo, overlay.point_vao.glo)
        for i in [0, 1, 2, 3, 4, 0]:
            viewer._step_to_frame(i)
            window.use()
            viewer.on_render(float(i), 1. / 30)
            viewer.ctx.finish()
            assert overlay.has_ray == (i in (0, 1))
            assert overlay.has_target == (i == 0)
            assert resource_ids == (overlay.line_vbo.glo, overlay.point_vbo.glo,
                                    overlay.line_vao.glo, overlay.point_vao.glo)
            pixels = np.frombuffer(window.fbo.read(components=3), dtype=np.uint8)
            pixels = pixels.reshape(window.height, window.width, 3)
            width, _ = viewer._get_3d_viewport_size()
            view = pixels[:, -width:]
            marker = (view[..., 0] > 240) & (view[..., 1] < 30) & (view[..., 2] > 180)
            assert marker.sum() >= 25 if i == 0 else not marker.any()
        if context["frame_local"]:
            assert not viewer.tracker.get_all_tracks()
            assert viewer._traj_all_segments is None
        # Actual autoplay and the selected gaze timestamp stay in lockstep.
        viewer.is_playing = True
        viewer._last_step_time = 0.
        viewer.on_render(10., 1. / 30)
        assert viewer.current_frame_idx == 1
        assert viewer._gaze_state is context["gaze_frames"][int(times[1])]
    finally:
        if viewer is not None:
            viewer.on_close()
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context()
        context["loader"].close()
        window.destroy()
