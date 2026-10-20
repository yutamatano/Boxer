"""COLMAP preview never places metre-sized detections in an arbitrary world."""

import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_neon_loader import recording as native_recording_fixture
from test_gaze_target import boxes
from convert_colmap_trajectory import convert
from loaders.neon_loader import NeonLoader
from utils.neon_geometry import require_world_operations
from utils.neon_colmap_pipeline import select_prepared
from utils.neon_pipeline_viewer import assemble_replay, selected_frame_ids
from utils.neon_viewer import build_neon_context
from utils.file_io import ObbCsvWriter2, read_obb_csv

recording = native_recording_fixture


@pytest.fixture
def scene(recording, tmp_path):
    root, times = recording
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            dict(
                recording=str(root),
                excluded_without_imu=[],
                T_imu_camera=np.eye(4).tolist(),
                K_rectified=[[20, 0, 16], [0, 20, 16], [0, 0, 1]],
                image_size=[32, 32],
                imu_offset_ms_applied=0,
                imu_noise_source="test",
            )
        )
    )
    with (prepared / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "time_ns", "slam_time_s", "image"])
        w.writerows(
            (i, int(t), i * 0.1, "images/test.png") for i, t in enumerate(times)
        )
    (prepared / "camera.yaml").write_text("test")
    source = tmp_path / "models"
    source.mkdir()
    for m, ids, offset in [(0, [0, 1, 3], 100), (1, [1, 2, 4], 1000)]:
        folder = source / str(m)
        folder.mkdir()
        p = np.tile(np.eye(4), (len(ids), 1, 1))
        p[:, 0, 3] = np.array(ids) + offset
        np.savez(
            folder / "visual_trajectory.npz",
            frame_ids=np.array(ids, np.int64),
            timestamps_ns=times[ids],
            camera_to_world=p[:, :3],
            metric=False,
        )
        (folder / "quality.json").write_text(json.dumps(dict(metric=False)))
    return root, times, prepared, source


def test_conversion_no_inverse_and_exact_world_partition(scene, tmp_path):
    root, times, prepared, source = scene
    summary = convert(prepared, source, tmp_path / "largest")
    assert (
        summary["visual_pose_frames"] == 3
        and summary["usable_frames"] == 0
        and summary["metric_pose_frames"] == 0
    )
    assert [(r["start_n"], r["max_n"]) for r in summary["segments"]] == [(0, 2), (3, 1)]
    with np.load(summary["maps"][0]["geometry"]) as d:
        np.testing.assert_array_equal(d["T_world_camera"][:, 0, 3], [100, 101, 103])
        assert d["T_world_camera"].shape == (3, 4, 4)
    summary = convert(prepared, source, tmp_path / "separated", policy="separated")
    assert summary["visual_pose_frames"] == 5
    assert [(r["start_n"], r["max_n"], r["world_id"]) for r in summary["segments"]] == [
        (0, 2, 0),
        (2, 1, 1),
        (3, 1, 0),
        (4, 1, 1),
    ]


def test_wrong_time_invalid_rotation_and_rejected_quality(scene, tmp_path):
    _, _, prepared, source = scene
    path = source / "0/visual_trajectory.npz"
    with np.load(path) as d:
        original = {k: d[k].copy() for k in d.files}
    changed = dict(original)
    changed["timestamps_ns"] = original["timestamps_ns"] + 1
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="timestamp"):
        convert(prepared, source, tmp_path / "time")
    changed = dict(original)
    changed["camera_to_world"] = original["camera_to_world"].copy()
    changed["camera_to_world"][0, 0, 0] = 2
    np.savez(path, **changed)
    with pytest.raises(ValueError):
        convert(prepared, source, tmp_path / "rotation")
    np.savez(path, **original)
    (source / "0/quality.json").write_text('{"quality_status":"fail"}')
    with pytest.raises(ValueError, match="Rejected"):
        convert(prepared, source, tmp_path / "quality")


def test_scaled_rotation_is_not_scaled_and_stays_unverified(
    scene, tmp_path, monkeypatch
):
    _, _, prepared, source = scene
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    monkeypatch.setattr(
        "convert_colmap_trajectory.gravity_rotation",
        lambda *a: (R, dict(status="gravity_aligned_unverified")),
    )
    s = convert(prepared, source, tmp_path / "scaled", scale=2, align_gravity=True)
    with np.load(s["maps"][0]["geometry"]) as d:
        np.testing.assert_allclose(d["T_world_camera"][0, :3, :3], R)
        np.testing.assert_allclose(
            d["T_world_camera"][0, :3, 3], R @ [200, 0, 0], atol=1e-10
        )
    assert not s["metric"] and s["metric_pose_frames"] == 0


def test_loader_preview_uses_local_objects_and_blocks_world_operations(scene, tmp_path):
    root, _, prepared, source = scene
    s = convert(prepared, source, tmp_path / "geo")
    geometry = s["maps"][0]["geometry"]
    with pytest.raises(ValueError, match="nonmetric_preview"):
        NeonLoader(root, geometry_path=geometry, point_cloud="off")
    loader = NeonLoader(
        root, geometry_path=geometry, point_cloud="off", nonmetric_preview=True
    )
    try:
        assert not loader.has_world_poses and loader.visual_pose_valid.tolist() == [
            True,
            True,
            False,
            True,
            False,
        ]
        assert loader.frame_pose(0).t.norm() == 0
        for name in ["track", "fuse", "gaze_targets"]:
            with pytest.raises(ValueError, match="disabled"):
                require_world_operations(loader.pose_metadata, **{name: True})
    finally:
        loader.close()


def test_selected_prepared_never_expands_to_full_recording(scene, tmp_path):
    root, _, prepared, _ = scene
    selected = tmp_path / "subset"
    select_prepared(prepared, selected, root, 1, 3)
    assert selected_frame_ids(selected) == [1, 2, 3]
    assert json.loads((selected / "manifest.json").read_text())[
        "selection_source_prepared"
    ] == str(prepared)


def separated_replay(scene, tmp_path):
    root, times, prepared, source = scene
    out = tmp_path / "run"
    out.mkdir()
    s = convert(prepared, source, out / "geometry", policy="separated")
    s["frame_only_intervals"] = []
    for i, segment in enumerate(s["segments"]):
        folder = out / f"part{i}" / root.name
        folder.mkdir(parents=True)
        loader = NeonLoader(
            root,
            start_n=segment["start_n"],
            max_n=segment["max_n"],
            resize=32,
            geometry_path=segment["geometry"],
            point_cloud="off",
            nonmetric_preview=True,
        )
        try:
            loader.write_metadata(folder, "boxer")
            with_writer = ObbCsvWriter2(folder / "boxer_3dbbs.csv")
            for frame in loader:
                with_writer.write(
                    boxes([[0, 0, 4]]), frame["time_ns0"], sem_id_to_name={0: "object"}
                )
            with_writer.close()
        finally:
            loader.close()
        segment["results"] = str(folder)
    viewer = assemble_replay(s, out)
    path = viewer / root.name
    return root, times, s, path


def test_separated_replay_keeps_local_boxes_and_visual_worlds(scene, tmp_path):
    root, times, s, path = separated_replay(scene, tmp_path)
    context, timeline = build_neon_context(
        root, None, path, "boxer", read_obb_csv(path / "boxer_3dbbs.csv")
    )
    try:
        assert len(timeline) == 5 and context["nonmetric_preview"]
        assert not context["world_pose_valid"].any()
        assert [context["preview_segment_by_time"][int(t)] for t in times] == [
            0,
            0,
            1,
            0,
            1,
        ]
        assert context["preview_traj"][0].t[0] == 100
        assert all(p.t.norm() == 0 for p in context["traj"])
    finally:
        context["loader"].close()


def test_headless_preview_does_not_track_or_connect_worlds(
    scene, tmp_path, monkeypatch
):
    import moderngl_window as mglw
    from moderngl_window.context.headless import Window
    from moderngl_window.timers.clock import Timer
    from utils import imgui_compat as imgui
    from utils.viewer_3d import TrackerViewer

    monkeypatch.setenv("MESA_SHADER_CACHE_DISABLE", "true")
    monkeypatch.setattr("utils.fuse_3d_boxes._load_cached_text_embeddings", lambda: {})
    root, times, _, path = separated_replay(scene, tmp_path)
    context, timeline = build_neon_context(
        root, None, path, "boxer", read_obb_csv(path / "boxer_3dbbs.csv")
    )
    window = Window(size=(1600, 900), backend="egl", samples=0)
    mglw.activate_context(window=window)
    viewer = None
    try:
        viewer = TrackerViewer(
            timeline,
            str(path),
            seq_ctx=context,
            force_cpu=True,
            ctx=window.ctx,
            wnd=window,
            timer=Timer(),
        )
        imgui.get_io().set_ini_filename("")

        def forbidden(*args, **kwargs):
            pytest.fail("Nonmetric preview must never update world tracks")

        # Resets create a new tracker at map changes, so patch the class method.
        monkeypatch.setattr(type(viewer.tracker), "update", forbidden)
        viewer.follow_view = True
        for n, i in enumerate([0, 1, 2, 3, 4, 0]):
            viewer._step_to_frame(i)
            window.use()
            viewer.on_render(float(n), 1 / 30)
            window.ctx.finish()
            assert (
                viewer._active_neon_world_segment
                == context["preview_segment_by_time"][int(times[i])]
            )
            assert viewer._current_frame_local
            assert viewer.tracker.tracks == []
            # Only frames 0->1 share a world. No edges may cross other maps.
            assert len(viewer._traj_all_segments) == (
                1 if viewer._active_neon_world_segment == 0 else 0
            )
    finally:
        if viewer:
            viewer.on_close()
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context()
        context["loader"].close()
        window.destroy()


def test_visual_preparation_without_orb_or_imu(recording, tmp_path):
    from utils.neon_colmap_pipeline import prepare_visual

    root, times = recording
    out = tmp_path / "visual"
    meta = prepare_visual(root, out, 1, 3, width=32)
    assert not meta["imu_available"]
    rows = list(csv.DictReader((out / "frames.csv").open()))
    assert [int(r["time_ns"]) for r in rows] == times[1:4].tolist()
    assert [int(r["frame_id"]) for r in rows] == [1, 2, 3]
    assert not (out / "imu.csv").exists()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    path = tmp_path / "extrinsics.json"
    path.write_text(json.dumps(transform.tolist()))
    calibrated = prepare_visual(
        root, tmp_path / "visual_calibrated", 1, 3, width=32, extrinsics=path
    )
    np.testing.assert_allclose(calibrated["T_imu_camera"], transform)
    assert calibrated["extrinsics_source"] == str(path)


def test_pipeline_fresh_colmap_stages_and_failure_preservation(
    scene, tmp_path, monkeypatch
):
    import run_neon_pipeline as pipeline

    root, _, prepared, source = scene
    monkeypatch.setattr(pipeline, "ROOT", tmp_path / "no_orb_binaries")
    calls = []

    def run(command, log, **kwargs):
        stage = command[command.index("--stage") + 1]
        calls.append(stage)
        if stage == "reconstruct":
            target = Path(command[command.index("--output") + 1]) / "mapping_auto_5"
            target.parent.mkdir(parents=True)
            shutil.copytree(source, target)

    monkeypatch.setattr(pipeline, "run_logged", run)
    out = tmp_path / "fresh"
    args = [
        "run_neon_pipeline.py",
        "--input",
        str(root),
        "--output",
        str(out),
        "--prepared",
        str(prepared),
        "--max_n",
        "5",
        "--pose_backend",
        "colmap",
        "--nonmetric_preview",
        "--poses-only",
    ]
    monkeypatch.setattr(sys, "argv", args)
    pipeline.main()
    assert calls == ["extract", "match", "database", "reconstruct"]
    result = json.loads((out / "pipeline.json").read_text())
    assert result["visual_pose_frames"] == 3
    assert result["metric_pose_frames"] == 0
    evaluations = [
        json.loads(Path(s["evaluation"]).read_text()) for s in result["segments"]
    ]
    assert [e["frames"] for e in evaluations] == [2, 1]
    assert all("speed_m_s" not in e for e in evaluations)

    def fail(command, log, **kwargs):
        marker = Path(log).parent / "partial_result.txt"
        marker.write_text("keep partial results")
        raise ValueError("Intentional reconstruction failure")

    monkeypatch.setattr(pipeline, "run_logged", fail)
    args[args.index(str(out))] = str(tmp_path / "failed")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        pipeline.main()
    failed = tmp_path / "failed"
    assert (failed / "partial_result.txt").read_text() == "keep partial results"
    assert json.loads((failed / "failure.json").read_text())["outputs_preserved"]


@pytest.mark.parametrize("flag", ["--track", "--fuse", "--gaze_targets"])
def test_boxer_cli_rejects_unvalidated_world_operations_before_inference(
    scene, tmp_path, monkeypatch, flag
):
    import run_boxer

    root, _, prepared, source = scene
    summary = convert(prepared, source, tmp_path / "geometry")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_boxer.py",
            "--input",
            str(root),
            "--output_dir",
            str(tmp_path / "boxes"),
            "--neon_geometry",
            summary["maps"][0]["geometry"],
            "--nonmetric_preview",
            "--point_cloud",
            "off",
            flag,
        ],
    )
    with pytest.raises(SystemExit):
        run_boxer.main()


def test_one_registered_pose_evaluation_and_direct_fusion_guard(scene, tmp_path):
    from evaluate_neon_geometry import evaluate
    from utils.fuse_3d_boxes import fuse_obbs_from_csv

    _, _, prepared, source = scene
    with np.load(source / "0/visual_trajectory.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    for key in ["frame_ids", "timestamps_ns", "camera_to_world"]:
        data[key] = data[key][:1]
    np.savez(source / "0/visual_trajectory.npz", **data)
    summary = convert(prepared, source, tmp_path / "geo", map_id=0)
    report = evaluate(
        prepared, summary["maps"][0]["geometry"], tmp_path / "evaluation.json"
    )
    assert report["frames"] == 1 and report["speed_visual_units_s"] is None
    (tmp_path / "boxer_input.json").write_text(
        json.dumps({"pose_geometry_metadata": summary["maps"][0]["metadata"]})
    )
    # Guard runs before the box CSV is opened or any embeddings are loaded.
    with pytest.raises(ValueError, match="disabled"):
        fuse_obbs_from_csv(str(tmp_path / "boxer_3dbbs.csv"))


@pytest.mark.parametrize(
    "missing", ["scale_status", "world_up_status", "quality_status", "pose_units"]
)
def test_claimed_world_inference_requires_every_validation(missing):
    from utils.neon_geometry import geometry_metadata

    metadata = dict(
        pose_backend="colmap",
        pose_frame="scene_camera",
        metric=True,
        pose_units="metres",
        scale_status="independently_verified",
        world_up_status="gravity_aligned",
        quality_status="pass",
        world_inference_allowed=True,
    )
    original = dict(metadata)
    del metadata[missing]
    with pytest.raises(ValueError, match="verified scale, gravity and quality"):
        geometry_metadata(
            {
                "metric": np.array(True),
                "geometry_metadata": np.array(json.dumps(metadata)),
            }
        )
    result = geometry_metadata(
        {"metric": np.array(True), "geometry_metadata": np.array(json.dumps(original))}
    )
    assert result["world_inference_allowed"]


def test_selected_timestamps_must_match_original_catalog(scene, tmp_path):
    root, _, prepared, source = scene
    selected = tmp_path / "selection"
    select_prepared(prepared, selected, root, 0, 5)
    with (selected / "frames.csv").open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["time_ns"] = str(int(rows[0]["time_ns"]) + 1)
    with (selected / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="source catalog"):
        convert(selected, source, tmp_path / "geometry")
