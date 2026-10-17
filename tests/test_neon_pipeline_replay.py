"""End-to-end pipeline replay, preserving every frame and independent worlds."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gaze_target import boxes
from test_neon_loader import make_geometry
from test_neon_loader import recording as native_recording_fixture

import run_neon_pipeline as pipeline
from loaders.neon_loader import NeonLoader
from utils.file_io import ObbCsvWriter2, read_obb_csv, save_bb2d_csv
from utils.gaze_target import GazeTargetCsvWriter, estimate_gaze_target
from utils.neon_gaze import GazeCsvWriter
from utils.neon_pipeline_viewer import assemble_replay, selected_frame_ids
from utils.neon_viewer import build_neon_context

recording = native_recording_fixture


@pytest.fixture(params=[0, 1, 2], ids=["no_world", "one_world", "two_worlds"])
def replay(recording, tmp_path, request):
    root, times = recording
    np.tile(np.array([[32.0, 24.0]], dtype="<f4"), (5, 1)).tofile(root / "gaze ps1.raw")
    times.tofile(root / "gaze ps1.time")
    out = tmp_path / "run"
    prepared = out / "prepared"
    prepared.mkdir(parents=True)
    # Frame zero was requested but could not enter VIO due to missing IMU.
    (prepared / "frames.csv").write_text("frame_id\n1\n2\n3\n4\n")
    (prepared / "manifest.json").write_text(json.dumps({"excluded_without_imu": [0]}))
    summary = dict(
        recording=str(root),
        prepared=str(prepared),
        segments=[],
        frame_only_intervals=[],
        usable_frames=2 * request.param,
    )
    parts = []
    if request.param:
        parts.append((1, 2, 0))
    if request.param == 2:
        parts.append((3, 2, 1))
    parts += [
        (row["start_n"], row["max_n"], -1)
        for row in pipeline.uncovered_intervals(
            range(5),
            summary["segments"]
            or [{"start_n": start, "max_n": count} for start, count, _ in parts],
        )
    ]
    for index, (start, count, world) in enumerate(parts):
        directory = out / f"part_{index:03d}" / root.name
        directory.mkdir(parents=True)
        geometry = out / f"geometry_{world}.npz" if world >= 0 else None
        if geometry:
            data = make_geometry(
                geometry,
                times[start : start + count],
                np.array([[2.0 + world * 100, 7.0, 1.0]], dtype=np.float32),
            )
            data["T_world_camera"][:, 0, 3] += world * 100
            np.savez(geometry, **data)
        loader = NeonLoader(
            root,
            start_n=start,
            max_n=count,
            resize=32,
            geometry_path=geometry,
            gaze=True,
        )
        writer = ObbCsvWriter2(directory / "boxer_3dbbs.csv")
        gaze = GazeCsvWriter(directory / "boxer_gaze.csv")
        targets = GazeTargetCsvWriter(directory / "boxer_gaze_targets.csv")
        try:
            for datum in loader:
                fid, ts = datum["frame_id"], datum["time_ns0"]
                pose = datum["T_world_rig0"].matrix.numpy()
                obbs = boxes([pose[:3, 3] + pose[:3, 2] * 4])
                writer.write(obbs, ts, sem_id_to_name={0: "object_0"})
                save_bb2d_csv(
                    directory / "owl_2dbbs.csv",
                    fid,
                    np.array([[4, 4, 28, 28]]),
                    [0.9],
                    ["object_0"],
                    time_ns=ts,
                    img_width=32,
                    img_height=32,
                    append=fid != start,
                )
                gaze.write(datum["gaze"])
                targets.write(
                    estimate_gaze_target(
                        datum["gaze"], obbs, pose, datum["coordinate_frame"]
                    )
                )
            loader.write_metadata(directory, "boxer", gaze_targets=True)
        finally:
            loader.close()
            writer.close()
            gaze.close()
            targets.close()
        part = dict(start_n=start, max_n=count, results=str(directory))
        if world >= 0:
            part["geometry"] = str(geometry)
            summary["segments"].append(part)
        else:
            summary["frame_only_intervals"].append(part)
    output = assemble_replay(summary, out)
    return root, times, summary, output / root.name


def test_assembled_replay_preserves_all_frames_and_world_identity(replay):
    root, times, summary, log = replay
    detections = read_obb_csv(log / "boxer_3dbbs.csv")
    assert list(detections) == times.tolist()
    context, timeline = build_neon_context(root, None, log, "boxer", detections)
    try:
        assert list(timeline) == times.tolist()
        assert context["source_frame_ids"].tolist() == list(range(5))
        assert len(context["world_contexts"]) == len(summary["segments"])
        assert sum(context["world_pose_by_time"].values()) == summary["usable_frames"]
        assert context["world_segment_by_time"][int(times[0])] == -1
        assert len(context["gaze_frames"]) == 5
        assert selected_frame_ids(summary["prepared"]) == list(range(5))
        assert summary["total_estimated_frames"] == 5
    finally:
        context["loader"].close()


def test_headless_world_switch_never_combines_maps_or_tracks(replay, monkeypatch):
    import moderngl_window as mglw
    from moderngl_window.context.headless import Window
    from moderngl_window.timers.clock import Timer

    from utils import imgui_compat as imgui
    from utils.viewer_3d import TrackerViewer

    monkeypatch.setenv("MESA_SHADER_CACHE_DISABLE", "true")
    monkeypatch.setattr("utils.fuse_3d_boxes._load_cached_text_embeddings", lambda: {})
    root, times, summary, log = replay
    context, timeline = build_neon_context(
        root, None, log, "boxer", read_obb_csv(log / "boxer_3dbbs.csv")
    )
    window = Window(size=(1600, 900), backend="egl", samples=0)
    mglw.activate_context(window=window)
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
        counts = {}
        for i in range(5):
            viewer._step_to_frame(i)
            window.use()
            viewer.on_render(float(i), 1.0 / 30)
            window.ctx.finish()
            assert len(viewer._bb2d_current_boxes) == 1
            assert len(viewer._rgb_projected_raw_lines) == 1
            world = context["world_segment_by_time"][int(times[i])]
            if world < 0:
                assert viewer._current_frame_local
                assert not viewer._gaze_overlay.has_ray
                continue
            counts[world] = counts.get(world, 0) + 1
            tracks = viewer.tracker.get_all_tracks()
            assert len(tracks) == 1 and tracks[0].support_count == counts[world]
            assert viewer._active_neon_world_segment == world
            assert viewer.point_count == 1
            assert viewer._point_positions[0, 0] == pytest.approx(2.0 + 100 * world)
            assert len(viewer._traj_all_segments) == 1
        if len(summary["segments"]) == 2:
            # Freeze cannot keep projecting the previous world's tracks.
            viewer.freeze_tracker = True
            viewer._step_to_frame(1)
            assert viewer._active_neon_world_segment == 0
            assert not viewer.tracker.get_all_tracks()
            assert viewer._point_positions[0, 0] == 2.0
            viewer.freeze_tracker = False
            viewer._step_to_frame(2)
            assert viewer.tracker.get_all_tracks()[0].support_count == 1
    finally:
        if viewer:
            viewer.on_close()
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context()
        context["loader"].close()
        window.destroy()


def test_assembly_rejects_missing_part_frames(replay):
    _, _, summary, log = replay
    summary["frame_only_intervals"] = []
    with pytest.raises(ValueError, match="every selected"):
        assemble_replay(summary, log.parent.parent)


def test_replay_rejects_changed_world_identity(replay):
    root, _, _, log = replay
    path = log / "boxer_frames.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["world_segment_id"] = "100"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="identity"):
        build_neon_context(root, None, log, "boxer", {})


@pytest.mark.parametrize("worlds", [0, 2])
def test_pipeline_main_prepares_converts_infers_and_assembles_all_frames(
    recording, tmp_path, monkeypatch, worlds
):
    import owl.owl_wrapper
    import run_boxer

    root, times = recording
    code_root = tmp_path / "code"
    binary = code_root / "output/neon_vio_build/mono_inertial_neon"
    vocab = code_root / "output/neon_vio_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    for path in (binary, vocab):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(pipeline, "ROOT", code_root)
    stages, inferred = [], []

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return torch.tensor([[1.0, 30.0, 1.0, 30.0]]), torch.ones(1), [0], None

    class Model(torch.nn.Module):
        hw = 32

        def forward(self, datum):
            inferred.append(datum["frame_id"])
            obbs = boxes([[0.0, 0.0, 4.0]])
            obbs.set_T_world_object(datum["T_world_rig0"] @ obbs.T_world_object)
            return {"obbs_pr_w": obbs.unsqueeze(0)}

    monkeypatch.setattr(owl.owl_wrapper, "OwlWrapper", Detector)
    monkeypatch.setattr(
        run_boxer.BoxerNet, "load_from_checkpoint", lambda *a, **k: Model()
    )
    monkeypatch.setattr("utils.fuse_3d_boxes._load_cached_text_embeddings", lambda: {})

    def run(command, log, env=None, **kwargs):
        if str(command[1]).endswith("prepare_neon_vio.py"):
            stages.append("prepare")
            path = Path(command[command.index("--output") + 1])
            path.mkdir()
            (path / "frames.csv").write_text("frame_id\n1\n2\n3\n4\n")
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "excluded_without_imu": [0],
                        "imu_offset_ms_applied": 0,
                        "imu_noise_source": "test",
                        "recording": str(root),
                    }
                )
            )
        elif command[0] == binary:
            stages.append("slam")
            path = Path(command[3])
            path.mkdir()
            (path / "run.json").write_text(json.dumps({"next_index": 4}))
        else:
            stages.append("infer")
            with monkeypatch.context() as patch:
                patch.setattr(sys, "argv", list(map(str, command[1:])))
                run_boxer.main()
        return 0

    def convert(prepared, trajectory, states, output, *a, **k):
        stages.append("convert")
        output.mkdir()
        segments = []
        for i in range(worlds):
            start = 1 + i * 2
            filename = f"geometry_segment_{i:03d}.npz"
            make_geometry(output / filename, times[start : start + 2])
            segments.append(dict(file=filename, start_n=start, max_n=2))
        return {"segments": segments}

    monkeypatch.setattr(pipeline, "run_logged", run)
    monkeypatch.setattr(pipeline, "convert", convert)
    monkeypatch.setattr(pipeline, "evaluate", lambda *a: None)
    out = tmp_path / "pipeline"
    args = [
        "run_neon_pipeline.py",
        "--input",
        str(root),
        "--output",
        str(out),
        "--skip_viz",
        "--force_cpu",
        "--labels",
        "chair",
    ]
    monkeypatch.setattr(sys, "argv", args)
    pipeline.main()
    assert stages[:3] == ["prepare", "slam", "convert"]
    assert sorted(inferred) == list(range(5))  # includes IMU-excluded frame zero
    summary = json.loads((out / "pipeline.json").read_text())
    assert summary["input_frames"] == summary["total_estimated_frames"] == 5
    assert summary["prepared_frames"] == 4
    log = Path(summary["viewer_results"])
    ctx, timeline = build_neon_context(
        root, None, log, "boxer", read_obb_csv(log / "boxer_3dbbs.csv")
    )
    ctx["loader"].close()
    assert list(timeline) == times.tolist()
    # Reuse the pipeline without rerunning preprocessing or SLAM.
    stages.clear()
    inferred.clear()
    monkeypatch.setattr(sys, "argv", args + ["--boxer-only"])
    pipeline.main()
    assert stages and set(stages) == {"infer"}
    assert sorted(inferred) == list(range(5))
