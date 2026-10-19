"""Standalone boundary: retain exact time/coordinates without certifying lost poses."""
import csv
import json

import numpy as np
import pytest

from standalone.orbslam3.run import export_geometry, pose_matrix, prepare_subset


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def recording(tmp_path):
    from standalone.orbslam3.run import REVISION
    root = tmp_path / "analysis"
    prep = root / "prepared"
    slam = root / "slam"
    prep.mkdir(parents=True)
    slam.mkdir()
    manifest = dict(recording="/example/neon", orb_slam3_commit=REVISION,
                    pose_frame="scene_camera", excluded_without_imu=[0],
                    imu_offset_ms_applied=0., imu_noise_source="test")
    (prep / "manifest.json").write_text(json.dumps(manifest))
    frames, states, trajectory = [], [], []
    for i in range(1, 9):
        frames.append(dict(frame_id=i, time_ns=1_790_000_000_000_000_123+i*100_000_003,
                           slam_time_s=i*.1, image=f"images/{i}.png"))
        states.append(dict(frame_id=i, slam_time_s=i*.1, state=3 if i == 4 else 2,
                           imu_initialized=0 if i == 3 else 1, map_id=1 if i in (5, 6) else 0))
        trajectory.append(dict(slam_time_s=i*.1, metric=1, map_id=1 if i in (5, 6) else 0,
                               tx=i, ty=0, tz=0, qx=0, qy=0, qz=0, qw=1))
    write_csv(prep / "frames.csv", frames)
    write_csv(slam / "states.csv", states)
    write_csv(slam / "camera_trajectory.csv", trajectory)
    write_csv(slam / "map_points.csv", [dict(map_id=0, point_id=4, x=9, y=3, z=2, observations=3)])
    (slam / "run.json").write_text(json.dumps(dict(processed_frames=8)))
    (prep / "camera.yaml").write_text("%YAML:1.0\nIMU.InsertKFsWhenLost: 0\nCamera.width: 800\n")
    (prep / "imu.csv").write_text("time,gx,gy,gz,ax,ay,az\n")
    return root


def test_export_keeps_exact_time_map_coordinates_and_excludes_lost(recording):
    summary = export_geometry(recording)
    assert summary["input_frames"] == 9
    assert summary["usable_frames"] == 6
    assert [(s["start_n"], s["max_n"], s["map_id"]) for s in summary["segments"]] == [(1, 2, 0), (5, 2, 1), (7, 2, 0)]
    for segment in summary["segments"]:
        with np.load(segment["geometry"]) as data:
            ids = data["source_frame_ids"]
            np.testing.assert_array_equal(data["timestamps_ns"], 1_790_000_000_000_000_123+ids*100_000_003)
            np.testing.assert_array_equal(data["T_world_camera"][:, 0, 3], ids)
            if segment["map_id"] == 0:
                np.testing.assert_array_equal(data["points_world"], [[9, 3, 2]])
    assert summary["standalone"]["final_metric_maps"] == [0, 1]
    maps = summary["standalone"]["map_outputs"]
    assert [(m["map_id"], m["pose_frames"]) for m in maps] == [(0, 4), (1, 2)]
    np.testing.assert_allclose(maps[0]["continuous_speed_m_s_p50_p99_max"], [1e9/100_000_003]*3)
    assert maps[0]["camera_extent_m"] == [7., 0., 0.]
    with np.load(maps[0]["geometry"]) as data:
        np.testing.assert_array_equal(data["source_frame_ids"], [1, 2, 7, 8])
        np.testing.assert_array_equal(data["T_world_camera"][:, 0, 3], [1, 2, 7, 8])
        np.testing.assert_array_equal(data["points_world"], [[9, 3, 2]])


def test_export_rejects_missing_states(recording):
    path = recording / "slam/states.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
    with pytest.raises(ValueError, match="every prepared frame"):
        export_geometry(recording)


def test_failed_native_output_cannot_be_manually_exported(recording):
    (recording / "failure.json").write_text('{"returncode": -11}')
    with pytest.raises(ValueError, match="Cannot export a failed"):
        export_geometry(recording)


@pytest.mark.parametrize("parts_key", ["segments", "frame_only_intervals"])
def test_export_preserves_existing_boxer_pipeline(recording, parts_key):
    path = recording / "pipeline.json"
    original = json.dumps({parts_key: [{"results": "existing/boxer"}]})
    path.write_text(original)
    with pytest.raises(ValueError, match="Cannot replace geometry after Boxer"):
        export_geometry(recording)
    assert path.read_text() == original


def test_export_rejects_unknown_or_duplicate_trajectory_time(recording):
    path = recording / "slam/camera_trajectory.csv"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[1]+"\n")
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        export_geometry(recording)


def test_small_runs_not_interpolated(recording):
    result = export_geometry(recording, min_frames=3)
    assert result["usable_frames"] == 0
    assert not result["standalone"]["interpolation"]


@pytest.mark.parametrize("corruption", ["duplicate_timestamp", "excluded_overlap"])
def test_corrupt_source_selection_rejected(recording, corruption):
    if corruption == "duplicate_timestamp":
        path = recording / "prepared/frames.csv"
        rows = list(csv.DictReader(path.open()))
        rows[1]["time_ns"] = rows[0]["time_ns"]
        write_csv(path, rows)
    else:
        path = recording / "prepared/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["excluded_without_imu"] = [1]
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid source frame/timestamp"):
        export_geometry(recording)


def test_subset_preserves_input_and_filters_excluded_frames(recording, tmp_path):
    original = recording / "prepared"
    before = (original / "camera.yaml").read_bytes()
    target = tmp_path / "subset"
    assert prepare_subset(original, target, 5, 2, "on") == 2
    assert "IMU.InsertKFsWhenLost: 1" in (target / "camera.yaml").read_text()
    assert (original / "camera.yaml").read_bytes() == before
    assert json.loads((target / "manifest.json").read_text())["excluded_without_imu"] == []
    rows = list(csv.DictReader((target / "frames.csv").open()))
    assert [r["frame_id"] for r in rows] == ["5", "6"]
    assert rows[0]["image"] == str(original / "images/5.png")


def test_invalid_quaternion_rejected():
    row = dict(qx=0, qy=0, qz=0, qw=0, tx=0, ty=0, tz=0)
    with pytest.raises(ValueError, match="Invalid metric"):
        pose_matrix(row)


def test_quaternion_direction_is_camera_to_world():
    q = np.sqrt(.5)
    pose = pose_matrix(dict(qx=0, qy=0, qz=q, qw=q, tx=1, ty=2, tz=3))
    np.testing.assert_allclose(pose @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-14)


def test_native_failure_never_becomes_a_successful_pipeline(recording, tmp_path, monkeypatch):
    import standalone.orbslam3.run as runner
    runtime = tmp_path / "runtime"
    (runtime / "build").mkdir(parents=True)
    (runtime / "build/neon_standalone").write_bytes(b"fake native executable")
    (runtime / "build/CMakeCache.txt").write_text("NEON_ORPHAN_GUARD:BOOL=OFF\n")
    monkeypatch.setattr(runner, "RUNTIME", runtime)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: type("Crash", (), {"wait": lambda self, **kw: -11})())
    output = tmp_path / "failed"
    monkeypatch.setattr("sys.argv", ["run.py", "--prepared", str(recording / "prepared"), "--output", str(output)])
    with pytest.raises(RuntimeError, match="exited -11"):
        runner.main()
    report = json.loads((output / "failure.json").read_text())
    assert report["signal"] == 11
    assert report["status"] == "failed"
    assert not report["geometry_exported"]
    assert not (output / "pipeline.json").exists()
