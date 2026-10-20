#!/usr/bin/env python3
"""Independent Neon VIO runner and exact-timestamp Boxer NPZ exporter.

Requires only NumPy and the standalone C++ build; imports no Boxer modules.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

REVISION = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "output/orbslam3_standalone"


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_subset(source, output, start_n, max_n, lost_keyframes):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("orb_slam3_commit") != REVISION or manifest.get("pose_frame") != "scene_camera":
        raise ValueError("Unsupported prepared revision or camera frame")
    all_rows = read_csv(source / "frames.csv")
    end = start_n + max_n if max_n is not None else math.inf
    rows = [r for r in all_rows if start_n <= int(r["frame_id"]) < end]
    excluded = [i for i in manifest.get("excluded_without_imu", []) if start_n <= i < end]
    if len(rows) < 2:
        raise ValueError("Select at least two prepared frames")
    ids = [int(r["frame_id"]) for r in rows]
    if any(b != a + 1 for a, b in zip(ids, ids[1:])):
        raise ValueError("Prepared frames must be contiguous and increasing")
    output.mkdir()
    # Absolute paths let SLAM read images without copying/re-encoding any pixels.
    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(dict(r, image=str((source / r["image"]).resolve())) for r in rows)
    (output / "imu.csv").symlink_to((source / "imu.csv").resolve())
    settings = (source / "camera.yaml").read_text()
    if lost_keyframes != "preserve":
        settings = re.sub(r"(?m)^IMU\.InsertKFsWhenLost:.*\n?", "", settings)
        settings += f"\nIMU.InsertKFsWhenLost: {int(lost_keyframes == 'on')}\n"
    (output / "camera.yaml").write_text(settings)
    manifest.update(prepared_frames=len(rows), excluded_without_imu=excluded,
                    selected_start_n=start_n, selected_max_n=max_n,
                    standalone_source_prepared=str(source),
                    standalone_lost_keyframes=lost_keyframes)
    write_json(output / "manifest.json", manifest)
    return len(rows)


def pose_matrix(row):
    q = np.array([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
    t = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
    if not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError("Invalid metric camera pose")
    x, y, z, w = q / np.linalg.norm(q)
    pose = np.eye(4)
    pose[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    pose[:3, 3] = t
    return pose


def export_geometry(output, min_frames=2):
    if min_frames < 2:
        raise ValueError("min_frames must be >= 2")
    if (output / "failure.json").exists():
        raise ValueError("Cannot export a failed native run")
    pipeline_path = output / "pipeline.json"
    if pipeline_path.exists():
        previous = json.loads(pipeline_path.read_text())
        parts = previous.get("segments", []) + previous.get("frame_only_intervals", [])
        if previous.get("viewer_results") or any(s.get("results") for s in parts):
            raise ValueError("Cannot replace geometry after Boxer inference; preserve its pipeline")
    prepared = output / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    frames = read_csv(prepared / "frames.csv")
    states = read_csv(output / "slam/states.csv")
    native_report = json.loads((output / "slam/run.json").read_text())
    if native_report.get("processed_frames") != len(frames):
        raise ValueError("Native completion report must cover every prepared frame")
    if len(states) != len(frames):
        raise ValueError("SLAM states must cover every prepared frame")
    ids = np.array([int(f["frame_id"]) for f in frames], dtype=np.int64)
    times = np.array([int(f["time_ns"]) for f in frames], dtype=np.int64)
    excluded = list(map(int, manifest.get("excluded_without_imu", [])))
    if (len(frames) < 2 or np.any(np.diff(ids) != 1) or np.any(np.diff(times) <= 0)
            or len(set(excluded)) != len(excluded) or set(excluded) & set(ids.tolist())):
        raise ValueError("Invalid source frame/timestamp selection")
    by_time = {}
    for frame, state in zip(frames, states):
        stamp = float(frame["slam_time_s"])
        if frame["frame_id"] != state["frame_id"] or stamp != float(state["slam_time_s"]):
            raise ValueError("State/source frame or timestamp mismatch")
        if stamp in by_time:
            raise ValueError("Duplicate source timestamp")
        by_time[stamp] = (frame, state)
    seen, valid = set(), {}
    for row in read_csv(output / "slam/camera_trajectory.csv"):
        stamp = float(row["slam_time_s"])
        if stamp in seen or stamp not in by_time:
            raise ValueError("Duplicate or unknown trajectory timestamp")
        seen.add(stamp)
        frame, state = by_time[stamp]
        if row["metric"] != "1" or state["state"] != "2" or state["imu_initialized"] != "1":
            continue
        map_id = int(row["map_id"])
        if map_id < 0 or int(state["map_id"]) < 0:
            raise ValueError("Invalid map ID")
        valid[int(frame["frame_id"])] = (frame, state, map_id, pose_matrix(row))
    points, point_ids = {}, set()
    for row in read_csv(output / "slam/map_points.csv"):
        key = (int(row["map_id"]), int(row["point_id"]))
        xyz = [float(row[k]) for k in ("x", "y", "z")]
        if key in point_ids or key[0] < 0 or not np.isfinite(xyz).all():
            raise ValueError("Invalid or duplicate map point")
        point_ids.add(key)
        if int(row["observations"]) >= 3:
            points.setdefault(key[0], []).append(xyz)
    runs = []
    for frame_id, value in sorted(valid.items()):
        key = (value[2], int(value[1]["map_id"]))
        if not runs or frame_id != runs[-1][-1][0]+1 or key != runs[-1][-1][1]:
            runs.append([])
        runs[-1].append((frame_id, key, value))
    geometry_dir = output / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    map_outputs = []
    # A final ORB map is one coordinate system even across missing poses. The
    # existing NeonLoader can consume these sparse timestamps directly, keeping
    # one Viewer world while still processing every other image camera-locally.
    for map_id in sorted(set(v[2] for v in valid.values())):
        entries = [(i, v) for i, v in sorted(valid.items()) if v[2] == map_id]
        if len(entries) < 2:
            continue
        path = geometry_dir / f"map_{map_id:03d}.npz"
        data = dict(timestamps_ns=np.array([int(v[0]["time_ns"]) for _, v in entries], dtype=np.int64),
                    T_world_camera=np.stack([v[3] for _, v in entries]),
                    source_frame_ids=np.array([i for i, _ in entries], dtype=np.int64))
        if map_id in points:
            data["points_world"] = np.asarray(points[map_id], dtype=np.float64)
        np.savez_compressed(path, **data)
        adjacent = np.diff(data["source_frame_ids"]) == 1
        speed = np.linalg.norm(np.diff(data["T_world_camera"][:, :3, 3], axis=0), axis=1)
        speed = (speed / (np.diff(data["timestamps_ns"])*1e-9))[adjacent]
        map_outputs.append(dict(map_id=map_id, geometry=str(path), pose_frames=len(entries),
                                coverage=len(entries)/ (len(frames)+len(excluded)),
                                continuous_speed_m_s_p50_p99_max=np.percentile(speed, [50, 99, 100]).tolist() if len(speed) else None,
                                camera_extent_m=np.ptp(data["T_world_camera"][:, :3, 3], axis=0).tolist(),
                                point_count=len(points.get(map_id, []))))
    segments = []
    for run in runs:
        if len(run) < min_frames:
            continue
        values = [r[2] for r in run]
        stamps = np.array([int(v[0]["time_ns"]) for v in values], dtype=np.int64)
        if np.any(np.diff(stamps) <= 0):
            raise ValueError("Source timestamps must be increasing")
        poses = np.stack([v[3] for v in values])
        map_id = values[0][2]
        path = geometry_dir / f"geometry_segment_{len(segments):03d}.npz"
        data = dict(timestamps_ns=stamps, T_world_camera=poses,
                    source_frame_ids=np.array([r[0] for r in run], dtype=np.int64))
        if map_id in points:
            data["points_world"] = np.asarray(points[map_id], dtype=np.float64)
        np.savez_compressed(path, **data)
        speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / (np.diff(stamps)*1e-9)
        segments.append(dict(file=path.name, geometry=str(path), start_n=run[0][0], max_n=len(run),
                             map_id=map_id, runtime_map_id=run[0][1][1], session=0,
                             point_count=len(points.get(map_id, [])),
                             duration_s=int(stamps[-1]-stamps[0])*1e-9,
                             speed_m_s_p50_p99_max=np.percentile(speed, [50, 99, 100]).tolist()))
    total = len(frames) + len(manifest.get("excluded_without_imu", []))
    count = sum(s["max_n"] for s in segments)
    execution_path = output / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    summary = dict(recording=manifest["recording"], prepared=str(prepared), input_frames=total,
                   prepared_frames=len(frames), segments=segments,
                   usable_frames=count, coverage=count/total,
                   sessions=[dict(session=0, **native_report)],
                   imu_offset_ms_applied=manifest["imu_offset_ms_applied"],
                   imu_noise_source=manifest["imu_noise_source"],
                   accuracy="IMU-initialized candidates, not accuracy-qualified; inspect motion diagnostics; independent maps are not aligned",
                   standalone=dict(upstream_revision=REVISION, metric_candidates=len(valid),
                                   orphan_cull_guard=execution.get("orphan_cull_guard", False),
                                   map_outputs=map_outputs,
                                   final_metric_maps=sorted(set(v[2] for v in valid.values())),
                                   excluded_short_run_frames=len(valid)-count,
                                   coordinates="final optimized OpenCV scene camera to ORB world; metres; gravity -Z",
                                   interpolation=False))
    write_json(output / "pipeline.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, help="Prepared Neon images/IMU/calibration directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start_n", type=int, default=0)
    parser.add_argument("--max_n", type=int)
    parser.add_argument("--pace", type=float, default=1., help="1=recorded timing; 0=fastest (may starve mapping)")
    parser.add_argument("--descriptor-ties",choices=["native","source"],default="native")
    parser.add_argument("--light-maps",action="store_true",help="Compact pre-IMU point/pose summaries, no heavy snapshots")
    parser.add_argument("--visual-snapshots",action="store_true",help="Capture initial visual map/BA numeric states; requires --diagnostics")
    parser.add_argument("--visual-max-source",type=int,help="Stop visual observation after this source frame")
    parser.add_argument("--imu-snapshots", action="store_true", help="Capture replayable numeric IMU initialization graphs (requires --diagnostics)")
    parser.add_argument("--diagnostics", action="store_true", help="Use the separate observation-only diagnostic build")
    parser.add_argument("--lost-keyframes", choices=("preserve", "on", "off"), default="on",
                        help="on restores upstream default; preserve keeps prepared setting for comparisons")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--convert-only", action="store_true", help="Re-export a completed standalone run before Boxer inference")
    args = parser.parse_args()
    if (args.descriptor_ties!="native" or args.light_maps) and not args.diagnostics:
        parser.error("Descriptor tie experiment/light maps require --diagnostics")
    if args.visual_snapshots and not args.diagnostics:
        parser.error("--visual-snapshots requires --diagnostics")
    if args.imu_snapshots and not args.diagnostics:
        parser.error("--imu-snapshots requires --diagnostics")
    if args.start_n < 0 or (args.max_n is not None and args.max_n < 2) or args.min_frames < 2 or not math.isfinite(args.pace) or args.pace < 0:
        parser.error("Invalid frame selection, min-frames, or pace")
    output = args.output.resolve()
    if not args.convert_only:
        if args.prepared is None:
            parser.error("--prepared is required unless --convert-only")
        build_dir = RUNTIME / ("build_diagnostics" if args.diagnostics else "build")
        binary = build_dir / "neon_standalone"
        if not binary.is_file():
            parser.error("Build first: bash standalone/orbslam3/build.sh")
        if args.diagnostics and "NEON_DIAGNOSTICS:BOOL=ON" not in (build_dir / "CMakeCache.txt").read_text():
            parser.error("--diagnostics requires NEON_DIAGNOSTICS=ON bash standalone/orbslam3/build.sh")
        if output.exists() and any(output.iterdir()):
            parser.error("Output must be empty; existing results are preserved")
        output.mkdir(parents=True, exist_ok=True)
        count = prepare_subset(args.prepared.resolve(), output / "prepared", args.start_n,
                               args.max_n, args.lost_keyframes)
        command = [str(binary), str(RUNTIME / "source/Vocabulary/ORBvoc.txt"),
                   str(output / "prepared"), str(output / "slam"), str(args.pace)]
        write_json(output / "execution.json", dict(command=command, source_prepared=str(args.prepared.resolve()),
                   upstream_revision=REVISION, lost_keyframes=args.lost_keyframes,
                   orphan_cull_guard="NEON_ORPHAN_GUARD:BOOL=ON" in (build_dir / "CMakeCache.txt").read_text(),
                   descriptor_ties=args.descriptor_ties,light_maps=args.light_maps,diagnostics=args.diagnostics, imu_snapshots=args.imu_snapshots, visual_snapshots=args.visual_snapshots, visual_max_source=args.visual_max_source,
                   library_sha256=hashlib.sha256((build_dir / "libstandalone_orb.so").read_bytes()).hexdigest(),
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   camera_yaml_sha256=hashlib.sha256((output / "prepared/camera.yaml").read_bytes()).hexdigest()))
        print("Running:", shlex.join(command), flush=True)
        env = os.environ.copy()
        env["NEON_DESCRIPTOR_TIES"]=args.descriptor_ties
        env["NEON_LIGHT_MAPS"]="1" if args.light_maps else "0"
        env["NEON_VISUAL_SNAPSHOTS"]="1" if args.visual_snapshots else "0"
        if args.visual_max_source is not None:env["NEON_VISUAL_MAX_SOURCE"]=str(args.visual_max_source)
        else:env.pop("NEON_VISUAL_MAX_SOURCE",None)
        env["NEON_IMU_SNAPSHOTS"] = "1" if args.imu_snapshots else "0"
        env["LD_LIBRARY_PATH"] = str(RUNTIME / "prefix/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        with (output / "slam.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=output)
            try:
                while True:
                    try:
                        code = process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        path = output / "slam/states.csv"
                        done = max(0, sum(1 for _ in path.open())-1) if path.exists() else 0
                        message = f"[ORB-SLAM3] {min(99.9, done/count*100):.1f}% ({done}/{count}); log: {output/'slam.log'}"
                        print(message, flush=True)
                        with (output / "progress.log").open("a") as stream:
                            stream.write(message+"\n")
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            state_path = output / "slam/states.csv"
            completed = read_csv(state_path) if state_path.exists() else []
            write_json(output / "failure.json", dict(status="failed", returncode=code,
                       signal=-code if code < 0 else None, processed_frames=len(completed),
                       last_source_frame=int(completed[-1]["frame_id"]) if completed else None,
                       log=str(output / "slam.log"), geometry_exported=False))
            raise RuntimeError(f"ORB-SLAM3 exited {code}; inspect {output/'slam.log'}")
    summary = export_geometry(output, args.min_frames)
    print(f"Metric-flagged candidate frames: {summary['usable_frames']}/{summary['input_frames']} ({summary['coverage']:.2%}); this is not an accuracy measurement", flush=True)
    for item in summary["standalone"]["map_outputs"]:
        print(f"Map {item['map_id']}: {item['pose_frames']} poses; continuous speed p50/p99/max (m/s): {item['continuous_speed_m_s_p50_p99_max']}", flush=True)
    print("Boxer (CUDA required; processes all candidate maps, so review their geometry first):", flush=True)
    print(shlex.join([str(ROOT/".venv/bin/python"), str(ROOT/"run_neon_pipeline.py"),
          "--input", summary["recording"], "--output", str(output), "--boxer-only", "--require_gpu", "--skip_viz", "--gaze_targets"]))


if __name__ == "__main__":
    main()
