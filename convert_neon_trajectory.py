#!/usr/bin/env python3
"""Export continuous, metric ORB-SLAM3 camera-pose segments for NeonLoader."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.neon_vio import (
    ORB_SLAM3_COMMIT, check_times, read_csv, split_valid_runs, write_json,
)


def convert(prepared, trajectory, states, output, min_frames=30):
    prepared, output = Path(prepared), Path(output)
    if min_frames < 2:
        raise ValueError("min_frames must be >= 2")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("orb_slam3_commit") != ORB_SLAM3_COMMIT:
        raise ValueError("Prepared input uses an unsupported ORB-SLAM3 revision")
    if manifest.get("pose_frame") != "scene_camera":
        raise ValueError("Expected scene-camera poses")
    frames = read_csv(prepared / "frames.csv")
    frame_ids = np.array([int(row["frame_id"]) for row in frames], dtype=np.int64)
    times = np.array([int(row["time_ns"]) for row in frames], dtype=np.int64)
    check_times(times, "Source frames")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Source frame IDs must be strictly increasing")
    slam_times = [float(row["slam_time_s"]) for row in frames]
    if not np.isfinite(slam_times).all() or np.any(np.diff(slam_times) <= 0):
        raise ValueError("SLAM times must be finite and strictly increasing")
    time_index = {stamp: i for i, stamp in enumerate(slam_times)}
    poses = [None] * len(frames)
    state_by_frame = {}
    for row in read_csv(states):
        index = int(row["frame_id"])
        if index in state_by_frame:
            raise ValueError("Duplicate frame ID in tracking states")
        state_by_frame[index] = row
    if set(state_by_frame) != set(frame_ids.tolist()):
        raise ValueError("Tracking states must cover the exact prepared frame IDs")
    for i, index in enumerate(frame_ids):
        row = state_by_frame[int(index)]
        if float(row["slam_time_s"]) != slam_times[i]:
            raise ValueError("Tracking-state timestamp does not match prepared frame")
    ordered_states = [state_by_frame[int(index)] for index in frame_ids]
    latencies = [float(row["track_ms"]) for row in state_by_frame.values() if "track_ms" in row]
    if latencies and (not np.isfinite(latencies).all() or min(latencies) < 0):
        raise ValueError("Invalid tracking latency")
    seen = set()
    map_ids = np.full(len(frames), -1, dtype=np.int64)
    runtime_maps = np.full(len(frames), -1, dtype=np.int64)
    valid = np.zeros(len(frames), dtype=bool)
    for row in read_csv(trajectory):
        stamp = float(row["slam_time_s"])
        if stamp not in time_index:
            raise ValueError("Trajectory timestamp has no exact prepared-frame match")
        i = time_index[stamp]
        if i in seen:
            raise ValueError("Duplicate trajectory timestamp")
        seen.add(i)
        state = state_by_frame.get(int(frame_ids[i]))
        if state is None or float(state["slam_time_s"]) != stamp:
            raise ValueError("Missing or mismatched tracking-state timestamp")
        # Final map initialization alone cannot certify pre-initialization frame
        # history. Require both the runtime and the final map to be metric.
        if int(row["metric"]) != 1 or int(state["imu_initialized"]) != 1 or int(state["state"]) != 2:
            continue
        q = np.array([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
        t = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
        if not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
            raise ValueError("Invalid metric camera pose")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q).as_matrix()
        pose[:3, 3] = t
        poses[i] = pose
        map_ids[i] = int(row["map_id"])
        runtime_maps[i] = int(state["map_id"])
        if map_ids[i] < 0 or runtime_maps[i] < 0:
            raise ValueError("Metric pose has an invalid map ID")
        valid[i] = True
    # Runtime resets also split a run, even when maps subsequently merge.
    pairs = list(zip(map_ids.tolist(), runtime_maps.tolist()))
    runs = split_valid_runs(frame_ids, pairs, valid)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    segments = []
    for run in runs:
        if len(run) < min_frames:
            continue
        transforms = np.stack([poses[i] for i in run])
        # Translate the origin only; retain ORB's gravity-aligned world axes.
        transforms[:, :3, 3] -= transforms[0, :3, 3].copy()
        dt = np.diff(times[run]).astype(np.float64) * 1e-9
        speed = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1) / dt
        angles = Rotation.from_matrix(
            transforms[:-1, :3, :3].transpose(0, 2, 1) @ transforms[1:, :3, :3]
        ).magnitude()
        filename = f"geometry_segment_{len(segments):03d}.npz"
        np.savez_compressed(output / filename, timestamps_ns=times[run],
                            T_world_camera=transforms, source_frame_ids=frame_ids[run])
        segments.append({
            "file": filename, "start_n": int(frame_ids[run[0]]),
            "max_n": len(run), "map_id": int(map_ids[run[0]]),
            "runtime_map_id": int(runtime_maps[run[0]]),
            "first_timestamp_ns": int(times[run[0]]), "last_timestamp_ns": int(times[run[-1]]),
            "duration_s": float(np.sum(dt)),
            "speed_m_s_p50_p99_max": np.percentile(speed, [50, 99, 100]).tolist(),
            "rotation_step_deg_p50_p99_max": np.degrees(np.percentile(angles, [50, 99, 100])).tolist(),
            "path_length_m": float(np.sum(speed * dt)),
        })
    report = {
        "schema_version": 1, "prepared": str(prepared.resolve()),
        "trajectory": str(Path(trajectory).resolve()), "states": str(Path(states).resolve()),
        "orb_slam3_commit": ORB_SLAM3_COMMIT,
        "pose_frame": "scene_camera", "world_frame": "ORB gravity-aligned Z-up; translated origin",
        "units": "metres", "timestamps": "original scene .time int64 ns",
        "input_frames": len(frames), "metric_tracked_frames": int(valid.sum()),
        "runtime_metric_ok_frames": sum(
            row["state"] == "2" and row["imu_initialized"] == "1"
            for row in ordered_states
        ),
        # An active-map reset can retain its map ID. Record initialization
        # regressions separately rather than treating map count as reset count.
        "runtime_imu_deinitializations": sum(
            previous["imu_initialized"] == "1" and current["imu_initialized"] == "0"
            for previous, current in zip(ordered_states, ordered_states[1:])
        ),
        "tracking_state_counts": dict(Counter(row["state"] for row in state_by_frame.values())),
        "runtime_map_count": len(set(row["map_id"] for row in state_by_frame.values())),
        "tracking_latency_ms_p50_p99_max": np.percentile(latencies, [50, 99, 100]).tolist() if latencies else None,
        "coverage": float(valid.mean()), "min_frames": min_frames,
        "excluded_frames": int((~valid).sum()), "segments": segments,
        "interpolated_frames": 0, "points_exported": False,
        "imu_noise_source": manifest.get("imu_noise_source"),
        "extrinsics_source": manifest.get("extrinsics_source"),
        "physical_synchronization": manifest.get("physical_synchronization"),
        "accuracy": "unvalidated: no ground-truth trajectory supplied",
    }
    write_json(output / "report.json", report)
    if not segments:
        raise ValueError(f"No continuous metric segment >= {min_frames} frames; inspect {output / 'report.json'}")
    for segment in segments:
        print(f"{output / segment['file']}: --start_n {segment['start_n']} --max_n {segment['max_n']}")
    print(f"Metric tracking coverage: {valid.sum()}/{len(frames)} ({valid.mean():.1%})")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-frames", type=int, default=30)
    args = parser.parse_args()
    try:
        convert(args.prepared, args.trajectory, args.states, args.output, args.min_frames)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
