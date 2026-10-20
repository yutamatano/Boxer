#!/usr/bin/env python3
"""Check Neon geometry against IMU; optionally measure error against a reference."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.neon_vio import check_times, read_csv, validate_transform, write_json


def integrate_gyro(times, gyro, begin, end):
    if begin < times[0] or end > times[-1] or not begin < end:
        raise ValueError("IMU does not cover the requested interval")
    boundaries = np.r_[begin, times[(times > begin) & (times < end)], end]
    midpoints = (boundaries[:-1] + boundaries[1:]) * 0.5
    omega = np.column_stack(
        [np.interp(midpoints, times, gyro[:, axis]) for axis in range(3)]
    )
    delta = Rotation.identity()
    for vector, dt in zip(omega, np.diff(boundaries)):
        delta = delta * Rotation.from_rotvec(vector * dt)
    return delta.as_matrix()


def load_geometry(path, *, allow_single=False):
    with np.load(path, allow_pickle=False) as data:
        stamps = data["timestamps_ns"].copy()
        poses = np.asarray(data["T_world_camera"], dtype=np.float64)
        points = (
            np.asarray(data["points_world"], dtype=np.float64)
            if "points_world" in data
            else None
        )
    if not (allow_single and stamps.shape == (1,) and stamps.dtype == np.dtype("int64")):
        check_times(stamps, "Geometry timestamps")
    if poses.shape != (len(stamps), 4, 4):
        raise ValueError("Geometry needs one rigid camera pose per timestamp")
    for pose in poses:
        validate_transform(pose)
    if points is not None and (
        points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
    ):
        raise ValueError("Evaluation requires finite static map points of shape (N, 3)")
    return stamps, poses, points


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(
        zip(("p50", "p95", "max"), np.percentile(values, [50, 95, 100]).tolist())
    )


def reference_error(stamps, poses, reference):
    ref_times, ref_poses, _ = load_geometry(reference)
    indices = np.searchsorted(ref_times, stamps)
    if np.any(indices >= len(ref_times)) or not np.array_equal(
        ref_times[indices], stamps
    ):
        raise ValueError("Reference must cover every evaluated exact source timestamp")
    target = ref_poses[indices]
    # SLAM world origin/yaw are arbitrary. Align with one full reference pose,
    # using a rigid transform only: fitting scale would hide metric scale errors.
    rotation = target[0, :3, :3] @ poses[0, :3, :3].T
    translation = target[0, :3, 3] - rotation @ poses[0, :3, 3]
    positions = poses[:, :3, 3] @ rotation.T + translation
    orientations = rotation @ poses[:, :3, :3]
    position_errors = np.linalg.norm(positions - target[:, :3, 3], axis=1)
    rotation_errors = np.degrees(
        Rotation.from_matrix(
            orientations.transpose(0, 2, 1) @ target[:, :3, :3]
        ).magnitude()
    )
    return {
        "reference": str(Path(reference).resolve()),
        "alignment": "first camera pose, rigid transform, no scale fitting",
        "position_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
        "position_error_m": stats(position_errors),
        "rotation_error_deg": stats(rotation_errors),
    }


def evaluate(prepared, geometry, output, reference=None, *, source_interval=None):
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    frames = read_csv(prepared / "frames.csv")
    by_timestamp = {int(row["time_ns"]): row for row in frames}
    from utils.neon_geometry import geometry_metadata

    with np.load(geometry, allow_pickle=False) as data:
        meta = geometry_metadata(data)
    stamps, poses, points = load_geometry(
        geometry, allow_single=meta.get("pose_backend") == "colmap"
    )
    if source_interval is not None:
        start, count = source_interval
        keep = np.array(
            [
                int(stamp) in by_timestamp
                and start <= int(by_timestamp[int(stamp)]["frame_id"]) < start + count
                for stamp in stamps
            ]
        )
        stamps, poses = stamps[keep], poses[keep]
        if len(stamps) != count:
            raise ValueError("Geometry does not cover the complete evaluation interval")
    metric = bool(meta.get("metric"))
    if reference and not metric:
        raise ValueError(
            "Nonmetric preview cannot use metric reference-error evaluation"
        )
    if any(int(stamp) not in by_timestamp for stamp in stamps):
        raise ValueError("Geometry contains a timestamp outside prepared input")
    selected = [by_timestamp[int(stamp)] for stamp in stamps]
    frame_ids = np.array([int(row["frame_id"]) for row in selected])
    if np.any(np.diff(frame_ids) != 1):
        raise ValueError("Evaluate one continuous source interval at a time")
    physical_times = np.array([float(row["slam_time_s"]) for row in selected])
    imu = (
        np.genfromtxt(prepared / "imu.csv", delimiter=",", skip_header=1)
        if (prepared / "imu.csv").exists()
        else np.empty((0, 7))
    )
    if (
        imu.ndim != 2
        or imu.shape[1] != 7
        or not np.isfinite(imu).all()
        or np.any(np.diff(imu[:, 0]) <= 0)
    ):
        raise ValueError("Invalid prepared IMU samples")
    transform = validate_transform(manifest["T_imu_camera"])
    rotation = transform[:3, :3]
    residuals = []
    for i, (begin, end) in enumerate(zip(physical_times[:-1], physical_times[1:])):
        if not len(imu):
            break
        delta = integrate_gyro(imu[:, 0], imu[:, 1:4], begin, end)
        imu_relative = rotation.T @ delta.T @ rotation
        visual_relative = poses[i + 1, :3, :3].T @ poses[i, :3, :3]
        residuals.append(
            np.degrees(
                Rotation.from_matrix(visual_relative @ imu_relative.T).magnitude()
            )
        )
    speeds = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.diff(
        physical_times
    )
    visible = []
    if points is not None:
        K = np.array(manifest["K_rectified"])
        width, height = manifest["image_size"]
        for pose in poses:
            camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
            camera_points = camera_points[camera_points[:, 2] > 0]
            pixels = camera_points @ K.T
            pixels = pixels[:, :2] / pixels[:, 2:3]
            visible.append(
                int(
                    (
                        (pixels[:, 0] >= 0)
                        & (pixels[:, 0] < width)
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] < height)
                    ).sum()
                )
            )
    report = {
        "schema_version": 1,
        "geometry": str(Path(geometry).resolve()),
        "frames": len(stamps),
        "start_n": int(frame_ids[0]),
        "timestamps": "exact original int64 scene timestamps",
        "metric": metric,
        "pose_units": meta.get("pose_units"),
        ("speed_m_s" if metric else "speed_visual_units_s"): stats(speeds),
        "gyro_rotation_residual_deg": stats(residuals),
        "gyro_comparison": (
            "visual/gyro rotation consistency; calibration and synchronization unverified"
            if meta.get("pose_backend") == "colmap" else
            "internal consistency; same IMU also used by SLAM, not independent accuracy"
        ),
        "map_point_count": len(points) if points is not None else 0,
        "projected_points_in_image": stats(visible),
        "projection_semantics": "static map support, not verified current visibility",
        "imu_offset_ms_applied": manifest["imu_offset_ms_applied"],
        "imu_noise_source": manifest["imu_noise_source"],
        "reference_accuracy": reference_error(stamps, poses, reference)
        if reference
        else {
            "status": "not measured: no independent reference trajectory supplied",
        },
    }
    write_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference", help="NPZ: exact timestamps_ns and metric T_world_camera"
    )
    args = parser.parse_args()
    try:
        result = evaluate(args.prepared, args.geometry, args.output, args.reference)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
