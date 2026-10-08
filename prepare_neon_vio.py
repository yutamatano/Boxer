#!/usr/bin/env python3
"""Prepare rectified Neon scene frames and SI IMU for offline ORB-SLAM3."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from utils.neon_vio import (
    ORB_SLAM3_COMMIT, check_times, imu_in_si, noise_parameters, recording_path,
    rectification, relative_seconds, scene_to_imu, write_json, write_orb_settings,
)


def rotation_sync_check(pairs, imu_times, gyro, T_imu_camera):
    """Diagnostic rotation residuals; an inconclusive fit is NOT synchronization.

    Each pair is (previous_time_s, current_time_s, visual current-from-previous R).
    The candidate offset shifts IMU timestamps relative to the chosen image clock.
    """
    if len(pairs) < 8:
        return {"status": "inconclusive", "visual_pairs": len(pairs)}
    Rbc = T_imu_camera[:3, :3]
    candidates = np.arange(-0.1, 0.1001, 0.005)
    scores = []
    for shift in candidates:
        residuals = []
        for t0, t1, visual in pairs:
            t0, t1 = t0 - shift, t1 - shift
            if t0 < imu_times[0] or t1 > imu_times[-1]:
                continue
            middle = imu_times[(imu_times > t0) & (imu_times < t1)]
            boundaries = np.r_[t0, middle, t1]
            midpoints = (boundaries[:-1] + boundaries[1:]) * 0.5
            omega = np.column_stack([
                np.interp(midpoints, imu_times, gyro[:, axis]) for axis in range(3)
            ])
            delta = Rotation.identity()
            for vector, dt in zip(omega, np.diff(boundaries)):
                delta = delta * Rotation.from_rotvec(vector * dt)
            expected = Rbc.T @ delta.as_matrix().T @ Rbc
            residuals.append(Rotation.from_matrix(visual @ expected.T).magnitude())
        scores.append(float(np.median(residuals)) if residuals else np.pi)
    best = int(np.argmin(scores))
    improvement = float(scores[len(candidates) // 2] - scores[best])
    return {
        "status": "diagnostic_only; verify on independent moving intervals",
        "visual_pairs": len(pairs),
        "additional_imu_offset_ms_candidate": float(candidates[best] * 1000),
        "median_rotation_residual_deg": float(np.degrees(scores[best])),
        "improvement_over_current_offset_deg": float(np.degrees(improvement)),
        "at_search_boundary": best in (0, len(candidates) - 1),
        "offsets_ms": (candidates * 1000).round(6).tolist(),
        "residuals_deg": np.degrees(scores).tolist(),
        "applied_automatically": False,
    }


def visual_rotation(previous, current, K):
    features = cv2.goodFeaturesToTrack(previous, 800, 0.01, 8)
    if features is None or len(features) < 30:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
    if tracked is None:
        return None
    a = features[status.ravel() == 1].reshape(-1, 2)
    b = tracked[status.ravel() == 1].reshape(-1, 2)
    if len(a) < 30:
        return None
    essential, mask = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
    if essential is None or essential.shape != (3, 3):
        return None
    count, rotation, _, _ = cv2.recoverPose(essential, a, b, K, mask=mask)
    return rotation if count >= 30 else None


def prepare(args):
    import pupil_labs.neon_recording as nr
    from pupil_labs.neon_recording.utils import find_sorted_multipart_files

    root = recording_path(args.input)
    noise, noise_source = noise_parameters(args.imu_noise, args.experimental_noise)
    if args.start_n < 0 or args.max_n < 2 or args.width < 32:
        raise ValueError("Require start_n >= 0, max_n >= 2, width >= 32")
    out = Path(args.output or f"output/neon_vio/{root.name}/prepared").resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output is not empty; choose a new directory: {out}")
    rec = nr.open(root)
    try:
        scene = rec.scene
        times = np.concatenate([
            np.fromfile(time_path, dtype="<i8")
            for _, time_path in find_sorted_multipart_files(root, "Neon Scene Camera v1", ".mp4")
        ])
        aux_parts = []
        for _, time_path in find_sorted_multipart_files(root, "Neon Scene Camera v1", ".mp4"):
            aux_path = time_path.with_suffix(".time_aux")
            part_count = len(np.fromfile(time_path, dtype="<i8"))
            aux_part = (np.fromfile(aux_path, dtype="<i8") if aux_path.exists()
                        else np.full(part_count, -1, dtype=np.int64))
            if len(aux_part) != part_count:
                raise ValueError("Scene .time_aux count differs from its video part")
            aux_parts.append(aux_part)
        aux = np.concatenate(aux_parts)
        check_times(times, "Scene .time")
        if len(times) != len(scene) or len(aux) != len(times) or not np.array_equal(times, scene.time):
            raise ValueError("Scene video/.time/.time_aux count or frame mapping mismatch")
        physical = times if args.time_source == "primary" else aux
        if np.any(physical < 0):
            raise ValueError("Selected image clock is missing")
        check_times(physical, "Selected image clock")
        imu = rec.imu[:]
        imu_ns = np.asarray(imu.time, dtype=np.int64)
        check_times(imu_ns, "IMU")
        gyro, accel = imu_in_si(imu)
        if len(gyro) != len(imu_ns):
            raise ValueError("IMU timestamp and measurement count mismatch")
        quaternion = np.asarray(imu.rotation, dtype=np.float64)
        norms = np.linalg.norm(quaternion, axis=1)
        valid_quaternion = np.isfinite(quaternion).all(axis=1) & (np.abs(norms - 1) < 0.01)
        origin = int(physical[0])
        image_seconds = relative_seconds(physical, origin)
        imu_seconds = relative_seconds(imu_ns, origin) + args.imu_offset_ms * 1e-3
        indices = np.arange(args.start_n, min(len(times), args.start_n + args.max_n))
        covered = (image_seconds[indices] > imu_seconds[0]) & (image_seconds[indices] < imu_seconds[-1])
        excluded = indices[~covered].tolist()
        indices = indices[covered]
        if len(indices) < 2:
            raise ValueError("Fewer than two selected images have bracketing IMU samples")
        cal = rec.calibration
        if cal is None:
            raise ValueError("calibration.bin is missing")
        K = np.asarray(cal.scene_camera_matrix, dtype=np.float64)
        distortion = np.asarray(cal.scene_distortion_coefficients, dtype=np.float64)
        if not np.isfinite(K).all() or not np.isfinite(distortion).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Invalid camera intrinsics")
        first = scene[int(indices[0])].bgr
        h, w = first.shape[:2]
        corrected, maps, height = rectification(K, distortion, (w, h), args.width)
        Tbc = scene_to_imu()
        if args.extrinsics:
            from utils.neon_vio import validate_transform
            Tbc = validate_transform(json.loads(Path(args.extrinsics).read_text()))
        out.mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir()
        begin = max(0, int(np.searchsorted(imu_seconds, image_seconds[indices[0]])) - 1)
        end = min(len(imu_seconds), int(np.searchsorted(imu_seconds, image_seconds[indices[-1]], side="right")) + 1)
        imu_data = np.column_stack([imu_seconds[begin:end], gyro[begin:end], accel[begin:end]])
        np.savetxt(out / "imu.csv", imu_data, delimiter=",", fmt="%.17g",
                   header="time_s,gx_rad_s,gy_rad_s,gz_rad_s,ax_m_s2,ay_m_s2,az_m_s2", comments="")
        means, counts, pairs = [], [], []
        detector = cv2.ORB_create(1000)
        cv2.setRNGSeed(0)
        previous = None
        with (out / "frames.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_id", "time_ns", "time_aux_ns", "slam_time_s", "image", "gray_mean", "orb_features"])
            for n, index in enumerate(indices):
                image = first if n == 0 else scene[int(index)].bgr
                if image.shape[:2] != (h, w):
                    raise ValueError("Scene resolution changes within the selected interval")
                gray = cv2.cvtColor(cv2.remap(image, *maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
                filename = f"images/{int(index):010d}.png"
                if not cv2.imwrite(str(out / filename), gray):
                    raise OSError(f"Cannot write {filename}")
                mean = float(gray.mean())
                count = len(detector.detect(gray, None))
                means.append(mean)
                counts.append(count)
                writer.writerow([int(index), int(times[index]), int(aux[index]),
                                 format(image_seconds[index], ".17g"), filename, mean, count])
                if args.check_sync and previous is not None and int(index) % 5 == 0:
                    rotation = visual_rotation(previous[1], gray, corrected)
                    if rotation is not None:
                        pairs.append((previous[0], image_seconds[index], rotation))
                previous = (image_seconds[index], gray)
                if n % 100 == 0:
                    print(f"Prepared {n + 1}/{len(indices)} frames", flush=True)
        frequency = float((len(imu_ns) - 1) / ((imu_ns[-1] - imu_ns[0]) * 1e-9))
        fps = float((len(indices) - 1) / (image_seconds[indices[-1]] - image_seconds[indices[0]]))
        settings = {
            "File.version": "1.0", "Camera.type": "PinHole",
            "Camera1.fx": corrected[0, 0], "Camera1.fy": corrected[1, 1],
            "Camera1.cx": corrected[0, 2], "Camera1.cy": corrected[1, 2],
            "Camera.width": args.width, "Camera.height": height,
            "Camera.fps": round(fps), "Camera.RGB": 0,
            "IMU.Frequency": frequency, "IMU.InsertKFsWhenLost": 0,
            "ORBextractor.nFeatures": 2000, "ORBextractor.scaleFactor": 1.2,
            "ORBextractor.nLevels": 8, "ORBextractor.iniThFAST": 20,
            "ORBextractor.minThFAST": 7, "Viewer.KeyFrameSize": 0.05,
            "Viewer.KeyFrameLineWidth": 1.0, "Viewer.GraphLineWidth": 0.9,
            "Viewer.PointSize": 2.0, "Viewer.CameraSize": 0.08,
            "Viewer.CameraLineWidth": 3.0, "Viewer.ViewpointX": 0.0,
            "Viewer.ViewpointY": -0.7, "Viewer.ViewpointZ": -1.8,
            "Viewer.ViewpointF": 500.0,
        }
        settings.update({f"IMU.{k}": v for k, v in noise.items()})
        write_orb_settings(out / "camera.yaml", settings, Tbc)
        report = {
            "schema_version": 1, "recording": str(root), "orb_slam3_commit": ORB_SLAM3_COMMIT,
            "pose_frame": "scene_camera", "time_origin_ns": origin,
            "image_clock": args.time_source, "imu_offset_ms_applied": args.imu_offset_ms,
            "timestamps_exported": "original scene .time, int64 nanoseconds",
            "selected_start_n": args.start_n, "selected_max_n": args.max_n,
            "prepared_frames": len(indices), "excluded_without_imu": excluded,
            "image_size": [args.width, height], "K_rectified": corrected.tolist(),
            "K_original": K.tolist(), "distortion": distortion.tolist(),
            "T_imu_camera": Tbc.tolist(),
            "extrinsics_source": str(args.extrinsics) if args.extrinsics else "manufacturer nominal",
            "imu_units": {"gyro": "rad/s", "acceleration": "m/s^2, includes gravity"},
            "quaternion_audit": {"valid_rows": int(valid_quaternion.sum()),
                                 "invalid_rows": int((~valid_quaternion).sum()),
                                 "used_by_slam": False},
            "imu_noise": noise, "imu_noise_source": noise_source,
            "scene_fps": fps, "imu_mean_frequency_hz": frequency,
            "imu_dt_ms_percentiles": np.percentile(np.diff(imu_ns) * 1e-6, [0, 50, 99, 100]).tolist(),
            "dark_frames_mean_below_5": int(np.sum(np.array(means) < 5)),
            "feature_count_percentiles": np.percentile(counts, [0, 50, 100]).tolist(),
            "physical_synchronization": "not independently verified",
            "sync_check": rotation_sync_check(pairs, imu_seconds, gyro, Tbc) if args.check_sync else {"status": "not run"},
        }
        write_json(out / "manifest.json", report)
        print(f"Prepared input: {out}")
        print(f"IMU noise: {noise_source}")
        print(f"Sync diagnostic: { {k: v for k, v in report['sync_check'].items() if k not in ('offsets_ms', 'residuals_deg')} }")
        return report
    finally:
        rec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--start_n", type=int, default=0)
    parser.add_argument("--max_n", type=int, default=1200)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--time-source", choices=["primary", "aux"], default="primary")
    parser.add_argument("--imu-offset-ms", type=float, default=0.0,
                        help="Add to IMU timestamps; source .time labels remain unchanged")
    parser.add_argument("--imu-noise", help="JSON: NoiseGyro, NoiseAcc, GyroWalk, AccWalk")
    parser.add_argument("--experimental-noise", action="store_true",
                        help="Explicitly use an uncalibrated pilot baseline")
    parser.add_argument("--extrinsics", help="JSON 4x4 camera-to-IMU transform in metres")
    parser.add_argument("--check-sync", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.imu_offset_ms):
        parser.error("IMU time offset must be finite")
    try:
        prepare(args)
    except (ValueError, OSError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
