"""Small, explicit contracts shared by Neon VIO preparation and conversion."""

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ORB_SLAM3_COMMIT = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"


def recording_path(value):
    path = Path(value).expanduser()
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "sample_data" / value
    if not path.is_dir():
        raise ValueError(f"Recording directory not found: {value}")
    return path.resolve()


def scene_to_imu():
    """Manufacturer nominal geometry, NOT a measured world trajectory.

    https://docs.pupil-labs.com/alpha-lab/imu-transformations/
    """
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", -102, degrees=True).as_matrix()
    transform[:3, 3] = [0, -0.0013, -0.00662]
    return transform


def relative_seconds(timestamps_ns, origin_ns):
    values = np.asarray(timestamps_ns)
    if values.dtype != np.dtype("int64"):
        raise ValueError("Timestamps must be int64 nanoseconds")
    # Subtract the epoch before float conversion: individual ns stay distinct.
    return (values - np.int64(origin_ns)).astype(np.float64) * 1e-9


def check_times(values, name):
    values = np.asarray(values)
    if values.dtype != np.dtype("int64") or values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} needs at least two int64 timestamps")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} timestamps must be strictly increasing")


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Transform must be a finite 4x4 matrix")
    r = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(r), 1, atol=1e-5)
    ):
        raise ValueError("Transform must be rigid and right handed")
    return matrix


def rectification(K, distortion, source_size, width):
    sw, sh = source_size
    height = max(1, round(sh * width / sw))
    corrected = np.asarray(K, dtype=np.float64).copy()
    scale = np.array([width / sw, height / sh])
    corrected[0, 0] *= scale[0]
    corrected[1, 1] *= scale[1]
    corrected[:2, 2] = (corrected[:2, 2] + 0.5) * scale - 0.5
    maps = cv2.initUndistortRectifyMap(
        K, distortion, None, corrected, (width, height), cv2.CV_32FC1
    )
    return corrected, maps, height


def imu_in_si(imu):
    gyro = np.asarray(imu.angular_velocity, dtype=np.float64) * np.pi / 180
    accel = np.asarray(imu.acceleration, dtype=np.float64) * 9.80665
    if gyro.ndim != 2 or gyro.shape[1] != 3 or accel.shape != gyro.shape:
        raise ValueError("IMU acceleration and angular velocity must be matching Nx3 arrays")
    if not np.isfinite(gyro).all() or not np.isfinite(accel).all():
        raise ValueError("IMU contains nonfinite acceleration or angular velocity")
    return gyro, accel


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_orb_settings(path, settings, transform):
    """ORB uses dotted keys, which OpenCV 5 FileStorage refuses to write."""
    lines = ["%YAML:1.0", "---"]
    for key, value in settings.items():
        if not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError("Invalid ORB setting name")
        if isinstance(value, str):
            encoded = json.dumps(value)
        elif isinstance(value, (int, np.integer)):
            encoded = str(int(value))
        else:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("Nonfinite ORB setting")
            encoded = repr(value)
        lines.append(f"{key}: {encoded}")
    transform = validate_transform(transform)
    # Upstream Converter reads this matrix through cv::Mat::at<float>.
    lines.extend(["IMU.T_b_c1: !!opencv-matrix", "   rows: 4", "   cols: 4",
                  "   dt: f", "   data: " + json.dumps(transform.ravel().tolist())])
    Path(path).write_text("\n".join(lines) + "\n")


def noise_parameters(path, experimental):
    keys = ("NoiseGyro", "NoiseAcc", "GyroWalk", "AccWalk")
    if path:
        data = json.loads(Path(path).expanduser().read_text())
        values = {key: float(data[key]) for key in keys}
        source = data.get("source", "user supplied; calibration status unspecified")
    elif experimental:
        # Explicit opt-in baseline, not a device-specific calibration. These are
        # conservative order-of-magnitude starting values for a pilot experiment.
        values = dict(NoiseGyro=0.001, NoiseAcc=0.02, GyroWalk=0.0001, AccWalk=0.001)
        source = "uncalibrated experimental baseline; validate before research use"
    else:
        raise ValueError("Supply --imu-noise JSON or explicitly --experimental-noise")
    if any(not np.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("IMU noise densities and random walks must be finite and > 0")
    return values, source


def split_valid_runs(frame_ids, map_ids, valid):
    """Never bridge a lost frame, a source-frame gap, or a map boundary."""
    runs, current = [], []
    for i, good in enumerate(valid):
        contiguous = not current or (
            frame_ids[i] == frame_ids[current[-1]] + 1
            and map_ids[i] == map_ids[current[-1]]
        )
        if current and (not good or not contiguous):
            runs.append(np.array(current, dtype=np.int64))
            current = []
        if good:
            current.append(i)
    if current:
        runs.append(np.array(current, dtype=np.int64))
    return runs
