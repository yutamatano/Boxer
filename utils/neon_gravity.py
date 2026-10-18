"""Neon IMU tilt for camera-local predictions; never a position estimate."""

import numpy as np
from scipy.spatial.transform import Rotation

from utils.neon_vio import scene_to_imu, validate_transform


MAX_IMU_GAP_NS = 50_000_000


def camera_gravity(frame_times_ns, imu_times_ns, quaternions_xyzw,
                   max_gap_ns=MAX_IMU_GAP_NS, T_imu_camera=None):
    """SLERP adjacent native IMU orientations at image capture timestamps.

    The official reader exposes xyzw, although the on-disk fields are wxyz.
    Invalid/unbracketed samples and gaps over 50 ms return NaN, not an
    extrapolated orientation. Absolute compass heading cancels from gravity.
    """
    frames, times = np.asarray(frame_times_ns), np.asarray(imu_times_ns)
    q = np.asarray(quaternions_xyzw, dtype=np.float64)
    if (frames.ndim != 1 or times.ndim != 1
            or frames.dtype != np.dtype("int64") or times.dtype != np.dtype("int64")
            or q.shape != (len(times), 4)
            or np.any(times[1:] <= times[:-1]) or max_gap_ns <= 0):
        raise ValueError("IMU gravity requires increasing int64 times and Nx4 xyzw quaternions")
    result = np.full((len(frames), 3), np.nan)
    if not len(times):
        return result
    upper = np.searchsorted(times, frames)
    hi = np.minimum(upper, len(times) - 1)
    exact = (upper < len(times)) & (times[hi] == frames)
    lo = np.where(exact, hi, np.maximum(hi - 1, 0))
    norms = np.linalg.norm(q, axis=1)
    valid_q = np.isfinite(q).all(axis=1) & (np.abs(norms - 1) < 0.01)
    gap = times[hi] - times[lo]
    valid = ((frames >= times[0]) & (frames <= times[-1])
             & (gap <= max_gap_ns) & valid_q[lo] & valid_q[hi])
    if not valid.any():
        return result
    left = Rotation.from_quat(q[lo[valid]])
    right = Rotation.from_quat(q[hi[valid]])
    fraction = (frames[valid] - times[lo[valid]]) / np.maximum(gap[valid], 1)
    relative = left.inv() * right
    interpolated = left * Rotation.from_rotvec(relative.as_rotvec() * fraction[:, None])
    gravity_imu = interpolated.inv().apply([0., 0., -1.])
    transform = scene_to_imu() if T_imu_camera is None else validate_transform(T_imu_camera)
    result[valid] = gravity_imu @ transform[:3, :3]
    return result
