"""Read Tag Aligner's aligned camera poses (not raw RIM poses).

Format: https://github.com/pupil-labs/tag-aligner
Camera axes remain OpenCV; only the world basis changes to Boxer Z-up.
"""

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


WORLD_FROM_TAG = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
POSE_FIELDS = [
    "start_timestamp", "end_timestamp", "translation_x", "translation_y",
    "translation_z", "rotation_x", "rotation_y", "rotation_z", "rotation_w",
]


def relative_seconds_to_ns(value, start_ns):
    """Add relative seconds without ever converting the epoch to float."""
    try:
        seconds = Decimal(value)
        if not seconds.is_finite() or seconds < 0:
            raise ValueError("Pose timestamps must be finite, nonnegative seconds")
        offset = int((seconds * 1_000_000_000).to_integral_value(ROUND_HALF_EVEN))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid relative timestamp: {value!r}") from exc
    result = start_ns + offset
    if not 0 <= result <= np.iinfo(np.int64).max:
        raise ValueError("Pose timestamp is outside int64 nanoseconds")
    return result


class AlignedPoses:
    def __init__(self, path, recording_start_ns, meters_per_unit):
        if type(recording_start_ns) is not int or recording_start_ns < 0:
            raise ValueError("info.json start_time must be integer epoch nanoseconds")
        if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be finite and positive")
        with Path(path).open(newline="") as file:
            reader = csv.DictReader(file)
            if not set(POSE_FIELDS) <= set(reader.fieldnames or []):
                raise ValueError("Expected Tag Aligner aligned_poses.csv columns")
            rows = list(reader)
        if not rows:
            raise ValueError("Aligned pose CSV is empty")
        self.starts = np.array([
            relative_seconds_to_ns(row["start_timestamp"], recording_start_ns)
            for row in rows
        ], dtype=np.int64)
        self.ends = np.array([
            relative_seconds_to_ns(row["end_timestamp"], recording_start_ns)
            for row in rows
        ], dtype=np.int64)
        if (
            np.any(self.ends <= self.starts)
            or np.any(np.diff(self.starts) <= 0)
            or np.any(self.starts[1:] < self.ends[:-1])
        ):
            raise ValueError("Pose intervals must be positive, sorted and nonoverlapping")
        values = np.array([[float(row[key]) for key in POSE_FIELDS[2:]] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError("Aligned poses must be finite")
        quaternions = values[:, 3:]
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-3):
            raise ValueError("Pose rotations must be unit quaternions in xyzw order")
        self.matrices = np.tile(np.eye(4), (len(rows), 1, 1))
        self.matrices[:, :3, :3] = WORLD_FROM_TAG @ Rotation.from_quat(quaternions).as_matrix()
        # aligned_poses.csv already includes Tag Aligner's scale/correction.
        # This factor only converts the surveyed reference's units to metres.
        self.matrices[:, :3, 3] = (values[:, :3] * meters_per_unit) @ WORLD_FROM_TAG.T

    def match(self, timestamps_ns):
        """Match [start, end) intervals; return -1 in gaps, never extrapolate."""
        times = np.asarray(timestamps_ns)
        if times.ndim != 1 or times.dtype != np.dtype("int64"):
            raise ValueError("Frame times must be an int64 vector")
        indices = np.searchsorted(self.starts, times, side="right") - 1
        valid = (indices >= 0) & (times < self.ends[np.maximum(indices, 0)])
        return np.where(valid, indices, -1)
