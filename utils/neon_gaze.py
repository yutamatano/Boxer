"""Frame-sampled Neon gaze, with camera-centred world-ray approximation."""

import csv

import cv2
import numpy as np


GAZE_FIELDS = [
    "frame_id", "time_ns", "gaze_time_ns", "gaze_delta_ns", "status",
    "raw_x", "raw_y", "image_x", "image_y", "in_image",
    "direction_camera_x", "direction_camera_y", "direction_camera_z",
    "origin_world_x", "origin_world_y", "origin_world_z",
    "direction_world_x", "direction_world_y", "direction_world_z",
]


class NeonGaze:
    def __init__(self, timestamps_ns, points, max_delta_ms=10.):
        self.times = np.asarray(timestamps_ns)
        self.points = np.asarray(points, dtype=np.float64)
        if (
            self.times.ndim != 1 or self.times.dtype != np.dtype("int64")
            or not len(self.times) or np.any(np.diff(self.times) <= 0)
            or self.points.shape != (len(self.times), 2)
        ):
            raise ValueError("Gaze requires increasing int64 times and paired (N,2) pixels")
        if not np.isfinite(max_delta_ms) or not 0 <= max_delta_ms <= 1000:
            raise ValueError("gaze_max_delta_ms must be between 0 and 1000 ms")
        self.max_delta_ns = round(max_delta_ms * 1_000_000)

    def sample(self, frame_id, time_ns, K, distortion, K_out, raw_size, out_size,
               T_world_camera=None):
        row = dict.fromkeys(GAZE_FIELDS, "")
        row.update(frame_id=frame_id, time_ns=time_ns, status="missing_gaze", in_image=False)
        right = int(np.searchsorted(self.times, time_ns))
        candidates = [i for i in (right - 1, right) if 0 <= i < len(self.times)]
        index = min(candidates, key=lambda i: abs(int(self.times[i]) - time_ns))
        delta = int(self.times[index]) - time_ns
        if abs(delta) > self.max_delta_ns:
            return row
        row.update(gaze_time_ns=int(self.times[index]), gaze_delta_ns=delta)
        pixel = self.points[index]
        if not np.isfinite(pixel).all():
            row["status"] = "invalid_gaze"
            return row
        row.update(raw_x=float(pixel[0]), raw_y=float(pixel[1]))
        if not (0 <= pixel[0] < raw_size[0] and 0 <= pixel[1] < raw_size[1]):
            row["status"] = "outside_scene"
            return row
        xy = cv2.undistortPoints(pixel.reshape(1, 1, 2), K, distortion).reshape(2)
        ray = np.r_[xy, 1.]
        norm = np.linalg.norm(ray)
        image = K_out @ ray
        if not np.isfinite(image).all() or not np.isfinite(norm) or norm == 0:
            row["status"] = "invalid_gaze"
            return row
        image = image[:2] / image[2]
        ray /= norm
        row.update(
            image_x=float(image[0]), image_y=float(image[1]),
            in_image=bool(0 <= image[0] < out_size[0] and 0 <= image[1] < out_size[1]),
            status="camera_only",
        )
        for axis, value in zip("xyz", ray):
            row[f"direction_camera_{axis}"] = float(value)
        if T_world_camera is not None:
            world_ray = T_world_camera[:3, :3] @ ray
            for axis, origin, direction in zip("xyz", T_world_camera[:3, 3], world_ray):
                row[f"origin_world_{axis}"] = float(origin)
                row[f"direction_world_{axis}"] = float(direction)
            row["status"] = "world_ray"
        return row


class GazeCsvWriter:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=GAZE_FIELDS)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def draw_gaze(image, row):
    """Draw on a visualization copy only; never modify a detector input."""
    if row is not None and row["in_image"]:
        pixel = (round(row["image_x"]), round(row["image_y"]))
        cv2.circle(image, pixel, 18, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.circle(image, pixel, 18, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    return image
