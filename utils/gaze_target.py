"""Estimate per-frame gaze targets on predicted OBB surfaces, without a mesh.

These are spatial estimates, not measured surface points or temporal fixations.
The boxes and camera pose must use the same coordinate frame.
"""

import csv

import cv2
import numpy as np


TARGET_FIELDS = [
    "frame_id", "time_ns", "gaze_time_ns", "gaze_delta_ns", "status",
    "method", "coordinate_frame", "point_cloud_used", "box_count",
    "intersection_count", "detection_index", "label", "box_confidence",
    "camera_inside_box", "distance_m",
    "position_camera_x", "position_camera_y", "position_camera_z",
    "position_frame_x", "position_frame_y", "position_frame_z",
]


def ray_obb_intersections(origin, direction, bounds, poses):
    """Return first forward surface distance per OBB; misses are +inf.

    bounds: (N,6), [xmin,xmax,ymin,ymax,zmin,zmax] in object coordinates.
    poses: (N,4,4), frame-from-object rigid transforms. Direction is normalized.
    A ray starting inside a box hits its exit surface. Invalid boxes are ignored.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("Ray origin and direction must have shape (3,)")
    norm = np.linalg.norm(direction)
    if not np.isfinite(origin).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Ray must be finite with a nonzero direction")
    if bounds.ndim != 2 or bounds.shape[1] != 6 or poses.shape != (len(bounds), 4, 4):
        raise ValueError("OBBs require (N,6) bounds and (N,4,4) poses")
    distances = np.full(len(bounds), np.inf)
    inside = np.zeros(len(bounds), dtype=bool)
    valid = np.isfinite(bounds).all(axis=1) & np.isfinite(poses).all(axis=(1, 2))
    valid &= (bounds[:, 1::2] > bounds[:, ::2]).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return distances, inside
    rotations = poses[indices, :3, :3]
    rigid = np.isclose(
        rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4
    ).all(axis=(1, 2))
    rigid &= np.isclose(np.linalg.det(rotations), 1, atol=1e-4)
    rigid &= np.isclose(poses[indices, 3], [0, 0, 0, 1], atol=1e-5).all(axis=1)
    indices = indices[rigid]
    if not len(indices):
        return distances, inside
    inverse_rotation = poses[indices, :3, :3].transpose(0, 2, 1)
    local_origin = np.einsum("nij,nj->ni", inverse_rotation, origin - poses[indices, :3, 3])
    local_direction = np.einsum("nij,j->ni", inverse_rotation, direction / norm)
    lower, upper = bounds[indices, ::2], bounds[indices, 1::2]
    parallel = np.abs(local_direction) < 1e-12
    outside = ((local_origin < lower) | (local_origin > upper)) & parallel
    safe_direction = np.where(parallel, 1., local_direction)
    first = (lower - local_origin) / safe_direction
    second = (upper - local_origin) / safe_direction
    enter = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
    leave = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
    hit = ~outside.any(axis=1) & (leave >= np.maximum(enter, 0.)) & (leave > 0.)
    distances[indices] = np.where(hit, np.where(enter > 0., enter, leave), np.inf)
    inside[indices] = ((local_origin >= lower) & (local_origin <= upper)).all(axis=1)
    return distances, inside


def estimate_gaze_target(gaze, obbs, T_frame_camera, coordinate_frame, point_cloud_used=False):
    """Use current detections only; a missing target never inherits an old hit."""
    row = dict.fromkeys(TARGET_FIELDS, "")
    row.update(
        {key: gaze[key] for key in ("frame_id", "time_ns", "gaze_time_ns", "gaze_delta_ns")}
    )
    row.update(
        status="no_gaze", method="predicted_obb_surface",
        coordinate_frame=coordinate_frame, point_cloud_used=bool(point_cloud_used),
        box_count=0 if obbs is None else len(obbs), intersection_count=0,
    )
    if gaze["status"] not in ("camera_only", "world_ray"):
        row["status"] = gaze["status"]
        return row
    if not gaze["in_image"]:
        row["status"] = "outside_image"
        return row
    if obbs is None or not len(obbs):
        row["status"] = "no_boxes"
        return row
    transform = np.asarray(T_frame_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_frame_camera must be a finite (4,4) transform")
    direction_camera = np.array([gaze[f"direction_camera_{axis}"] for axis in "xyz"])
    direction_frame = transform[:3, :3] @ direction_camera
    origin = transform[:3, 3]
    distances, inside = ray_obb_intersections(
        origin, direction_frame, obbs.bb3_object.detach().cpu().numpy(),
        obbs.T_world_object.matrix.detach().cpu().numpy(),
    )
    hits = np.flatnonzero(np.isfinite(distances))
    row["intersection_count"] = len(hits)
    if not len(hits):
        row["status"] = "no_intersection"
        return row
    index = int(hits[np.argmin(distances[hits])])
    distance = float(distances[index])
    position_frame = origin + distance * direction_frame / np.linalg.norm(direction_frame)
    position_camera = transform[:3, :3].T @ (position_frame - origin)
    row.update(
        status="estimated", detection_index=index, label=obbs[index].text_string(),
        box_confidence=float(obbs[index].prob.item()), camera_inside_box=bool(inside[index]),
        distance_m=distance,
    )
    for axis, camera, frame in zip("xyz", position_camera, position_frame):
        row[f"position_camera_{axis}"] = float(camera)
        row[f"position_frame_{axis}"] = float(frame)
    return row


class GazeTargetCsvWriter:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=TARGET_FIELDS)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def draw_gaze_target(image, gaze, target):
    if target is None:
        return image
    if target["status"] == "estimated":
        text = f"Gaze estimate: {target['label']} ~{target['distance_m']:.2f} m"
        pixel = (round(gaze["image_x"]), round(gaze["image_y"]))
        cv2.circle(image, pixel, 4, (0, 255, 0), -1, cv2.LINE_AA)
    else:
        text = f"Gaze estimate: {target['status']}"
    scale = min(image.shape[1] / 960., 1.)
    position = (10, max(15, round(100 * scale)))
    for color, thickness in [((0, 0, 0), 3), ((0, 255, 0), 1)]:
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.6 * scale), color, thickness, cv2.LINE_AA)
    return image
