"""Replay exported gaze in the detection coordinate frame, with bounded GPU storage."""

import csv
from pathlib import Path

import numpy as np

from utils.gaze_target import TARGET_FIELDS
from utils.neon_gaze import GAZE_FIELDS


def _rows(path, fields, frame_by_time):
    if not path.exists():
        return {}
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(fields) <= set(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path.name}")
        for row in reader:
            # Never parse epoch nanoseconds through float64.
            ts, frame_id = int(row["time_ns"]), int(row["frame_id"])
            if ts in result or frame_by_time.get(ts) != frame_id:
                raise ValueError(f"Duplicate or mismatched frame/time in {path.name}")
            result[ts] = row
    return result


def _vector(row, prefix):
    values = np.array([float(row[f"{prefix}_{axis}"]) for axis in "xyz"])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {prefix} in gaze CSV")
    return values


def load_gaze_frames(root, prefix, frame_ids, timestamps, poses, coordinate_frame,
                     world_pose_valid=None):
    """Load optional CSVs once; missing rows remain missing during seeks/replay.

    Saved world rays and both target coordinate representations must agree with
    the supplied camera poses. This also catches loading a different SLAM map.
    """
    root = Path(root)
    frames = dict(zip(map(int, timestamps), map(int, frame_ids)))
    gaze = _rows(root / f"{prefix}_gaze.csv", GAZE_FIELDS, frames)
    targets = _rows(root / f"{prefix}_gaze_targets.csv", TARGET_FIELDS, frames)
    if set(targets) - set(gaze):
        raise ValueError("Gaze targets require matching gaze CSV rows")
    result = {}
    for index, (ts, pose) in enumerate(zip(map(int, timestamps), poses)):
        frame_coordinates = coordinate_frame
        if world_pose_valid is not None:
            frame_coordinates = (
                "world_z_up" if world_pose_valid[index] else "per_frame_assumed_upright"
            )
        row = gaze.get(ts)
        if row is None:
            continue
        state = dict(status=row["status"], origin=None, direction=None,
                     target=None, image_xy=None, label="", distance_m=None)
        result[ts] = state
        target = targets.get(ts)
        if target is not None:
            if target["coordinate_frame"] != frame_coordinates:
                raise ValueError("Gaze target coordinate_frame differs from detections")
            if any(target[key] != row[key] for key in ("gaze_time_ns", "gaze_delta_ns")):
                raise ValueError("Gaze target uses a different gaze sample")
            state["status"] = target["status"]
        if row["status"] not in ("camera_only", "world_ray"):
            if target is not None and target["status"] == "estimated":
                raise ValueError("Estimated target requires a valid gaze ray")
            continue
        direction_camera = _vector(row, "direction_camera")
        if not np.isclose(np.linalg.norm(direction_camera), 1., atol=1e-5):
            raise ValueError("Gaze direction must be a unit vector")
        origin = np.asarray(pose[:3, 3], dtype=np.float64)
        direction = pose[:3, :3] @ direction_camera
        if frame_coordinates == "world_z_up":
            if row["status"] != "world_ray" or not (
                np.allclose(_vector(row, "origin_world"), origin, atol=1e-4, rtol=0)
                and np.allclose(_vector(row, "direction_world"), direction, atol=1e-4, rtol=0)
            ):
                raise ValueError("Gaze world ray disagrees with --neon_geometry")
        elif row["status"] != "camera_only":
            raise ValueError("World gaze cannot be replayed as a local camera frame")
        state.update(origin=origin, direction=direction / np.linalg.norm(direction))
        if row["in_image"] not in ("True", "False"):
            raise ValueError("Invalid in_image flag in gaze CSV")
        if row["in_image"] == "True":
            xy = np.array([float(row["image_x"]), float(row["image_y"])])
            if not np.isfinite(xy).all():
                raise ValueError("Non-finite gaze image position")
            state["image_xy"] = xy
        if target is None or target["status"] != "estimated":
            continue
        if row["in_image"] != "True" or target["method"] != "predicted_obb_surface":
            raise ValueError("Estimated target requires in-image gaze and OBB surface method")
        point = _vector(target, "position_frame")
        camera_point = _vector(target, "position_camera")
        distance = float(target["distance_m"])
        if not np.isfinite(distance) or distance <= 0 or not (
            np.allclose(point, origin + pose[:3, :3] @ camera_point, atol=1e-4, rtol=1e-5)
            and np.allclose(point, origin + state["direction"] * distance,
                            atol=1e-4, rtol=1e-5)
        ):
            raise ValueError("Gaze target position disagrees with its ray / camera pose")
        state.update(target=point, label=target["label"], distance_m=distance)
    return result


class GazeOverlay:
    """One ray and one target marker; GPU objects are reused across every frame.

    Uses the viewer's existing shaders and right-hand 3D viewport. A ray without
    an estimated hit has a display-only length, never a target at its endpoint.
    """

    def __init__(self, viewer):
        self.ctx = viewer.ctx
        self.line_prog = viewer.line_prog
        self.point_prog = viewer.point_prog
        self.line_vbo = self.ctx.buffer(reserve=10 * 4)
        self.line_vao = self.ctx.vertex_array(self.line_prog, [
            (viewer.quad_vbo, "2f", "in_quad_pos"),
            (self.line_vbo, "3f 3f 3f 1f /i", "start_pos", "end_pos", "line_color", "line_prob"),
        ])
        self.point_vbo = self.ctx.buffer(reserve=6 * 4)
        self.point_vao = self.ctx.vertex_array(self.point_prog, [
            (self.point_vbo, "3f 3f", "in_position", "in_color"),
        ])
        self.has_ray = self.has_target = False

    def update(self, state, ray_length=3.):
        self.has_ray = state is not None and state["direction"] is not None
        self.has_target = self.has_ray and state["target"] is not None
        if not self.has_ray:
            return
        endpoint = (state["target"] if self.has_target else
                    state["origin"] + ray_length * state["direction"])
        self.line_vbo.write(np.r_[state["origin"], endpoint, [1., .45, 0., 1.]].astype("f4"))
        if self.has_target:
            self.point_vbo.write(np.r_[state["target"], [1., 0., .8]].astype("f4"))

    def render(self, mvp, size, show_ray=True, show_target=True):
        self.ctx.disable(self.ctx.DEPTH_TEST)
        if show_ray and self.has_ray:
            self.line_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.line_prog["viewport_size"].value = tuple(size)
            self.line_prog["alpha"].value = 1.
            self.line_prog["prob_threshold"].value = 0.
            self.line_prog["line_width"].value = 4.
            self.line_vao.render(mode=self.ctx.TRIANGLES, instances=1)
        if show_target and self.has_target:
            self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
            self.point_prog["mvp"].write(mvp.astype("f4").tobytes())
            self.point_prog["point_size"].value = 14.
            self.point_prog["alpha"].value = 1.
            self.point_vao.render(mode=self.ctx.POINTS, vertices=1)

    def release(self):
        for resource in (self.line_vao, self.point_vao, self.line_vbo, self.point_vbo):
            resource.release()
