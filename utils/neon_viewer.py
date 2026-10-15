"""Exact-time Neon replay context, independent of the desktop/OpenGL backend."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from loaders.neon_loader import NeonLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


def resolve_neon_recording(path):
    """Accept a recording or a parent containing exactly one native recording."""
    root = Path(path).expanduser()
    if NeonLoader.is_recording(root):
        return root
    if not root.is_dir():
        return None
    recordings = sorted(p for p in root.iterdir() if NeonLoader.is_recording(p))
    if len(recordings) > 1:
        raise ValueError("Multiple Neon recordings found; pass one recording directory")
    return recordings[0] if recordings else None


def build_neon_context(recording, geometry, log_dir, write_name, detections,
                       point_cloud="auto"):
    """Validate replay inputs and include processed frames with zero detections.

    Geometry must belong to the same reconstruction as the detection CSV.
    No timestamp interpolation, pose substitution or independent-map merging.
    """
    root = Path(log_dir)
    metadata = json.loads((root / f"{write_name}_input.json").read_text())
    world = metadata.get("coordinate_frame") == "world_z_up"
    local = metadata.get("coordinate_frame") == "per_frame_assumed_upright"
    if metadata.get("dataset") != "neon" or not (
        (world and metadata.get("has_world_poses") is True)
        or (local and metadata.get("has_world_poses") is False)
    ):
        raise ValueError("Inconsistent Neon coordinate_frame / has_world_poses metadata")
    if world and not geometry:
        raise ValueError("World replay requires --neon_geometry from the Boxer run")
    if local and geometry:
        raise ValueError("Local detections cannot be combined with world geometry")
    if "image_size" not in metadata:
        raise ValueError("Neon input metadata must include image_size")
    with (root / f"{write_name}_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    frame_ids = np.array([int(row["frame_id"]) for row in rows], dtype=np.int64)
    timestamps = np.array([int(row["time_ns"]) for row in rows], dtype=np.int64)
    loader = NeonLoader(
        recording,
        frame_ids=frame_ids,
        resize=metadata["image_size"],
        geometry_path=geometry,
        point_cloud=point_cloud,
    )
    try:
        if not np.array_equal(loader.timestamps_ns[frame_ids], timestamps):
            raise ValueError("Frame CSV timestamps must equal the source .time exactly")
        if set(detections) - set(map(int, timestamps)):
            raise ValueError("3D detections contain timestamps outside the frame CSV")
        first = loader.load(0)
        poses = (
            loader.poses[loader.geometry_indices] if world
            else np.repeat(first["T_world_rig0"].matrix.numpy()[None], len(frame_ids), axis=0)
        )
        context = {
            "source": "neon",
            "loader": loader,
            "rgb_num_frames": len(frame_ids),
            "rgb_timestamps": timestamps,
            "source_frame_ids": frame_ids,
            "coordinate_frame": metadata["coordinate_frame"],
            "frame_local": local,
            "rgb_images": None,
            "is_nebula": True,
            "traj": [PoseTW.from_matrix(torch.from_numpy(p)) for p in poses],
            "pose_ts": timestamps,
            "calibs": [first["cam0"]] * len(frame_ids),
            "calib_ts": timestamps,
            "time_to_uids_slaml": {},
            "time_to_uids_slamr": {},
            "uid_to_p3": {},
        }
        # Reuse the viewer's point/observation representation. A static map has
        # no per-frame observation evidence: its observation dictionaries stay empty.
        if loader.points is not None:
            if loader.points.ndim == 2:
                points = loader.points[np.isfinite(loader.points).all(axis=1)]
            else:
                chunks, offset = [], 0
                for ts, gi in zip(timestamps, loader.geometry_indices):
                    chunk = loader.points[gi]
                    chunk = chunk[np.isfinite(chunk).all(axis=1)]
                    context["time_to_uids_slaml"][int(ts)] = list(
                        range(offset, offset + len(chunk))
                    )
                    chunks.append(chunk)
                    offset += len(chunk)
                points = np.concatenate(chunks)
            context["uid_to_p3"] = {i: (*p, 0.0, 0.0) for i, p in enumerate(points)}
        timeline = {
            int(ts): detections.get(int(ts), ObbTW(torch.zeros(0, 165)))
            for ts in timestamps
        }
        from utils.gaze_viewer import load_gaze_frames

        context["gaze_frames"] = load_gaze_frames(
            root, write_name, frame_ids, timestamps, poses, metadata["coordinate_frame"]
        )
        return context, timeline
    except Exception:
        loader.close()
        raise
