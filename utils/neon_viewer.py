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
    if "pipeline_parts" in metadata:
        if geometry:
            raise ValueError("Pipeline replay uses its saved per-segment geometry")
        return _build_pipeline_context(
            recording, root, write_name, metadata, detections, point_cloud
        )
    world = metadata.get("coordinate_frame") == "world_z_up"
    local = metadata.get("coordinate_frame") == "per_frame_assumed_upright"
    mixed = metadata.get("coordinate_frame") == "mixed_world_and_per_frame"
    if metadata.get("dataset") != "neon" or not (
        ((world or mixed) and metadata.get("has_world_poses") is True)
        or (local and metadata.get("has_world_poses") is False)
    ):
        raise ValueError("Inconsistent Neon coordinate_frame / has_world_poses metadata")
    geometry = geometry or metadata.get("geometry_path")
    if (world or mixed) and not geometry:
        raise ValueError("World replay requires --neon_geometry from the Boxer run")
    if local and geometry and metadata.get("schema_version", 1) < 2:
        raise ValueError("Local detections cannot be combined with world geometry")
    if "image_size" not in metadata:
        raise ValueError("Neon input metadata must include image_size")
    with (root / f"{write_name}_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    frame_ids = np.array([int(row["frame_id"]) for row in rows], dtype=np.int64)
    timestamps = np.array([int(row["time_ns"]) for row in rows], dtype=np.int64)
    if mixed and any("has_world_pose" not in row for row in rows):
        raise ValueError("Mixed replay requires per-frame has_world_pose flags")
    flags = [row.get("has_world_pose", "1" if world else "0") for row in rows]
    if any(flag not in ("0", "1") for flag in flags):
        raise ValueError("has_world_pose must be 0 or 1")
    world_pose_valid = np.array([flag == "1" for flag in flags])
    loader = NeonLoader(
        recording,
        frame_ids=frame_ids,
        resize=metadata["image_size"],
        geometry_path=geometry,
        point_cloud=point_cloud,
        nonmetric_preview=metadata.get("nonmetric_preview", False),
    )
    try:
        if not np.array_equal(loader.timestamps_ns[frame_ids], timestamps):
            raise ValueError("Frame CSV timestamps must equal the source .time exactly")
        if (
            not np.array_equal(loader.world_pose_valid, world_pose_valid)
            or loader.coordinate_frame != metadata["coordinate_frame"]
        ):
            raise ValueError("Replay geometry does not match saved per-frame pose status")
        if set(detections) - set(map(int, timestamps)):
            raise ValueError("3D detections contain timestamps outside the frame CSV")
        first = loader.load(0)
        poses = np.stack([
            loader.frame_pose(i).matrix.numpy() for i in range(len(frame_ids))
        ])
        context = {
            "source": "neon",
            "loader": loader,
            "rgb_num_frames": len(frame_ids),
            "rgb_timestamps": timestamps,
            "source_frame_ids": frame_ids,
            "coordinate_frame": metadata["coordinate_frame"],
            "frame_local": local,
            "world_pose_valid": world_pose_valid,
            "world_pose_by_time": dict(
                zip(map(int, timestamps), map(bool, world_pose_valid))
            ),
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
        if metadata.get("nonmetric_preview"):
            preview_valid = loader.visual_pose_valid
            context.update(nonmetric_preview=True, pose_metadata=loader.pose_metadata,
                preview_traj=[PoseTW.from_matrix(torch.from_numpy(loader.poses[gi])) if gi >= 0 else loader.local_camera_pose() for gi in loader.geometry_indices],
                preview_pose_by_time=dict(zip(map(int, timestamps), map(bool, preview_valid))),
                preview_segment_by_time={int(ts): 0 if ok else -1 for ts, ok in zip(timestamps, preview_valid)},
                active_world_segment=0, world_contexts={0: {"uid_to_p3": {}, "time_to_uids_slaml": {}, "time_to_uids_slamr": {}}})
        # Reuse the viewer's point/observation representation. A static map has
        # no per-frame observation evidence: its observation dictionaries stay empty.
        if loader.points is not None and loader.has_world_poses:
            if loader.points.ndim == 2:
                points = loader.points[np.isfinite(loader.points).all(axis=1)]
            else:
                chunks, offset = [], 0
                for ts, gi in zip(timestamps, loader.geometry_indices):
                    if gi < 0:
                        continue
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
            root, write_name, frame_ids, timestamps, poses, metadata["coordinate_frame"],
            world_pose_valid=world_pose_valid,
        )
        return context, timeline
    except Exception:
        loader.close()
        raise


def _build_pipeline_context(recording, root, prefix, metadata, detections, point_cloud):
    """Join exact-time timelines, retaining independent maps for world switching."""
    if metadata.get("dataset") != "neon":
        raise ValueError("Pipeline replay must contain Neon metadata")
    with (root / f"{prefix}_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = np.array([int(row["frame_id"]) for row in rows], dtype=np.int64)
    times = np.array([int(row["time_ns"]) for row in rows], dtype=np.int64)
    preview = bool(metadata.get("nonmetric_preview"))
    frame_data, worlds, preview_data, world_geometry = {}, {}, {}, {}
    asset_keys = ("uid_to_p3", "time_to_uids_slaml", "time_to_uids_slamr")
    for part in metadata["pipeline_parts"]:
        directory = (root / part["results"]).resolve()
        leaf_metadata = json.loads((directory / f"{prefix}_input.json").read_text())
        if "pipeline_parts" in leaf_metadata:
            raise ValueError("Nested pipeline replay is not supported")
        world_id = int(part["world_segment_id"])
        expected = "world_z_up" if world_id >= 0 and not preview else "per_frame_assumed_upright"
        if leaf_metadata.get("coordinate_frame") != expected or any(
            leaf_metadata.get(key) != metadata.get(key) for key in (
                "image_size", "scene_camera_matrix", "scene_distortion_coefficients"
            )
        ):
            raise ValueError("Pipeline part has inconsistent coordinates/calibration")
        # Validate every leaf's geometry, source frame IDs, and saved gaze before
        # closing it. Only the all-frame RGB loader remains open for playback.
        geometry = (root / part["geometry"]).resolve() if part.get("geometry") else None
        leaf, timeline = build_neon_context(
            recording, geometry, directory, prefix, {},
            point_cloud=point_cloud if world_id >= 0 else "off",
        )
        try:
            if world_id >= 0:
                if world_id in worlds and (not preview or world_geometry[world_id] != str(geometry)):
                    raise ValueError("Duplicate independent world segment ID")
                world_geometry[world_id] = str(geometry)
                worlds[world_id] = {key: leaf[key] for key in asset_keys}
            for i, ts in enumerate(timeline):
                if ts in frame_data:
                    raise ValueError("Pipeline replay parts overlap")
                if preview:
                    preview_data[ts] = leaf["preview_traj"][i] if leaf.get("preview_pose_by_time", {}).get(ts, False) else None
                frame_data[ts] = (int(leaf["source_frame_ids"][i]), leaf["traj"][i],
                                  leaf["calibs"][i], world_id)
        finally:
            leaf["loader"].close()
    if set(map(int, times)) != set(frame_data) or set(detections) - set(frame_data):
        raise ValueError("Pipeline replay does not cover its saved source timeline")
    for row in rows:
        fid, ts = int(row["frame_id"]), int(row["time_ns"])
        if (frame_data[ts][0] != fid
                or frame_data[ts][3] != int(row["world_segment_id"])
                or row["has_world_pose"] != ("1" if frame_data[ts][3] >= 0 and not preview else "0")):
            raise ValueError("Pipeline replay has inconsistent frame/world identity")
    valid = np.array([frame_data[int(ts)][3] >= 0 and not preview for ts in times])
    expected = "world_z_up" if valid.any() else "per_frame_assumed_upright"
    if valid.any() and not valid.all():
        expected = "mixed_world_and_per_frame"
    if (metadata.get("coordinate_frame") != expected
            or metadata.get("has_world_poses") != bool(valid.any())
            or metadata.get("independent_world_segments") != len(worlds)):
        raise ValueError("Pipeline replay coordinate metadata is inconsistent")
    loader = NeonLoader(recording, frame_ids=ids, resize=metadata["image_size"],
                        point_cloud="off")
    try:
        if not np.array_equal(loader.timestamps_ns[ids], times):
            raise ValueError("Pipeline timestamps differ from the source .time")
        poses = [frame_data[int(ts)][1] for ts in times]
        active_world = next((frame_data[int(ts)][3] for ts in times
                             if frame_data[int(ts)][3] >= 0), None)
        context = {
            "source": "neon", "loader": loader, "rgb_num_frames": len(ids),
            "rgb_timestamps": times, "source_frame_ids": ids, "rgb_images": None,
            "coordinate_frame": expected, "frame_local": not valid.any(),
            "world_pose_valid": valid,
            "world_pose_by_time": dict(zip(map(int, times), map(bool, valid))),
            "world_segment_by_time": {int(ts): frame_data[int(ts)][3] for ts in times},
            "world_contexts": worlds, "active_world_segment": active_world,
            "is_nebula": True, "traj": poses, "pose_ts": times,
            "calibs": [frame_data[int(ts)][2] for ts in times], "calib_ts": times,
            **{key: {} for key in asset_keys},
        }
        if active_world is not None:
            context.update(worlds[active_world])
        if preview:
            if any((preview_data[int(ts)] is not None) != bool(int(row.get("visual_pose_available", "0"))) for ts, row in zip(times, rows)):
                raise ValueError("Preview visual pose flags differ from exact geometry")
            context.update(nonmetric_preview=True, pose_metadata=metadata["pose_geometry_metadata"],
                preview_traj=[preview_data[int(ts)] if preview_data[int(ts)] is not None else poses[i] for i, ts in enumerate(times)],
                preview_pose_by_time={int(ts): preview_data[int(ts)] is not None for ts in times},
                preview_segment_by_time={int(ts): frame_data[int(ts)][3] for ts in times})
        from utils.gaze_viewer import load_gaze_frames

        context["gaze_frames"] = load_gaze_frames(
            root, prefix, ids, times, [pose.matrix.numpy() for pose in poses],
            expected, world_pose_valid=valid,
        )
        return context, {int(ts): detections.get(int(ts), ObbTW(torch.zeros(0, 165)))
                         for ts in times}
    except Exception:
        loader.close()
        raise
