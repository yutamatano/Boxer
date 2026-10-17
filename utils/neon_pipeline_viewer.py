"""Assemble disjoint Neon inference outputs into one exact-time replay."""

import csv
import json
import os
from pathlib import Path

from utils.neon_vio import read_csv, write_json


def selected_frame_ids(prepared):
    """Include requested images without bracketing IMU, which cannot enter VIO."""
    prepared = Path(prepared)
    ids = [int(row["frame_id"]) for row in read_csv(prepared / "frames.csv")]
    if not ids or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("Prepared frame IDs must be nonempty and increasing")
    manifest = prepared / "manifest.json"
    excluded = (
        json.loads(manifest.read_text()).get("excluded_without_imu", [])
        if manifest.exists()
        else []
    )
    if set(ids) & set(excluded):
        raise ValueError("IMU-excluded frames overlap prepared frames")
    return sorted(set(ids) | set(map(int, excluded)))


def assemble_replay(summary, out):
    """Keep each world's identity while joining detections and empty frames.

    Geometry and original interval outputs remain the source of truth. Nothing
    estimates a transform between independent SLAM maps.
    """
    out = Path(out).resolve()
    root = out / "viewer" / Path(summary["recording"]).name
    parts, frames, by_frame = [], [], {}
    sources = [(i, part) for i, part in enumerate(summary["segments"])]
    sources += [(-1, part) for part in summary["frame_only_intervals"]]
    common = None
    for world_id, part in sources:
        path = Path(part["results"]).resolve()
        metadata = json.loads((path / "boxer_input.json").read_text())
        if metadata.get("dataset") != "neon" or "pipeline_parts" in metadata:
            raise ValueError("Replay parts must be individual Neon inference outputs")
        expected = "world_z_up" if world_id >= 0 else "per_frame_assumed_upright"
        if metadata.get("coordinate_frame") != expected:
            raise ValueError("Pipeline part has an inconsistent coordinate frame")
        calibration = {
            key: metadata[key]
            for key in (
                "image_size",
                "scene_camera_matrix",
                "scene_distortion_coefficients",
            )
        }
        if common is not None and common != calibration:
            raise ValueError("Pipeline parts use different image calibration/size")
        common = calibration
        rows = read_csv(path / "boxer_frames.csv")
        if [int(row["frame_id"]) for row in rows] != list(
            range(part["start_n"], part["start_n"] + part["max_n"])
        ):
            raise ValueError("Pipeline part does not cover its exact source interval")
        times = {}
        for row in rows:
            fid, ts = int(row["frame_id"]), int(row["time_ns"])
            if fid in by_frame or ts in times:
                raise ValueError("Pipeline replay parts overlap or repeat timestamps")
            flag = "1" if world_id >= 0 else "0"
            if row.get("has_world_pose", flag) != flag:
                raise ValueError("Pipeline part has inconsistent pose flags")
            record = dict(
                frame_id=fid,
                time_ns=ts,
                time_aux_ns=int(row.get("time_aux_ns", -1)),
                has_world_pose=int(flag),
                world_segment_id=world_id,
            )
            by_frame[fid] = record
            frames.append(record)
            times[ts] = fid
        parts.append(
            {
                "path": path,
                "times": times,
                "metadata": metadata,
                "world_segment_id": world_id,
                "geometry": part.get("geometry"),
            }
        )
    frames.sort(key=lambda row: row["frame_id"])
    if [row["frame_id"] for row in frames] != selected_frame_ids(summary["prepared"]):
        raise ValueError("Pipeline outputs do not cover every selected source frame")
    if any(b["time_ns"] <= a["time_ns"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Merged source timestamps must be strictly increasing")
    parts.sort(key=lambda part: min(part["times"]))
    root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "boxer_3dbbs.csv",
        "owl_2dbbs.csv",
        "boxer_gaze.csv",
        "boxer_gaze_targets.csv",
    ):
        available = [part for part in parts if (part["path"] / filename).is_file()]
        if not available and filename == "boxer_3dbbs.csv":
            raise ValueError("Pipeline parts have no 3D detection CSV")
        if not available:
            continue
        if filename in ("boxer_3dbbs.csv", "boxer_gaze.csv", "boxer_gaze_targets.csv"):
            if len(available) != len(parts):
                raise ValueError(f"Incomplete pipeline CSV outputs: {filename}")
        fields = None
        with (root / filename).open("w", newline="") as stream:
            for part in available:
                with (part["path"] / filename).open(newline="") as source:
                    reader = csv.DictReader(source)
                    if fields is None:
                        fields = reader.fieldnames
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                    elif reader.fieldnames != fields:
                        raise ValueError(f"Inconsistent CSV columns: {filename}")
                    for row in reader:
                        ts = int(row["time_ns"])
                        if ts not in part["times"] or (
                            "frame_id" in row
                            and int(row["frame_id"]) != part["times"][ts]
                        ):
                            raise ValueError(
                                f"CSV row has a mismatched source frame: {filename}"
                            )
                        writer.writerow(row)
    with (root / "boxer_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    metadata = dict(parts[0]["metadata"])
    valid = sum(row["has_world_pose"] for row in frames)
    coordinate_frame = "world_z_up" if valid else "per_frame_assumed_upright"
    if 0 < valid < len(frames):
        coordinate_frame = "mixed_world_and_per_frame"
    metadata.update(
        schema_version=3,
        coordinate_frame=coordinate_frame,
        has_world_poses=bool(valid),
        geometry_path=None,
        world_pose_frames=valid,
        independent_world_segments=len(summary["segments"]),
        pipeline_parts=[
            {
                "results": os.path.relpath(part["path"], root),
                "world_segment_id": part["world_segment_id"],
                "geometry": (
                    os.path.relpath(Path(part["geometry"]).resolve(), root)
                    if part["geometry"]
                    else None
                ),
            }
            for part in parts
        ],
    )
    metadata["point_cloud"] = "per_segment"
    if metadata.get("gaze"):
        metadata["gaze"]["world_frame"] = "world_z_up" if valid else None
    if metadata.get("gaze_targets"):
        metadata["gaze_targets"]["coordinate_frame"] = coordinate_frame
    write_json(root / "boxer_input.json", metadata)
    summary["viewer_results"] = str(root)
    summary["total_estimated_frames"] = len(frames)
    write_json(out / "pipeline.json", summary)
    return root.parent
