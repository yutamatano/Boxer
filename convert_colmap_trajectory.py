#!/usr/bin/env python3
"""Convert saved COLMAP camera-to-world poses into explicitly nonmetric Neon geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from utils.neon_vio import read_csv, validate_transform, write_json


def gravity_rotation(recording, prepared, ids, poses):
    """Candidate only: native orientation, nominal extrinsics and unverified sync."""
    from loaders.neon_loader import NeonLoader

    loader = NeonLoader(
        recording,
        frame_ids=np.asarray(ids, dtype=np.int64),
        point_cloud="off",
        imu_config_path=prepared / "manifest.json",
    )
    try:
        g = loader.local_gravity_camera.copy()
    finally:
        loader.close()
    good = np.isfinite(g).all(axis=1)
    vectors = np.einsum("nij,nj->ni", poses[good, :3, :3], g[good])
    report = dict(
        samples=int(good.sum()),
        status="unverified",
        rotation=np.eye(3).tolist(),
        reason="Native orientation/extrinsics/time calibration not independently verified",
    )
    if len(vectors) < 10:
        return np.eye(3), report
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    centre = np.median(vectors, axis=0)
    centre /= np.linalg.norm(centre)
    spread = float(
        np.percentile(np.degrees(np.arccos(np.clip(vectors @ centre, -1, 1))), 95)
    )
    report["direction_spread_p95_deg"] = spread
    if spread > 5:
        return np.eye(3), report
    rotation = Rotation.align_vectors([[0.0, 0.0, -1.0]], [centre])[0].as_matrix()
    report.update(status="gravity_aligned_unverified", rotation=rotation.tolist())
    return rotation, report


def convert(
    prepared,
    source,
    output,
    *,
    policy="largest",
    map_id=None,
    scale=None,
    align_gravity=False,
):
    prepared, source, output = map(
        lambda p: Path(p).expanduser().resolve(), (prepared, source, output)
    )
    if policy not in ("largest", "separated"):
        raise ValueError("Unknown COLMAP map policy")
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Scale must be finite and positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve geometry: output must be new or empty")
    manifest = json.loads((prepared / "manifest.json").read_text())
    selected = read_csv(prepared / "frames.csv")
    wanted = {int(r["frame_id"]): int(r["time_ns"]) for r in selected}
    catalog_path = Path(manifest.get("selection_source_prepared", prepared))
    catalog = {
        int(r["frame_id"]): int(r["time_ns"])
        for r in read_csv(catalog_path / "frames.csv")
    }
    if not wanted or len(wanted) != len(selected):
        raise ValueError("Prepared selection must contain unique source frames")
    if any(catalog.get(frame_id) != stamp for frame_id, stamp in wanted.items()):
        raise ValueError(
            "Prepared selection frame_id/timestamp differs from its source catalog"
        )
    directories = (
        [source]
        if (source / "visual_trajectory.npz").is_file()
        else sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())
    )
    models = []
    for folder in directories:
        if map_id is not None and folder.name != str(map_id):
            continue
        quality = (
            json.loads((folder / "quality.json").read_text())
            if (folder / "quality.json").exists()
            else {}
        )
        if (
            quality.get("quality_status") in ("fail", "rejected")
            or quality.get("accepted") is False
            or quality.get("rejected") is True
        ):
            raise ValueError(f"Rejected COLMAP model: {folder}")
        if quality.get("metric") is True:
            raise ValueError(
                "Expected the saved visual COLMAP model, not an ambiguous metric export"
            )
        path = folder / "visual_trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            ids = data["frame_ids"].copy()
            times = data["timestamps_ns"].copy()
            poses = np.asarray(data["camera_to_world"], dtype=np.float64)
            if "metric" not in data or bool(data["metric"].item()):
                raise ValueError("Visual model must explicitly declare metric=False")
        if (
            ids.dtype != np.dtype("int64")
            or times.dtype != np.dtype("int64")
            or ids.ndim != 1
            or times.shape != ids.shape
            or not len(ids)
            or np.any(np.diff(ids) <= 0)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("COLMAP needs increasing int64 frame IDs and timestamps")
        if poses.shape == (len(ids), 3, 4):
            expanded = np.tile(np.eye(4), (len(ids), 1, 1))
            expanded[:, :3] = poses
            poses = expanded
        if poses.shape != (len(ids), 4, 4):
            raise ValueError("COLMAP camera_to_world must be Nx3x4 or Nx4x4")
        for pose in poses:
            validate_transform(pose)
        if any(catalog.get(int(i)) != int(t) for i, t in zip(ids, times)):
            raise ValueError("COLMAP frame_id/timestamp differs from prepared source")
        csv_path = folder / "visual_camera_to_world.csv"
        if csv_path.exists():
            rows = read_csv(csv_path)
            if [int(r["frame_id"]) for r in rows] != ids.tolist():
                raise ValueError("COLMAP CSV and NPZ frame identity differs")
            q = np.array(
                [[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows]
            )
            xyz = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows])
            if (
                not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4)
                or not np.allclose(
                    Rotation.from_quat(q).as_matrix(), poses[:, :3, :3], atol=1e-5
                )
                or not np.allclose(xyz, poses[:, :3, 3], atol=1e-5)
                or [int(r["time_ns"]) for r in rows] != times.tolist()
            ):
                raise ValueError("COLMAP CSV quaternion/pose/time differs from NPZ")
        mask = np.array([int(i) in wanted for i in ids])
        ids, times, poses = ids[mask], times[mask], poses[mask]
        if len(ids):
            models.append(
                dict(
                    id=folder.name,
                    ids=ids,
                    times=times,
                    poses=poses,
                    path=path,
                    quality=quality,
                )
            )
    models.sort(
        key=lambda m: (-len(m["ids"]), int(m["id"]) if m["id"].isdigit() else m["id"])
    )
    if not models:
        raise ValueError("No eligible COLMAP poses overlap the selected input")
    if policy == "largest":
        models = models[:1]
    output.mkdir(parents=True, exist_ok=True)
    assigned = set()
    segments = []
    maps = []
    input_count = len(wanted) + len(manifest.get("excluded_without_imu", []))
    for world, model in enumerate(models):
        mask = np.array([int(i) not in assigned for i in model["ids"]])
        ids, times, poses = (
            model["ids"][mask],
            model["times"][mask],
            model["poses"][mask].copy(),
        )
        if not len(ids):
            continue
        assigned.update(map(int, ids))
        rotation = np.eye(3)
        gravity = dict(status="unverified", reason="No gravity rotation applied")
        if align_gravity:
            rotation, gravity = gravity_rotation(
                manifest["recording"], prepared, ids, poses
            )
        poses[:, :3, :3] = rotation @ poses[:, :3, :3]
        poses[:, :3, 3] = (poses[:, :3, 3] @ rotation.T) * (
            scale if scale is not None else 1.0
        )
        meta = dict(
            pose_backend="colmap",
            pose_frame="scene_camera",
            source_map_id=model["id"],
            world_id=world,
            metric=False,
            pose_units="arbitrary_visual" if scale is None else "scaled_unvalidated",
            scale_status="unknown" if scale is None else "supplied_unverified",
            scale_m_per_unit=scale,
            world_up_status=gravity["status"],
            gravity=gravity,
            world_rotation=rotation.tolist(),
            origin=[0.0, 0.0, 0.0],
            quality_status="not_evaluated",
            world_inference_allowed=False,
            source_model=str(model["path"]),
            source_sha256=hashlib.sha256(model["path"].read_bytes()).hexdigest(),
            pose_convention="camera_to_world; no inverse applied",
            duplicate_policy="largest selected frame count first, then lowest model ID",
            interpolation=False,
        )
        file = output / f"geometry_map_{world:03d}.npz"
        np.savez_compressed(
            file,
            timestamps_ns=times,
            T_world_camera=poses,
            source_frame_ids=ids,
            metric=np.array(False),
            geometry_metadata=json.dumps(meta),
        )
        maps.append(
            dict(
                world_id=world,
                source_map_id=model["id"],
                geometry=str(file),
                pose_frames=len(ids),
                metadata=meta,
            )
        )
        begin = 0
        for end in range(1, len(ids) + 1):
            if end == len(ids) or ids[end] != ids[end - 1] + 1:
                segments.append(
                    dict(
                        geometry=str(file),
                        file=file.name,
                        start_n=int(ids[begin]),
                        max_n=end - begin,
                        world_id=world,
                        source_map_id=model["id"],
                        session=world,
                        metric=False,
                    )
                )
                begin = end
    segments.sort(key=lambda s: s["start_n"])
    summary = dict(
        recording=manifest["recording"],
        prepared=str(prepared),
        input_frames=input_count,
        prepared_frames=len(selected),
        pose_backend="colmap",
        pose_units="arbitrary_visual" if scale is None else "scaled_unvalidated",
        metric=False,
        world_up_status="unverified",
        model_policy=policy,
        nonmetric_preview=True,
        sessions=[],
        segments=segments,
        maps=maps,
        visual_pose_frames=len(assigned),
        metric_pose_frames=0,
        usable_frames=0,
        coverage=0.0,
        visual_coverage=len(assigned) / input_count,
        imu_offset_ms_applied=manifest.get("imu_offset_ms_applied", 0.0),
        imu_noise_source=manifest.get("imu_noise_source", "not used by visual SfM"),
        accuracy="not independently validated; local metric predictions are not placed in visual map",
    )
    write_json(output / "report.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--map-policy", choices=["largest", "separated"], default="largest")
    p.add_argument("--map-id", type=int)
    p.add_argument("--scale-m-per-unit", type=float)
    p.add_argument("--align-gravity", action="store_true")
    a = p.parse_args()
    try:
        print(
            json.dumps(
                convert(
                    a.prepared,
                    a.source,
                    a.output,
                    policy=a.map_policy,
                    map_id=a.map_id,
                    scale=a.scale_m_per_unit,
                    align_gravity=a.align_gravity,
                ),
                indent=2,
            )
        )
    except (ValueError, OSError, KeyError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
