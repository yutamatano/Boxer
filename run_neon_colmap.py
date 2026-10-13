#!/usr/bin/env python3
"""Experimental image-only Neon SfM with held-out image/observation evaluation."""

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from utils.neon_vio import check_times, recording_path, rectification, write_json


def split_frames(ids, holdout_every):
    ids = np.asarray(ids, dtype=np.int64)
    if holdout_every < 3 or len(ids) < 10:
        raise ValueError("Need >=10 selected images and holdout-every >=3")
    holdout = np.arange(len(ids)) % holdout_every == holdout_every - 1
    return ids[~holdout], ids[holdout]


def summarize_errors(errors):
    values = np.asarray(errors, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "median_px": float(np.median(finite)) if len(finite) else None,
        "p95_px": float(np.percentile(finite, 95)) if len(finite) else None,
        "rmse_px": float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
        "fraction_within_4px": float(np.mean(values <= 4)) if len(values) else None,
    }


def prepare_images(recording, output, start, max_n, stride, width, holdout_every):
    import pupil_labs.neon_recording as nr

    rec = nr.open(recording)
    try:
        scene = rec.scene
        times = np.asarray(scene.time)
        check_times(times, "Original scene timestamps")
        ids = np.arange(start, min(len(scene), start + max_n), stride, dtype=np.int64)
        train, holdout = split_frames(ids, holdout_every)
        train_set = set(train.tolist())
        cal = rec.calibration
        if cal is None:
            raise ValueError("Neon calibration.bin is required")
        K = np.asarray(cal.scene_camera_matrix, dtype=float)
        distortion = np.asarray(cal.scene_distortion_coefficients, dtype=float)
        first = scene[int(ids[0])].bgr
        h, w = first.shape[:2]
        corrected, maps, height = rectification(K, distortion, (w, h), width)
        images = output / "images"
        images.mkdir()
        rows = []
        for n, frame_id in enumerate(ids):
            raw = first if n == 0 else scene[int(frame_id)].bgr
            if raw.shape[:2] != (h, w):
                raise ValueError("Scene image size changes")
            image = cv2.remap(raw, *maps, cv2.INTER_LINEAR)
            name = f"{int(frame_id):010d}.png"
            if not cv2.imwrite(str(images / name), image):
                raise OSError(f"Cannot write {name}")
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "time_ns": int(times[frame_id]),
                    "name": name,
                    "split": "train" if frame_id in train_set else "holdout",
                }
            )
        with (output / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "recording": str(recording),
            "source_frames": len(scene),
            "selected_frames": len(ids),
            "start_n": start,
            "max_n": max_n,
            "skip_n": stride,
            "image_size": [width, height],
            "K_rectified": corrected.tolist(),
            "K_original": K.tolist(),
            "distortion_original": distortion.tolist(),
            "inputs": [
                "scene images",
                "factory camera calibration",
                "original scene timestamps",
            ],
            "imu_used": False,
            "orb_slam_used": False,
            "pose_priors_used": False,
            "units": "arbitrary SfM scale; not metres",
            "gravity_alignment": False,
            "holdout_every": holdout_every,
        }
        write_json(output / "manifest.json", manifest)
        return rows, manifest
    finally:
        rec.close()


def training_errors(model):
    errors = []
    for point in model.points3D.values():
        for observation in point.track.elements:
            image = model.images[observation.image_id]
            projected = image.project_point(point.xyz)
            xy = image.points2D[observation.point2D_idx].xy
            errors.append(
                float(np.linalg.norm(projected - xy))
                if projected is not None
                else np.inf
            )
    return summarize_errors(errors)


def holdout_correspondences(db, query, model):
    votes = defaultdict(Counter)
    for image_id in model.reg_image_ids():
        if not db.exists_two_view_geometry(query.image_id, image_id):
            continue
        matches = db.read_two_view_geometry(query.image_id, image_id).inlier_matches
        image = model.images[image_id]
        for query_idx, train_idx in matches:
            point = image.points2D[int(train_idx)]
            if point.has_point3D() and point.point3D_id in model.points3D:
                votes[int(query_idx)][int(point.point3D_id)] += 1
    # Resolve conflicting matches using consensus, then enforce unique 3D points.
    choices = sorted(
        (
            (counts.most_common(1)[0][1], q, counts.most_common(1)[0][0])
            for q, counts in votes.items()
        ),
        reverse=True,
    )
    seen = set()
    selected = []
    for _, q, point_id in choices:
        if point_id not in seen:
            selected.append((q, point_id))
            seen.add(point_id)
    keypoints = db.read_keypoints(query.image_id)
    xy = np.array([keypoints[q, :2] for q, _ in selected], dtype=float)
    xyz = np.array([model.points3D[p].xyz for _, p in selected], dtype=float)
    return xy, xyz


def evaluate_holdout(model, database, rows):
    import pycolmap

    db = pycolmap.Database.open(database)
    records = []
    all_errors = []
    try:
        images = {im.name: im for im in db.read_all_images()}
        for row in rows:
            if row["split"] != "holdout":
                continue
            query = images[row["name"]]
            assert all(
                model.images[i].name != row["name"] for i in model.reg_image_ids()
            )
            xy, xyz = holdout_correspondences(db, query, model)
            record = {
                "frame_id": row["frame_id"],
                "correspondences": len(xy),
                "pose_estimated": False,
            }
            records.append(record)
            if len(xy) < 30:
                continue
            order = np.random.default_rng(row["frame_id"]).permutation(len(xy))
            cutoff = max(20, int(len(order) * 0.7))
            fit, test = order[:cutoff], order[cutoff:]
            camera = db.read_camera(query.camera_id)
            estimation = pycolmap.AbsolutePoseEstimationOptions()
            estimation.ransac.max_error = 4
            estimation.ransac.random_seed = 0
            result = pycolmap.estimate_and_refine_absolute_pose(
                xy[fit], xyz[fit], camera, estimation
            )
            if result is None or result["num_inliers"] < 15:
                continue
            pose = result["cam_from_world"]
            projected = camera.img_from_cam(pose * xyz[test])
            errors = np.linalg.norm(projected - xy[test], axis=1)
            errors[~np.isfinite(errors)] = np.inf
            record.update(
                pose_estimated=True,
                fit_inliers=int(result["num_inliers"]),
                test_observations=len(test),
                test_reprojection=summarize_errors(errors),
            )
            all_errors.extend(errors.tolist())
    finally:
        db.close()
    return {
        "images": len(records),
        "localized_images": sum(r["pose_estimated"] for r in records),
        "test_reprojection": summarize_errors(all_errors),
        "per_image": records,
        "method": (
            "held-out images excluded from SfM; pose fit on 70% of matches; "
            "evaluation on remaining 30%"
        ),
        "limitations": (
            "visual consistency of geometrically verified matches, "
            "not independent metric position accuracy"
        ),
    }


def export_model(model, rows, out):
    model.write(out)
    model.export_PLY(out / "points.ply")
    by_name = {row["name"]: row for row in rows}
    images = sorted(
        (model.images[i] for i in model.reg_image_ids()), key=lambda im: im.name
    )
    poses = []
    for image in images:
        pose = np.eye(4)
        pose[:3] = image.cam_from_world().inverse().matrix()
        poses.append(pose)
    np.savez_compressed(
        out / "camera_poses.npz",
        timestamps_ns=np.array(
            [by_name[im.name]["time_ns"] for im in images], dtype=np.int64
        ),
        source_frame_ids=np.array(
            [by_name[im.name]["frame_id"] for im in images], dtype=np.int64
        ),
        T_reconstruction_camera=np.array(poses),
        units="arbitrary SfM scale",
        gravity_aligned=False,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = list(model.points3D.values())[:: max(1, model.num_points3D() // 15000)]
    xyz = np.array([p.xyz for p in points])
    rgb = np.array([p.color for p in points]) / 255
    cameras = np.array(poses)[:, :3, 3]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*xyz.T, c=rgb, s=1, alpha=0.6)
    ax.scatter(*cameras.T, c=np.arange(len(cameras)), cmap="plasma", s=10)
    for axis in ("x", "y", "z"):
        getattr(ax, "set_" + axis + "label")(axis + " (SfM units)")
    ax.set_title("COLMAP image-only reconstruction: points and camera centers")
    fig.savefig(out / "reconstruction.png", dpi=150)
    plt.close(fig)


def run(args):
    import pycolmap

    if (
        args.start_n < 0
        or args.max_n < 10
        or args.skip_n < 1
        or args.width < 64
        or args.features < 100
        or args.overlap < 1
        or args.loop_stride < 1
    ):
        raise ValueError("Invalid frame range or image width")
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.reuse_data:
        source = Path(args.reuse_data).resolve()
        manifest = json.loads((source / "manifest.json").read_text())
        if Path(manifest["recording"]) != recording_path(args.input):
            raise ValueError("Reuse source belongs to another recording")
        with (source / "frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["frame_id"] = int(row["frame_id"])
            row["time_ns"] = int(row["time_ns"])
        (out / "images").symlink_to(source / "images", target_is_directory=True)
        shutil.copy2(source / "database.db", out / "database.db")
        shutil.copy2(source / "frames.csv", out / "frames.csv")
        write_json(out / "manifest.json", manifest)
        database = out / "database.db"
        K = np.array(manifest["K_rectified"])
        K[:2, 2] += 0.5
    else:
        rows, manifest = prepare_images(
            recording_path(args.input),
            out,
            args.start_n,
            args.max_n,
            args.skip_n,
            args.width,
            args.holdout_every,
        )
        database = out / "database.db"
        K = np.array(manifest["K_rectified"])
        # COLMAP pixel coordinates use image corners; OpenCV uses integer centers.
        K[:2, 2] += 0.5
        reader = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE",
            camera_params=",".join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]])),
        )
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = 4
        extraction.sift.max_num_features = args.features
        device = pycolmap.Device.cpu if args.cpu else pycolmap.Device.cuda
        pycolmap.extract_features(
            database,
            out / "images",
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=extraction,
            device=device,
        )
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = 4
        matching.max_num_matches = max(8192, args.features * 2)
        pairing = pycolmap.SequentialPairingOptions(
            overlap=args.overlap, quadratic_overlap=True
        )
        if args.matching == "exhaustive":
            pycolmap.match_exhaustive(
                database, matching_options=matching, device=device
            )
        else:
            pycolmap.match_sequential(
                database,
                matching_options=matching,
                pairing_options=pairing,
                device=device,
            )
            if args.matching == "hybrid":
                # Add long-range pairs without matching every full-resolution frame
                # to every other frame. Image order is the original source order.
                anchors = [row["name"] for row in rows[:: args.loop_stride]]
                pair_file = out / "long_range_pairs.txt"
                with pair_file.open("w") as stream:
                    for i, first in enumerate(anchors):
                        for second in anchors[i + 1 :]:
                            stream.write(f"{first} {second}\n")
                imported = pycolmap.ImportedPairingOptions(match_list_path=pair_file)
                pycolmap.match_image_pairs(
                    database,
                    matching_options=matching,
                    pairing_options=imported,
                    device=device,
                )
    train_names = [r["name"] for r in rows if r["split"] == "train"]
    if args.mapper == "global":
        options = pycolmap.GlobalPipelineOptions()
        options.image_names = train_names
        options.num_threads = 4
        options.random_seed = 0
        options.mapper.num_threads = 4
        options.mapper.random_seed = 0
        options.mapper.global_positioning.use_gpu = False
        options.mapper.bundle_adjustment.ceres.use_gpu = False
        options.mapper.bundle_adjustment.ceres.solver_options.num_threads = 4
        options.mapper.bundle_adjustment.refine_focal_length = False
        options.mapper.bundle_adjustment.refine_principal_point = False
        options.mapper.bundle_adjustment.refine_extra_params = False
        options.mapper.rotation_averaging.use_gravity = False
        models = pycolmap.global_mapping(
            database, out / "images", out / "sparse", options=options
        )
    else:
        options = pycolmap.IncrementalPipelineOptions()
        options.image_names = train_names
        options.num_threads = 4
        options.random_seed = 0
        options.mapper.random_seed = 0
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
        models = pycolmap.incremental_mapping(
            database, out / "images", out / "sparse", options=options
        )
    report = {
        "pycolmap_version": pycolmap.__version__,
        "device": "CPU" if args.cpu else "CUDA",
        "source_frames": manifest["source_frames"],
        "selected_images": len(rows),
        "training_images": len(train_names),
        "holdout_images": len(rows) - len(train_names),
        "imu_used": False,
        "orb_slam_used": False,
        "intrinsics_refined": False,
        "absolute_metric_accuracy": "not measurable without independent ground truth",
        "units": "arbitrary scale; not metres",
        "gravity_alignment": False,
        "pose_priors_used": False,
        "models": [],
        "settings": {
            "features": args.features,
            "overlap": args.overlap,
            "width": manifest["image_size"][0],
            "stride": manifest["skip_n"],
            "matching": args.matching,
            "mapper": args.mapper,
            "reuse_data": str(Path(args.reuse_data).resolve())
            if args.reuse_data
            else None,
            "loop_stride": args.loop_stride,
            "colmap_K": K.tolist(),
        },
    }
    registered = set()
    for model_id, model in sorted(
        models.items(), key=lambda x: x[1].num_reg_images(), reverse=True
    ):
        names = {model.images[i].name for i in model.reg_image_ids()}
        if not names <= set(train_names):
            raise ValueError("Held-out images leaked into reconstruction")
        registered |= names
        folder = out / "sparse" / str(model_id)
        export_model(model, rows, folder)
        report["models"].append(
            {
                "id": model_id,
                "registered_training_images": len(names),
                "points": model.num_points3D(),
                "mean_track_length": model.compute_mean_track_length(),
                "training_reprojection": training_errors(model),
                "holdout": evaluate_holdout(model, database, rows),
            }
        )
    report["registered_training_images_union"] = len(registered)
    report["localized_holdout_images_union"] = len(
        {
            row["frame_id"]
            for model in report["models"]
            for row in model["holdout"]["per_image"]
            if row["pose_estimated"]
        }
    )
    report["status"] = (
        "no_reconstruction"
        if not models
        else "disconnected_components"
        if len(models) > 1
        else "single_component"
    )
    report["largest_model_training_fraction"] = (
        report["models"][0]["registered_training_images"] / len(train_names)
        if models
        else 0
    )
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not models:
        raise ValueError(f"No reconstruction; inspect {out / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start_n", type=int, default=0)
    parser.add_argument("--max_n", type=int, default=999999)
    parser.add_argument("--skip_n", type=int, default=3)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument(
        "--matching",
        choices=["sequential", "exhaustive", "hybrid"],
        default="sequential",
    )
    parser.add_argument("--loop-stride", type=int, default=3)
    parser.add_argument(
        "--mapper", choices=["incremental", "global"], default="incremental"
    )
    parser.add_argument(
        "--reuse-data",
        help="Reuse image-only feature database from an earlier experiment",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
