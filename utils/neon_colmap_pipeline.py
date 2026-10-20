"""COLMAP-only image preparation and exact input selection; no ORB executable."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from utils.neon_vio import (
    read_csv,
    write_json,
    rectification,
    relative_seconds,
    scene_to_imu,
    validate_transform,
)


def select_prepared(source, output, recording, start, count):
    source, output = (
        Path(source).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    manifest = json.loads((source / "manifest.json").read_text())
    if Path(manifest["recording"]).resolve() != Path(recording).resolve():
        raise ValueError("Prepared input belongs to a different recording")
    rows = [
        r
        for r in read_csv(source / "frames.csv")
        if start <= int(r["frame_id"]) < start + count
    ]
    excluded = [
        int(i)
        for i in manifest.get("excluded_without_imu", [])
        if start <= int(i) < start + count
    ]
    if not rows:
        raise ValueError("No prepared frames in requested start_n/max_n range")
    ids = [int(r["frame_id"]) for r in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("Invalid prepared frame selection")
    # Detect holes in the supplied prepared slice, rather than silently expanding it.
    expected = set(range(start, min(start + count, max(ids + excluded) + 1)))
    if set(ids) | set(excluded) != expected:
        raise ValueError("Prepared input does not cover the requested source interval")
    output.mkdir()
    for r in rows:
        r["image"] = str((source / r["image"]).resolve())
    with (output / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ["camera.yaml", "imu.csv"]:
        if (source / name).exists():
            (output / name).symlink_to((source / name).resolve())
    manifest.update(
        selection_source_prepared=manifest.get(
            "selection_source_prepared", str(source)
        ),
        selected_start_n=start,
        selected_max_n=count,
        prepared_frames=len(rows),
        excluded_without_imu=excluded,
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def prepare_visual(
    recording,
    output,
    start,
    count,
    width=800,
    imu_offset_ms=0.0,
    time_source="primary",
    extrinsics=None,
):
    from loaders.neon_loader import NeonLoader
    from utils.neon_vio import imu_in_si

    transform = (
        validate_transform(json.loads(Path(extrinsics).expanduser().read_text()))
        if extrinsics
        else scene_to_imu()
    )
    output = Path(output)
    output.mkdir()
    (output / "images").mkdir()
    loader = NeonLoader(recording, start_n=start, max_n=count, point_cloud="off")
    try:
        ids = loader.frame_indices
        first = loader.scene[int(ids[0])].bgr
        h, w = first.shape[:2]
        K, maps, height = rectification(loader.K, loader.distortion, (w, h), width)
        physical = (
            loader.timestamps_ns
            if time_source == "primary"
            else loader.aux_timestamps_ns
        )
        if np.any(physical[ids] < 0):
            raise ValueError("Requested physical image clock is missing")
        origin = int(physical[0])
        seconds = relative_seconds(physical, origin)
        rows = []
        for n, i in enumerate(ids):
            image = first if n == 0 else loader.scene[int(i)].bgr
            if image.shape[:2] != (h, w):
                raise ValueError("Scene resolution changed")
            path = output / "images" / f"{int(i):010d}.png"
            gray = cv2.cvtColor(
                cv2.remap(image, *maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY
            )
            if not cv2.imwrite(str(path), gray):
                raise OSError("Could not write prepared image")
            rows.append(
                dict(
                    frame_id=int(i),
                    time_ns=int(loader.timestamps_ns[i]),
                    time_aux_ns=int(loader.aux_timestamps_ns[i]),
                    slam_time_s=float(seconds[i]),
                    image=str(path.resolve()),
                )
            )
            if n % 100 == 0:
                print(f"[prepare visual] {n + 1}/{len(ids)}", flush=True)
        with (output / "frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        settings = {
            "Camera1.fx": K[0, 0],
            "Camera1.fy": K[1, 1],
            "Camera1.cx": K[0, 2],
            "Camera1.cy": K[1, 2],
            "Camera.width": width,
            "Camera.height": height,
        }
        (output / "camera.yaml").write_text(
            "%YAML:1.0\n"
            + "".join(f"{k}: {float(v):.17g}\n" for k, v in settings.items())
        )
        imu_available = False
        if any(Path(recording).glob("imu ps*.raw")):
            imu = loader.recording.imu[:]
            gyro, accel = imu_in_si(imu)
            times = (
                relative_seconds(np.asarray(imu.time, np.int64), origin)
                + imu_offset_ms * 1e-3
            )
            np.savetxt(
                output / "imu.csv",
                np.column_stack([times, gyro, accel]),
                delimiter=",",
                header="time_s,gx,gy,gz,ax,ay,az",
                comments="",
                fmt="%.17g",
            )
            imu_available = True
        meta = dict(
            recording=str(Path(recording).resolve()),
            pose_frame="scene_camera",
            preparation="visual_only; no ORB binary or noise parameters required",
            prepared_frames=len(ids),
            selected_start_n=start,
            selected_max_n=count,
            excluded_without_imu=[],
            image_size=[width, height],
            K_rectified=K.tolist(),
            image_clock=time_source,
            imu_offset_ms_applied=imu_offset_ms,
            imu_noise_source="not used by visual SfM",
            T_imu_camera=transform.tolist(),
            extrinsics_source=str(extrinsics)
            if extrinsics
            else "manufacturer nominal; not independently calibrated",
            physical_synchronization="not independently verified",
            imu_available=imu_available,
        )
        write_json(output / "manifest.json", meta)
        return meta
    finally:
        loader.close()
