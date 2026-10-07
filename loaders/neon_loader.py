"""Pupil Labs Neon scene video with optional, registered external geometry.

The scene camera is the rig. Without external poses, each frame has its own
assumed-upright coordinate system (x right, y forward, z up), not a shared world.
See README.md for the geometry NPZ contract and the monocular limitations.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.pose import PoseTW


class NeonLoader(BaseLoader):
    camera = "scene"
    device_name = "Pupil Labs Neon"

    @staticmethod
    def is_recording(path):
        path = Path(path)
        return path.is_dir() and any(path.glob("Neon Scene Camera v1 ps*.mp4"))

    def __init__(
        self,
        recording_dir,
        start_n=0,
        skip_n=1,
        max_n=None,
        resize=None,
        point_cloud="auto",
        geometry_path=None,
    ):
        if start_n < 0 or skip_n < 1 or (max_n is not None and max_n < 1):
            raise ValueError("Require start_n >= 0, skip_n >= 1 and max_n >= 1")
        if point_cloud not in ("auto", "on", "off"):
            raise ValueError("point_cloud must be auto, on or off")
        try:
            import pupil_labs.neon_recording as nr
            from pupil_labs.neon_recording.utils import find_sorted_multipart_files
        except ImportError as exc:
            raise ImportError(
                "Neon input requires: "
                "uv pip install 'pupil-labs-neon-recording>=2.1.6,<3'"
            ) from exc

        self.root = Path(recording_dir).expanduser()
        self.recording = nr.open(self.root)
        self.index = 0
        # Start prefetch only after run_boxer has set the model's image size.
        self._prefetch_thread = None
        self.resize = resize
        self._map_size = None
        try:
            self.scene = self.recording.scene
            calibration = self.recording.calibration
            if calibration is None:
                raise ValueError("Neon recording requires calibration.bin")
            self.K = np.array(calibration.scene_camera_matrix, dtype=np.float64)
            self.distortion = np.array(
                calibration.scene_distortion_coefficients, dtype=np.float64
            )
            if (
                not np.isfinite(self.K).all()
                or not np.isfinite(self.distortion).all()
                or min(self.K[0, 0], self.K[1, 1]) <= 0
            ):
                raise ValueError("Invalid Neon scene camera calibration")

            # Match the official reader's part order and .time timestamp choice.
            # Keep .time_aux separately; do not silently substitute a clock.
            pairs = find_sorted_multipart_files(
                self.root, "Neon Scene Camera v1", ".mp4"
            )
            videos = list(self.root.glob("Neon Scene Camera v1 ps*.mp4"))
            if len(pairs) != len(videos):
                raise ValueError("Every scene video part must have a .time file")
            times, aux_times = [], []
            for _, time_path in pairs:
                ts = self._read_times(time_path)
                aux_path = time_path.with_suffix(".time_aux")
                aux = (
                    self._read_times(aux_path)
                    if aux_path.exists() else np.full_like(ts, -1)
                )
                if len(aux) != len(ts):
                    raise ValueError(f"Timestamp count mismatch: {aux_path.name}")
                times.append(ts)
                aux_times.append(aux)
            self.timestamps_ns = np.concatenate(times)
            self.aux_timestamps_ns = np.concatenate(aux_times)
            if (
                len(self.timestamps_ns) != len(self.scene)
                or not np.array_equal(self.timestamps_ns, self.scene.time)
            ):
                raise ValueError("Scene video and .time samples do not match")
            if np.any(np.diff(self.timestamps_ns) <= 0):
                raise ValueError("Scene timestamps must be strictly increasing")
            self.frame_indices = np.arange(start_n, len(self.scene), skip_n)[:max_n]
            self.length = len(self.frame_indices)
            if not self.length:
                raise ValueError("No scene frames selected; check --start_n")

            self.poses = None
            self.points = None
            self.geometry_indices = None
            if geometry_path is not None:
                self._load_geometry(geometry_path, load_points=point_cloud != "off")
            self.has_world_poses = self.poses is not None
            self.use_points = point_cloud != "off" and self.points is not None
            if point_cloud == "on" and not self.use_points:
                raise ValueError(
                    "--point_cloud on requires --neon_geometry with points_world. "
                    "Use --point_cloud off for the original Neon recording."
                )
            self.coordinate_frame = (
                "world_z_up" if self.has_world_poses else "per_frame_assumed_upright"
            )
        except Exception:
            self.recording.close()
            raise

        print(f"==> Neon: {len(self.scene)} scene frames, {self.length} selected")
        print(f"==> Point cloud: {'on' if self.use_points else 'off'}")
        if not self.has_world_poses:
            print(
                "==> No camera trajectory: per-frame 3D predictions assume an upright "
                "camera; metric depth/size and gravity alignment are unverified. "
                "Tracking/fusion require registered camera poses."
            )

    @staticmethod
    def _read_times(path):
        if path.stat().st_size % 8:
            raise ValueError(f"Invalid timestamp file size: {path.name}")
        return np.fromfile(path, dtype="<i8")

    def _load_geometry(self, path, load_points):
        """Read exact scene-time poses and optional metric, Z-up world points."""
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            if not {"timestamps_ns", "T_world_camera"} <= set(data.files):
                raise ValueError("Geometry requires timestamps_ns and T_world_camera")
            ts = data["timestamps_ns"]
            if ts.ndim != 1 or ts.dtype != np.dtype("int64") or not len(ts):
                raise ValueError(
                    "Geometry timestamps_ns must be a nonempty int64 vector"
                )
            if np.any(np.diff(ts) <= 0):
                raise ValueError("Geometry timestamps_ns must be strictly increasing")
            selected_ts = self.timestamps_ns[self.frame_indices]
            indices = np.searchsorted(ts, selected_ts)
            if np.any(indices == len(ts)) or not np.array_equal(
                ts[indices], selected_ts
            ):
                raise ValueError(
                    "Geometry must contain the exact .time of every selected frame"
                )
            poses = np.asarray(data["T_world_camera"], dtype=np.float64)
            if poses.shape != (len(ts), 4, 4) or not np.isfinite(poses).all():
                raise ValueError(
                    "T_world_camera must contain finite (F, 4, 4) matrices"
                )
            rotation = poses[:, :3, :3]
            if (
                not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(
                    rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4
                )
                or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
            ):
                raise ValueError(
                    "T_world_camera must be rigid camera-to-world transforms"
                )
            self.poses = poses.astype(np.float32)
            self.geometry_indices = indices
            if load_points and "points_world" in data:
                points = np.asarray(data["points_world"], dtype=np.float32)
                if (
                    points.ndim not in (2, 3) or points.shape[-1] != 3
                    or (points.ndim == 3 and points.shape[0] != len(ts))
                ):
                    raise ValueError("points_world must have shape (N, 3) or (F, N, 3)")
                valid_rows = np.isfinite(points).all(axis=-1)
                padding_rows = np.isnan(points).all(axis=-1)
                if not (valid_rows | padding_rows).all() or not valid_rows.any():
                    raise ValueError(
                        "points_world needs finite points; "
                        "only all-NaN padding is allowed"
                    )
                self.points = points

    def load(self, idx):
        frame_index = int(self.frame_indices[idx])
        image = self.scene[frame_index].bgr
        height, width = image.shape[:2]
        if self.resize is None:
            out_w, out_h = width, height
        elif isinstance(self.resize, (int, np.integer)):
            out_w = out_h = int(self.resize)
        else:
            out_h, out_w = self.resize
        size = (width, height, out_w, out_h)
        if size != self._map_size:
            # OpenCV's resize convention: scale pixel centers, including the 0.5 offset.
            sx, sy = out_w / width, out_h / height
            K_out = self.K.copy()
            K_out[0, 0] *= sx
            K_out[1, 1] *= sy
            K_out[0, 2] = (K_out[0, 2] + 0.5) * sx - 0.5
            K_out[1, 2] = (K_out[1, 2] + 0.5) * sy - 0.5
            self._maps = cv2.initUndistortRectifyMap(
                self.K, self.distortion, None, K_out, (out_w, out_h), cv2.CV_32FC1
            )
            self._cam = self.pinhole_from_K(
                out_w, out_h, K_out[0, 0], K_out[1, 1], K_out[0, 2], K_out[1, 2]
            )
            self._map_size = size
        image = cv2.remap(image, *self._maps, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.has_world_poses:
            geometry_index = int(self.geometry_indices[idx])
            pose = PoseTW.from_matrix(torch.from_numpy(self.poses[geometry_index]))
        else:
            # OpenCV camera (right, down, forward) -> assumed Z-up local frame.
            pose = PoseTW.from_Rt(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
                torch.zeros(3),
            )
        points = torch.empty((0, 3), dtype=torch.float32)
        observed_points = None
        if self.use_points:
            per_frame = self.points.ndim == 3
            values = self.points[geometry_index] if per_frame else self.points
            points = torch.from_numpy(values[np.isfinite(values).all(axis=-1)])
            # A static map is not evidence that these points were observed now.
            if per_frame and len(points):
                observed_points = points
        return {
            "img0": self.img_to_tensor(image),
            "cam0": self._cam,
            "T_world_rig0": pose,
            "sdp_w": points,
            "observed_points": observed_points,
            "time_ns0": int(self.timestamps_ns[frame_index]),
            "frame_id": frame_index,
            "orig_size0": (width, height),
        }

    def write_metadata(self, output_dir, write_name):
        """Preserve the frame/time mapping even for frames with no detections."""
        root = Path(output_dir)
        metadata = {
            "dataset": "neon",
            "coordinate_frame": self.coordinate_frame,
            "point_cloud": "on" if self.use_points else "off",
            "depth_input": (
                "external_metric_points" if self.use_points else "missing (-1)"
            ),
            "timestamp_source": "Neon Scene Camera v1 ps*.time (int64 ns)",
            "images": "undistorted pinhole, resized; raw gaze needs the same transform",
            "image_size": self.resize,
            "scene_camera_matrix": self.K.tolist(),
            "scene_distortion_coefficients": self.distortion.tolist(),
            "has_world_poses": self.has_world_poses,
        }
        (root / f"{write_name}_input.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        indices = self.frame_indices
        np.savetxt(
            root / f"{write_name}_frames.csv",
            np.column_stack(
                (indices, self.timestamps_ns[indices], self.aux_timestamps_ns[indices])
            ),
            fmt="%d", delimiter=",", comments="",
            header="frame_id,time_ns,time_aux_ns",
        )

    def close(self):
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        self.recording.close()
