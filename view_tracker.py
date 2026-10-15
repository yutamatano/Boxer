#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""View pre-computed BoxerNet results in 3D tracker mode."""

import argparse
import os

from utils.file_io import read_obb_csv
from utils.viewer_3d import (
    TrackerViewer,
    add_common_args,
    build_seq_ctx,
    launch_viewer,
    load_common,
    resolve_bb2d_csv,
    scale_factor,
    subsample_timed_obbs,
)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="View BoxerNet results (tracker mode)")
    add_common_args(parser)
    parser.add_argument("--init_follow", action="store_true", help="Initialize with Follow View enabled")
    parser.add_argument("--init_follow_behind", type=float, default=None, help="Follow-view behind distance (meters)")
    parser.add_argument("--init_follow_above", type=float, default=6.0, help="Follow-view above distance (meters)")
    parser.add_argument("--init_follow_look_ahead", type=float, default=None, help="Follow-view look-ahead distance (meters)")
    parser.add_argument("--init_show_obs", action="store_true", help="Initially show observed points")
    parser.add_argument("--autoplay", action="store_true", help="Automatically start playback")
    parser.add_argument("--autorecord", action="store_true")
    parser.add_argument("--record_fps", type=float, default=0.0, help="Recording FPS (0 = auto)")
    parser.add_argument("--teaser", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tracker logging")
    parser.add_argument("--bb2d_csv", type=str, default="", help="2D BB CSV filename (relative to log_dir)")
    # fmt: on
    parser.add_argument(
        "--neon_geometry", type=str,
        help="The same metric Z-up geometry NPZ used to generate the Neon CSV",
    )
    parser.add_argument(
        "--point_cloud", choices=("auto", "on", "off"), default="auto",
        help="Neon replay point cloud: auto if available, require on, or disable off",
    )
    args = parser.parse_args()
    if args.start_n < 0 or args.skip_n < 1 or args.max_n < 0:
        parser.error("Require start_n >= 0, skip_n >= 1 and max_n >= 0")

    input_path, dataset_type, seq_name, log_dir, view_path, load_view_data = (
        load_common(args)
    )

    # Load OBBs
    raw_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    # Online tracking needs per-frame detections, not already-fused track history.
    csv_path = raw_csv
    if not os.path.exists(csv_path):
        raise IOError(f"3D BB CSV not found: {csv_path}")

    if args.verbose:
        print(f"==> Loading OBBs from {csv_path}")
    timed_obbs = read_obb_csv(csv_path)
    bb2d_csv_path = resolve_bb2d_csv(log_dir, args.bb2d_csv, args.write_name)
    if dataset_type == "neon":
        from utils.neon_viewer import build_neon_context

        try:
            seq_ctx, timed_obbs = build_neon_context(
                input_path, args.neon_geometry, log_dir, args.write_name, timed_obbs,
                point_cloud=args.point_cloud,
            )
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
    else:
        seq_ctx = build_seq_ctx(input_path, dataset_type)
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    if args.verbose:
        total_dets = sum(len(obbs) for obbs in timed_obbs.values())
        print(f"==> Loaded {len(timed_obbs)} frames, {total_dets} detections")

    default_w, default_h = 2250 * scale_factor, 1100 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class Viewer(TrackerViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                bb2d_csv_path=bb2d_csv_path,
                init_rgb_text_scale=args.init_rgb_text_scale,
                init_color_mode=args.init_color_mode,
                init_follow=args.init_follow,
                init_follow_behind=args.init_follow_behind,
                init_follow_above=args.init_follow_above,
                init_follow_look_ahead=args.init_follow_look_ahead,
                init_show_obs=args.init_show_obs,
                init_image_panel_width=args.init_image_panel_width,
                autorecord=args.autorecord,
                record_fps=args.record_fps,
                teaser=args.teaser,
                verbose=args.verbose,
                load_view_data=load_view_data,
                view_save_path=view_path,
                scannet_scene=args.scannet_scene,
                scannet_annotation_path=args.scannet_annotation_path,
                **kw,
            )
            if args.autoplay:
                self.is_playing = True
                self.follow_view = True

    try:
        launch_viewer(Viewer)
    finally:
        if dataset_type == "neon":
            seq_ctx["loader"].close()


if __name__ == "__main__":
    main()
