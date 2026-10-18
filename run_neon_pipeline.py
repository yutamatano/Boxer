#!/usr/bin/env python3
"""Prepare Neon, estimate trajectories, infer all frames, and assemble replay."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from convert_neon_trajectory import convert
from evaluate_neon_geometry import evaluate
from utils.neon_pipeline_viewer import assemble_replay, selected_frame_ids
from utils.neon_progress import StageProgress
from utils.neon_vio import read_csv, recording_path, write_json

ROOT = Path(__file__).resolve().parent


def gaze_options(args):
    options = []
    if getattr(args, "gaze_targets", False):
        options.append("--gaze_targets")
    elif getattr(args, "neon_gaze", False):
        options.append("--neon_gaze")
    if options:
        options += ["--gaze_max_delta_ms", getattr(args, "gaze_max_delta_ms", 10.)]
    return options


def run_logged(command, log, env=None, allowed=(0,), *, progress=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    print(f"Log: {log}", flush=True)
    worker_env = dict(os.environ if env is None else env)
    worker_env["PYTHONUNBUFFERED"] = "1"
    if progress:
        progress.update(progress.offset, status="開始", force=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=worker_env,
            cwd=ROOT,
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if progress:
                        progress.poll(log)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if progress:
        progress.poll(log)
        progress.finish(code in allowed)
    if code not in allowed:
        raise RuntimeError(f"Command failed ({code}); inspect {log}")
    return code


def run_boxer_segments(summary, out, args):
    recording = Path(summary["recording"])
    completed = 0
    total = summary.get("input_frames", sum(s["max_n"] for s in summary["segments"]))
    for i, segment in enumerate(summary["segments"]):
        results = out / f"boxer_{i:03d}"
        command = [
            sys.executable,
            ROOT / "run_boxer.py",
            "--input",
            recording,
            "--neon_geometry",
            segment["geometry"],
            "--start_n",
            segment["start_n"],
            "--max_n",
            segment["max_n"],
            "--skip_n",
            1,
            "--point_cloud",
            getattr(args, "point_cloud", "auto"),
            "--track",
            "--output_dir",
            results,
            "--labels",
            args.labels,
        ]
        if args.force_cpu:
            command += ["--force_cpu"]
        if getattr(args, "require_gpu", False):
            command += ["--require_gpu"]
        if getattr(args, "skip_viz", False):
            command += ["--skip_viz"]
        command += gaze_options(args)
        run_logged(command, out / f"boxer_{i:03d}.log", progress=StageProgress(
            out, "物体検出・3D推論", total, offset=completed,
            source="inference", detail=f"軌跡あり区間 {i + 1}/{len(summary['segments'])}",
        ))
        completed += segment["max_n"]
        if args.fuse:
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "utils.fuse_3d_boxes",
                    "--input",
                    results / recording.name / "boxer_3dbbs.csv",
                ],
                out / f"fusion_{i:03d}.log",
                progress=StageProgress(out, "3D物体融合", 1,
                                       detail=f"区間 {i + 1}/{len(summary['segments'])}"),
            )
        segment["results"] = str(results / recording.name)
        write_json(out / "pipeline.json", summary)


def uncovered_intervals(frame_ids, segments):
    """Partition actual selected source IDs; never infer a pose across a gap."""
    ids = [int(i) for i in frame_ids]
    if not ids or any(b <= a for a, b in zip(ids[:-1], ids[1:])):
        raise ValueError("Selected frame IDs must be nonempty and strictly increasing")
    covered = set()
    for segment in segments:
        interval = set(range(segment["start_n"], segment["start_n"] + segment["max_n"]))
        if interval & covered or not interval <= set(ids):
            raise ValueError("Metric intervals overlap or lie outside selected frames")
        covered |= interval
    missing = [i for i in ids if i not in covered]
    runs = []
    for frame_id in missing:
        if runs and frame_id == runs[-1]["start_n"] + runs[-1]["max_n"]:
            runs[-1]["max_n"] += 1
        else:
            runs.append({"start_n": frame_id, "max_n": 1})
    return runs


def run_frame_estimates(summary, out, args):
    recording = Path(summary["recording"])
    frame_ids = selected_frame_ids(summary["prepared"])
    intervals = uncovered_intervals(frame_ids, summary["segments"])
    completed = summary["usable_frames"]
    summary["frame_only_intervals"] = []
    for i, interval in enumerate(intervals):
        results = out / f"frame_only_{i:03d}"
        command = [
            sys.executable,
            ROOT / "run_boxer.py",
            "--input",
            recording,
            "--start_n",
            interval["start_n"],
            "--max_n",
            interval["max_n"],
            "--skip_n",
            1,
            "--point_cloud",
            "off",
            "--output_dir",
            results,
            "--labels",
            args.labels,
        ]
        if args.force_cpu:
            command += ["--force_cpu"]
        if getattr(args, "require_gpu", False):
            command += ["--require_gpu"]
        if getattr(args, "skip_viz", False):
            command += ["--skip_viz"]
        command += gaze_options(args)
        run_logged(command, out / f"frame_only_{i:03d}.log", progress=StageProgress(
            out, "物体検出・3D推論", len(frame_ids), offset=completed,
            source="inference", detail=f"軌跡なし区間 {i + 1}/{len(intervals)}",
        ))
        completed += interval["max_n"]
        summary["frame_only_intervals"].append(
            dict(
                interval,
                results=str(results / recording.name),
                coordinate_frame="per_frame_assumed_upright",
                camera_trajectory_available=False,
            )
        )
        summary["frame_only_frames"] = sum(
            s["max_n"] for s in summary["frame_only_intervals"]
        )
        summary["total_estimated_frames"] = (
            summary["usable_frames"] + summary["frame_only_frames"]
        )
        write_json(out / "pipeline.json", summary)


def evaluate_segments(summary, out, reference=None):
    if not summary["segments"]:
        StageProgress(out, "軌跡評価", 0).update(0, status="対象区間なし", force=True)
        write_json(out / "pipeline.json", summary)
        return
    with StageProgress(out, "軌跡評価", len(summary["segments"])) as progress:
        for i, segment in enumerate(summary["segments"]):
            output = out / f"geometry_evaluation_{i:03d}.json"
            evaluate(summary["prepared"], segment["geometry"], output, reference)
            segment["evaluation"] = str(output)
            progress.update(i + 1)
    write_json(out / "pipeline.json", summary)


def run_inference(summary, out, args):
    run_boxer_segments(summary, out, args)
    run_frame_estimates(summary, out, args)
    with StageProgress(out, "Viewer用結果統合", 1):
        viewer_output = assemble_replay(summary, out)
    command = [sys.executable, str(ROOT / "view_tracker.py"), "--input",
               summary["recording"], "--output_dir", str(viewer_output), "--autoplay"]
    if getattr(args, "require_gpu", False):
        command.append("--require_gpu")
    summary["viewer_command"] = command
    write_json(out / "pipeline.json", summary)
    print("Viewer:", shlex.join(command), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", required=True, help="New run directory, e.g. output/neon_run_01"
    )
    parser.add_argument(
        "--prepared",
        help="Reuse existing prepared data; its clock/calibration settings apply",
    )
    parser.add_argument("--start_n", type=int, default=0)
    parser.add_argument("--max_n", type=int, default=999999)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--orb-features", type=int, default=2000)
    parser.add_argument("--imu-offset-ms", type=float, default=0.0)
    parser.add_argument("--time-source", choices=["primary", "aux"], default="primary")
    parser.add_argument("--imu-noise")
    parser.add_argument("--experimental-noise", action="store_true")
    parser.add_argument("--extrinsics")
    parser.add_argument(
        "--min-frames", type=int, default=2,
        help="Minimum consecutive valid metric poses (default: 2; use 30 for the former length filter)",
    )
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--labels", default="lvisplus")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--require_gpu", action="store_true",
                        help="Require CUDA for inference; disable CPU fallback")
    parser.add_argument("--skip_viz", action="store_true",
                        help="Skip inference JPEG/MP4 output; Viewer CSVs are still saved")
    parser.add_argument("--point_cloud", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--neon_gaze", action="store_true")
    parser.add_argument("--gaze_targets", action="store_true")
    parser.add_argument("--gaze_max_delta_ms", type=float, default=10.)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument(
        "--frame-fallback",
        action="store_true",
        help="Compatibility option: untracked frames are now always processed",
    )
    parser.add_argument(
        "--reference",
        help="Independent reference trajectory NPZ for accuracy evaluation",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--vio-only", action="store_true")
    mode.add_argument(
        "--boxer-only",
        action="store_true",
        help="Use pipeline.json in an existing output directory",
    )
    args = parser.parse_args()
    if args.require_gpu:
        if args.force_cpu:
            parser.error("--require_gpu cannot be combined with --force_cpu")
        import torch
        if not torch.cuda.is_available():
            parser.error("--require_gpu: CUDA is unavailable; CPU fallback is disabled")
    try:
        recording = recording_path(args.input)
        out = Path(args.output).expanduser().resolve()
        if args.boxer_only:
            summary = json.loads((out / "pipeline.json").read_text())
            if Path(summary["recording"]).resolve() != recording:
                raise ValueError("Existing pipeline has no matching usable recording")
            evaluate_segments(summary, out, args.reference)
            run_inference(summary, out, args)
            StageProgress(out, "選択した全工程", 1).finish(True)
            print(f"Results: {out}")
            return
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output is not empty; choose a new directory: {out}")
        if args.min_frames < 2:
            raise ValueError("--min-frames must be >= 2")
        binary = ROOT / "output/neon_vio_build/mono_inertial_neon"
        vocabulary = ROOT / "output/neon_vio_deps/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        if not binary.is_file() or not vocabulary.is_file():
            raise ValueError("Build the adapter first: bash scripts/build_neon_vio.sh")
        out.mkdir(parents=True, exist_ok=True)
        prepared = (
            Path(args.prepared).expanduser().resolve()
            if args.prepared
            else out / "prepared"
        )
        if args.prepared:
            manifest = json.loads((prepared / "manifest.json").read_text())
            if Path(manifest["recording"]).resolve() != recording:
                raise ValueError("Prepared input belongs to a different recording")
            print(
                "Reusing prepared input, including its calibration and clock settings.",
                flush=True,
            )
            StageProgress(out, "前処理", 1).update(1, status="既存データを再利用", force=True)
        else:
            command = [
                sys.executable,
                ROOT / "prepare_neon_vio.py",
                "--input",
                recording,
                "--output",
                prepared,
                "--start_n",
                args.start_n,
                "--max_n",
                args.max_n,
                "--width",
                args.width,
                "--imu-offset-ms",
                args.imu_offset_ms,
                "--time-source",
                args.time_source,
                "--orb-features",
                args.orb_features,
                "--check-sync",
            ]
            for flag in ("imu_noise", "extrinsics"):
                if getattr(args, flag):
                    command += ["--" + flag.replace("_", "-"), getattr(args, flag)]
            if args.experimental_noise:
                command += ["--experimental-noise"]
            run_logged(command, out / "prepare.log", progress=StageProgress(
                out, "前処理", None, source="prepare",
            ))
            manifest = json.loads((prepared / "manifest.json").read_text())
        frames = read_csv(prepared / "frames.csv")
        env = os.environ.copy()
        prefix = ROOT / "output/neon_vio_deps/prefix/usr/lib"
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(prefix / "x86_64-linux-gnu"),
                str(prefix),
                env.get("LD_LIBRARY_PATH", ""),
            ]
        )
        summary = {
            "recording": str(recording),
            "prepared": str(prepared),
            "input_frames": len(selected_frame_ids(prepared)),
            "prepared_frames": len(frames),
            "sessions": [],
            "segments": [],
            "imu_offset_ms_applied": manifest["imu_offset_ms_applied"],
            "imu_noise_source": manifest["imu_noise_source"],
            "accuracy": (
                "unvalidated without ground truth; segments have independent origins"
            ),
        }
        start_index = 0
        while start_index < len(frames) - 1:
            session_id = len(summary["sessions"])
            session = out / f"session_{session_id:03d}"
            session.mkdir()
            run_logged(
                [
                    binary,
                    vocabulary,
                    prepared,
                    session / "slam",
                    args.pace,
                    "--stop-on-metric-loss",
                    "--start-index",
                    start_index,
                ],
                session / "slam.log",
                env,
                allowed=(0, 3),
                progress=StageProgress(
                    out, "カメラ軌跡推定", len(frames), offset=start_index,
                    source="slam", states=session / "slam/states.csv",
                    detail=f"セッション {session_id + 1}",
                ),
            )
            state = json.loads((session / "slam/run.json").read_text())
            next_index = state["next_index"]
            if not start_index < next_index <= len(frames):
                raise ValueError("SLAM did not advance through the selected frames")
            state["session"] = session_id
            summary["sessions"].append(state)
            geometry = session / "geometry"
            with StageProgress(out, "軌跡変換", 1, detail=f"セッション {session_id + 1}"):
                try:
                    report = convert(
                        prepared,
                        session / "slam/camera_trajectory.csv",
                        session / "slam/states.csv",
                        geometry,
                        args.min_frames,
                        points=session / "slam/map_points.csv",
                        allow_partial=True,
                    )
                except ValueError:
                    # Only an explicitly diagnosed lack of usable intervals is recoverable.
                    if not (geometry / "report.json").is_file():
                        raise
                    report = json.loads((geometry / "report.json").read_text())
                    if report["segments"]:
                        raise
                    print(
                        f"Session {session_id}: no usable metric interval; "
                        f"see {geometry / 'report.json'}",
                        flush=True,
                    )
            for segment in report["segments"]:
                item = dict(
                    segment,
                    geometry=str(geometry / segment["file"]),
                    session=session_id,
                )
                summary["segments"].append(item)
            write_json(out / "pipeline.json", summary)
            start_index = next_index
        summary["usable_frames"] = sum(s["max_n"] for s in summary["segments"])
        summary["coverage"] = summary["usable_frames"] / summary["input_frames"]
        write_json(out / "pipeline.json", summary)
        evaluate_segments(summary, out, args.reference)
        if not args.vio_only:
            run_inference(summary, out, args)
        print(f"Usable metric frames: {summary['usable_frames']}/{summary['input_frames']}")
        print(f"Summary: {out / 'pipeline.json'}")
        if not args.vio_only and not args.skip_viz:
            print("Remote SSH: download the MP4 in VS Code, or forward port 8000 and run:")
            print(
                f"  {sys.executable} -m http.server 8000 --bind 127.0.0.1 --directory {out}"
            )
        StageProgress(out, "選択した全工程", 1).finish(True)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
