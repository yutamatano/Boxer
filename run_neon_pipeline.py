#!/usr/bin/env python3
"""Prepare Neon, preserve metric SLAM intervals, and run Boxer per interval."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from convert_neon_trajectory import convert
from evaluate_neon_geometry import evaluate
from utils.neon_vio import read_csv, recording_path, write_json

ROOT = Path(__file__).resolve().parent


def run_logged(command, log, env=None, allowed=(0,)):
    print("Running:", " ".join(map(str, command)), flush=True)
    print(f"Log: {log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            list(map(str, command)),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
        )
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")
    return result.returncode


def run_boxer_segments(summary, out, args):
    recording = Path(summary["recording"])
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
            "auto",
            "--track",
            "--output_dir",
            results,
            "--labels",
            args.labels,
        ]
        if args.force_cpu:
            command += ["--force_cpu"]
        run_logged(command, out / f"boxer_{i:03d}.log")
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
    frames = read_csv(Path(summary["prepared"]) / "frames.csv")
    intervals = uncovered_intervals(
        [row["frame_id"] for row in frames], summary["segments"]
    )
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
        run_logged(command, out / f"frame_only_{i:03d}.log")
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
    for i, segment in enumerate(summary["segments"]):
        output = out / f"geometry_evaluation_{i:03d}.json"
        evaluate(summary["prepared"], segment["geometry"], output, reference)
        segment["evaluation"] = str(output)
    write_json(out / "pipeline.json", summary)


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
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--labels", default="lvisplus")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument(
        "--frame-fallback",
        action="store_true",
        help="Process untracked frames as independent local 3D estimates",
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
    try:
        recording = recording_path(args.input)
        out = Path(args.output).expanduser().resolve()
        if args.boxer_only:
            summary = json.loads((out / "pipeline.json").read_text())
            if Path(summary["recording"]).resolve() != recording or (
                not summary["segments"] and not args.frame_fallback
            ):
                raise ValueError("Existing pipeline has no matching usable recording")
            run_boxer_segments(summary, out, args)
            evaluate_segments(summary, out, args.reference)
            if args.frame_fallback:
                run_frame_estimates(summary, out, args)
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
            run_logged(command, out / "prepare.log")
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
            "input_frames": len(frames),
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
            )
            state = json.loads((session / "slam/run.json").read_text())
            next_index = state["next_index"]
            if not start_index < next_index <= len(frames):
                raise ValueError("SLAM did not advance through the selected frames")
            state["session"] = session_id
            summary["sessions"].append(state)
            geometry = session / "geometry"
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
        summary["coverage"] = summary["usable_frames"] / len(frames)
        write_json(out / "pipeline.json", summary)
        if not summary["segments"] and not args.frame_fallback:
            raise ValueError(
                f"No metric intervals. Inspect {out / 'pipeline.json'} and session logs"
            )
        evaluate_segments(summary, out, args.reference)
        if not args.vio_only:
            run_boxer_segments(summary, out, args)
            if args.frame_fallback:
                run_frame_estimates(summary, out, args)
        print(f"Usable metric frames: {summary['usable_frames']}/{len(frames)}")
        print(f"Summary: {out / 'pipeline.json'}")
        print("Remote SSH: download the MP4 in VS Code, or forward port 8000 and run:")
        print(
            f"  {sys.executable} -m http.server 8000 --bind 127.0.0.1 --directory {out}"
        )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
