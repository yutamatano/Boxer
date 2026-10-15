"""Convert a Neon recording's Tag Aligner poses to Boxer geometry segments."""

import argparse
import json
from pathlib import Path

import numpy as np

from loaders.neon_loader import NeonLoader
from utils.tag_aligner import AlignedPoses, WORLD_FROM_TAG


def convert(recording, aligned_poses, output, meters_per_unit):
    recording, aligned_poses, output = map(Path, (recording, aligned_poses, output))
    info = json.loads((recording / "info.json").read_text())
    poses = AlignedPoses(aligned_poses, info.get("start_time"), meters_per_unit)
    loader = NeonLoader(recording, point_cloud="off")
    try:
        times = loader.timestamps_ns.copy()
    finally:
        loader.close()
    matches = poses.match(times)
    covered = np.flatnonzero(matches >= 0)
    if not len(covered):
        raise ValueError("No scene frames overlap aligned poses; verify recording and clock")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source": "tag_aligner",
        "pose_file": aligned_poses.name,
        "meters_per_source_unit": meters_per_unit,
        "world_rotation_from_tag_aligner": WORLD_FROM_TAG.tolist(),
        "world_axes": "x=tag.x, y=tag.z, z=-tag.y; metres; Z-up",
        "recording_start_ns": info["start_time"],
        "time_matching": "relative seconds + start_time; [start,end); no interpolation",
        "pose_convention": "T_world_camera; OpenCV camera axes; quaternion xyzw",
        "accuracy": "depends on supplied Tag Aligner alignment; not independently measured",
    }
    segments = []
    # Separate absent pose intervals even if no video frame fell in the gap.
    previous, following = matches[covered[:-1]], matches[covered[1:]]
    gap_prefix = np.r_[0, np.cumsum(poses.starts[1:] > poses.ends[:-1])]
    split = (np.diff(covered) != 1) | (gap_prefix[following] != gap_prefix[previous])
    for index, frame_ids in enumerate(np.split(covered, np.flatnonzero(split) + 1)):
        path = output / f"geometry_segment_{index:03d}.npz"
        np.savez_compressed(
            path, timestamps_ns=times[frame_ids],
            T_world_camera=poses.matrices[matches[frame_ids]],
            source_frame_ids=frame_ids,
            alignment_metadata=json.dumps(provenance),
        )
        segments.append({
            "geometry": path.name, "start_n": int(frame_ids[0]),
            "max_n": len(frame_ids),
        })
    report = {
        **provenance, "total_scene_frames": len(times),
        "aligned_frames": len(covered), "missing_frames": len(times) - len(covered),
        "segments": segments,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Native Neon recording directory")
    parser.add_argument("--aligned-poses", required=True, help="Same recording's aligned_poses.csv")
    parser.add_argument("--output", required=True, help="Empty output directory")
    parser.add_argument("--meters-per-unit", required=True, type=float,
                        help="Metres per surveyed Tag Aligner unit: 1 for m, 0.001 for mm")
    args = parser.parse_args()
    try:
        report = convert(args.input, args.aligned_poses, args.output, args.meters_per_unit)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
