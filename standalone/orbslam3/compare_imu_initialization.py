#!/usr/bin/env python3
"""Compare numeric IMU graphs and repeat a graph in fresh native processes.

All vectors/matrices in snapshots are row-major. No SLAM images are loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def estimates(result):
    # Pose extrinsics are irrelevant to the fixed-pose IMU equations; compare
    # all variables actually consumed by those equations, keyed by vertex ID.
    return {int(v['id']): {k: v[k] for k in ('type', 'fixed', 'value', 'Rwb', 'twb') if k in v}
            for v in result['vertices']}


def numeric_difference(a, b, path=''):
    """Return the first ordered structural/numeric difference (not a verdict)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return dict(path=path, left_keys=list(a), right_keys=list(b))
        for key in a:
            delta = numeric_difference(a[key], b[key], f'{path}/{key}')
            if delta is not None:
                return delta
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return dict(path=path, left_length=len(a), right_length=len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            delta = numeric_difference(x, y, f'{path}/{i}')
            if delta is not None:
                return delta
        return None
    try:
        equal = float(a) == float(b)
    except (TypeError, ValueError):
        equal = a == b
    return None if equal else dict(path=path, left=a, right=b)


def max_estimate_error(a, b):
    a, b = estimates(a), estimates(b)
    if a.keys() != b.keys():
        raise ValueError('Different output vertex sets')
    errors = []
    for key in a:
        if a[key]['type'] != b[key]['type']:
            raise ValueError('Different output vertex types')
        for field in ('value', 'Rwb', 'twb'):
            if field in a[key]:
                errors.append(float(np.max(np.abs(np.asarray(a[key][field], float) -
                                                  np.asarray(b[key][field], float)))))
    return max(errors, default=0.)


def scale_distances(snapshot, result):
    """Camera-center distances before/after the scale-only application.

    Gravity rotation preserves distances; later full BA is deliberately separate.
    """
    vertices = {int(v['id']): v for v in snapshot['vertices_insertion_order']}
    scale = next(float(v['value']) for v in result['vertices'] if v['type'] == 'scale')
    frames = sorted((k for k in snapshot['keyframes_map_order'] if int(k['kf_id']) in vertices),
                    key=lambda k: float(k['time_s']))
    rows = []
    for a, b in zip(frames, frames[1:]):
        pa = np.asarray(vertices[int(a['kf_id'])].get('camera_center',vertices[int(a['kf_id'])]['twb']), float)
        pb = np.asarray(vertices[int(b['kf_id'])].get('camera_center',vertices[int(b['kf_id'])]['twb']), float)
        before = float(np.linalg.norm(pb-pa))
        rows.append(dict(source_a=int(a['source_frame_id']), source_b=int(b['source_frame_id']),
                         before=before, scale=scale, after_scale_only=before*scale))
    return rows



def observed_distances(path, distances):
    for stage in ('after_scale','after_full_ba'):
        checkpoint=Path(str(path)+'.'+stage+'.json')
        if not checkpoint.exists():
            continue
        positions={int(k['source_frame_id']):np.asarray(k['camera_center'],float)
                   for k in read(checkpoint)['keyframes']}
        for row in distances:
            a,b=row['source_a'],row['source_b']
            if a in positions and b in positions:
                row[stage]=float(np.linalg.norm(positions[b]-positions[a]))
    return distances


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--compare', type=Path)
    p.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[2]/
                   'output/orbslam3_standalone/build_diagnostics/replay_imu_initialization')
    p.add_argument('--correct-scale-jacobian',action='store_true',help='Experimental replay-only chain-rule correction; live SLAM is unchanged')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('--repeats must be at least 2')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.binary.resolve().parent.parent
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(args.binary.resolve().parent)+':'+str(runtime/'prefix/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    graph = read(args.snapshot)
    original = read(str(args.snapshot)+'.result.json')
    runs = []
    for i in range(args.repeats):
        path = args.output/f'replay_{i:02}.json'
        command=[str(args.binary.resolve()),str(args.snapshot.resolve()),str(path.resolve())]
        if args.correct_scale_jacobian:command.append('--correct-scale-jacobian')
        subprocess.run(command,
                       env=env, check=True)
        runs.append(read(path))
    summary = dict(correct_scale_jacobian=args.correct_scale_jacobian,
                   binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                   library_sha256=hashlib.sha256((args.binary.parent/"libstandalone_orb.so").read_bytes()).hexdigest(),snapshot=str(args.snapshot.resolve()),
                   snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                   repeats=args.repeats, max_error_vs_native=[max_estimate_error(original, r) for r in runs],
                   max_error_vs_first=[max_estimate_error(runs[0], r) for r in runs],
                   native_chi2=float(original["chi2"]), chi2=[float(r['chi2']) for r in runs],
                   native_scale_only_distances=observed_distances(args.snapshot,scale_distances(graph, original)),
                   scale_only_distances=scale_distances(graph,runs[0]) if args.correct_scale_jacobian else observed_distances(args.snapshot,scale_distances(graph, original)))
    if args.compare:
        other = read(args.compare)
        summary['first_input_difference'] = numeric_difference(graph, other)
        summary['other_scale_only_distances'] = observed_distances(args.compare,scale_distances(other, read(str(args.compare)+'.result.json')))
    (args.output/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if 'distances' not in k}, indent=2))


if __name__ == '__main__':
    main()
