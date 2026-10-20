"""Run and audit the fixed-input ORB / learned-COLMAP development comparison.

Neural inference requires CUDA. Geometric solvers retain the existing CPU
implementation. Repeats rebuild empty maps from a frozen feature/match database;
cache preparation and reconstruction timings are reported separately.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from .neon_sfm import ROOT, intervals, read_rows, write_json

FIELDS = ('run_id method condition recording_id frame_id source_timestamp '
          'timestamp_error_ns world_id map_id map_epoch pose_available pose_origin '
          'metric_status quality_status tx ty tz qx qy qz qw').split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pose(row):
    values = np.array([float(row[k]) for k in ('tx','ty','tz','qx','qy','qz','qw')])
    if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1)>1e-3:
        raise ValueError('Nonfinite translation or invalid unit quaternion')


def evaluate(rows, frames):
    """Union availability across worlds; continuity only within each world."""
    timestamps = {int(r['frame_id']):int(r['time_ns']) for r in frames}
    if len(timestamps)!=len(frames) or not frames:
        raise ValueError('Evaluation must have unique source frame IDs')
    worlds = {}; seen = set(); metric = set(); quality = {'pass':0,'fail':0,'not_evaluated':0}
    for r in rows:
        i=int(r['frame_id']);key=(r['run_id'],i,r['world_id'])
        if i not in timestamps or key in seen:
            raise ValueError('Unknown evaluation frame or duplicate frame/world pose')
        seen.add(key)
        if r['pose_available'] not in (True,'True','true','1',1):continue
        validate_pose(r)
        worlds.setdefault(r['world_id'],set()).add(i)
        if r['metric_status']=='imu_metric_candidate':metric.add(i)
        quality[r['quality_status']]+=1
    present=set().union(*worlds.values()) if worlds else set()
    spans=[dict(world_id=w,first_frame=a,last_frame=b,frames=b-a+1,
                duration_s=(timestamps[b]-timestamps[a])*1e-9)
           for w,ids in worlds.items() for a,b in intervals(ids)]
    longest=max(spans,key=lambda s:(s['frames'],s['duration_s']),default=None)
    return dict(evaluation_frames=len(frames),pose_rows=sum(len(v) for v in worlds.values()),
                registered_unique=len(present),output_rate=len(present)/len(frames),
                largest_single_world=max(map(len,worlds.values()),default=0),
                single_world_rate=max(map(len,worlds.values()),default=0)/len(frames),
                longest_contiguous=longest,independent_worlds=len(worlds),
                metric_candidate_unique=len(metric),metric_candidate_rate=len(metric)/len(frames),
                missing_intervals=intervals(set(timestamps)-present),continuous_intervals=spans,
                quality_counts=quality,accuracy='not evaluated; no reference trajectory')


def common_row(run, method, recording, frame, world, map_id, values, metric):
    r=dict(run_id=run,method=method,condition='O' if method=='ORB-SLAM3' else 'C0',
           recording_id=recording,frame_id=int(frame['frame_id']),source_timestamp=frame['time_ns'],
           timestamp_error_ns=0,world_id=world,map_id=map_id,map_epoch='final_surviving',
           pose_available=True,pose_origin='final_native_export',metric_status=metric,
           quality_status='not_evaluated')
    r.update({k:values[k] for k in ['tx','ty','tz','qx','qy','qz','qw']});validate_pose(r)
    return r


def collect_orb(target,frames,recording):
    by_time={float(r['slam_time_s']):r for r in frames};rows=[]
    if len(by_time)!=len(frames):raise ValueError('Ambiguous ORB source time')
    for p in read_rows(target/'slam/camera_trajectory.csv'):
        frame=by_time[float(p['slam_time_s'])];m=p['map_id']
        rows.append(common_row(target.name,'ORB-SLAM3',recording,frame,f'{target.name}:final_map:{m}',m,p,
                              'imu_metric_candidate' if p['metric']=='1' else 'arbitrary_scale'))
    states=read_rows(target/'slam/states.csv')
    return rows,dict(runtime_tracking_ok=sum(r['state']=='2' for r in states),
                     runtime_metric_ok=sum(r['state']=='2' and r['imu_initialized']=='1' for r in states),
                     processed_frames=len(states),final_epoch_note='Final exporter retains surviving history; do not combine earlier runtime epochs sharing a map ID')


def collect_colmap(target,frames,recording):
    lookup={int(r['frame_id']):r for r in frames};rows=[]
    folder=target/f'mapping_auto_{len(frames)}'
    for path in sorted(folder.glob('[0-9]*/visual_camera_to_world.csv')):
        m=path.parent.name
        for p in read_rows(path):
            frame=lookup[int(p['frame_id'])]
            if int(p['time_ns'])!=int(frame['time_ns']):raise ValueError('COLMAP timestamp mismatch')
            r=common_row(target.name,'SuperPoint+LightGlue+COLMAP',recording,frame,f'{target.name}:model:{m}',m,p,'arbitrary_scale')
            r.update(map_epoch='0',pose_origin='registered_image_inverse_cam_from_world');rows.append(r)
    return rows,dict(native_summary=json.loads((folder/'summary.json').read_text()))


def write_poses(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


def run_command(command,log):
    start=time.monotonic()
    with log.open('w') as f:process=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return dict(returncode=process.returncode,wall_seconds=time.monotonic()-start,command=command,log=str(log))


def summarize(out):
    rows=json.loads((out/'results.json').read_text());groups={}
    for row in rows:groups.setdefault((row['evaluation_set'],row['condition']),[]).append(row)
    summary=[]
    for (window,condition),runs in groups.items():
        ok=[r for r in runs if r['status']=='completed'];fields=['registered_unique','output_rate','largest_single_world','single_world_rate','independent_worlds']
        data=dict(evaluation_set=window,condition=condition,attempts=len(runs),completed=len(ok),failed=len(runs)-len(ok),runs=[r['run_id'] for r in runs])
        for field in fields:
            values=[r['metrics'][field] for r in ok];data[field]=dict(values=values,median=float(np.median(values)) if values else None,min=min(values) if values else None,max=max(values) if values else None)
        data['longest_contiguous_frames']=[r['metrics']['longest_contiguous']['frames'] if r['metrics']['longest_contiguous'] else 0 for r in ok]
        data['longest_contiguous_seconds']=[r['metrics']['longest_contiguous']['duration_s'] if r['metrics']['longest_contiguous'] else 0 for r in ok]
        summary.append(data)
    write_json(out/'summary.json',dict(groups=summary,accuracy='not measured',RIM='separate scan feasibility check; excluded from this local comparison'))
    return summary


def experiment(prepared,out,repeats):
    if out.exists():raise ValueError('Preserve previous experiments; output must be new')
    all_frames=read_rows(prepared/'frames.csv');manifest=json.loads((prepared/'manifest.json').read_text())
    recording=json.loads((Path(manifest['recording'])/'info.json').read_text())['recording_id']
    windows={name:[r for r in all_frames if a<=int(r['frame_id'])<=b] for name,a,b in [('Q-good',9988,10887),('Q-hard',25834,26733)]}
    assert all(len(v)==900 for v in windows.values())
    out.mkdir(parents=True)
    build=ROOT/'output/orbslam3_standalone/build';binary=build/'neon_standalone';library=build/'libstandalone_orb.so'
    assert 'NEON_ORPHAN_GUARD:BOOL=ON' in (build/'CMakeCache.txt').read_text()
    frozen={str(p):sha(p) for p in [binary,library,prepared/'camera.yaml',prepared/'manifest.json',prepared/'frames.csv']}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),repeats_per_condition=repeats,order='Per window O1,C1,O2,C2,...; windows good then hard',
              evaluation_and_estimation_frames={k:[int(r['frame_id']) for r in v] for k,v in windows.items()},
              initial_state='empty maps on every trial; no existing reconstruction',additional_images=False,
              orb=dict(pace=1,lost_keyframes='on',orphan_guard=True,descriptor_ties='native',diagnostics=False,scale_jacobian_correction=False),
              colmap=dict(pair_offsets=[1,2,3,5],seed_policy='auto',random_seed=0,feature_dtype='float32',neural_backend='CUDA',ba_backend='CERES_CPU',feature_match_cache='Frozen once per window, five independent reconstructions with separate DB copies',intrinsics_fixed=True),
              hashes=frozen,manifest=manifest,metric_accuracy='not measured; no independent reference',
              timing='Existing image/IMU preprocessing is cached, its original wall time is unavailable. Fresh GPU feature/match/verification preparation timed separately. Trial clocks include ORB subset preparation/export or COLMAP DB copy/mapping/export. Not five cold end-to-end repetitions.',
              prior_result='883/900 historical hard baseline is not a trial here; rejected boundary connections excluded')
    write_json(out/'protocol.json',plan);results=[];write_json(out/'results.json',results)
    for window,frames in windows.items():
        start=int(frames[0]['frame_id']);cache=out/f'{window}_cache';costs=[]
        for stage in ['extract','match','database']:
            command=[sys.executable,str(ROOT/'standalone/orbslam3/neon_sfm.py'),'--prepared',str(prepared),'--output',str(cache),'--start_n',str(start),'--max_n','900','--stage',stage]
            if cache.exists():command+=['--resume']
            print(f'PREPARE {window} {stage}',flush=True)
            cost=run_command(command,out/f'{window}_{stage}.log');costs.append(dict(stage=stage,**cost));write_json(out/f'{window}_preparation_timing.json',costs)
            if cost['returncode']:raise RuntimeError(f'Cache {stage} failed; inspect {cost["log"]}')
        cache_sha=sha(cache/'database.db')
        write_json(out/f'{window}_cache_identity.json',dict(database_sha256=cache_sha,feature_done=json.loads((cache/'features_done.json').read_text()),match_done=json.loads((cache/'matches_done.json').read_text()),database_done=json.loads((cache/'database_done.json').read_text())))
        with (out/f'{window}_frames.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['recording_id','frame_id','timestamp_ns','evaluation_set']);writer.writerows((recording,r['frame_id'],r['time_ns'],window) for r in frames)
        for i in range(1,repeats+1):
            for condition in ['O','C0']:
                for path,digest in frozen.items():
                    if sha(Path(path))!=digest:raise RuntimeError('Frozen input/binary changed during experiment')
                name=f'{window}_{condition}_{i:02}';target=out/name;print('START',name,flush=True);wall=time.monotonic()
                if condition=='O':
                    command=[str(ROOT/'output/orbslam3_standalone/.venv/bin/python'),str(ROOT/'standalone/orbslam3/run.py'),'--prepared',str(prepared),'--output',str(target),'--start_n',str(start),'--max_n','900','--pace','1','--lost-keyframes','on']
                else:
                    target.mkdir();shutil.copy2(cache/'database.db',target/'database.db');(target/'images').symlink_to((cache/'images').resolve(),target_is_directory=True);shutil.copy2(cache/'identity.json',target/'identity.json')
                    command=[sys.executable,str(ROOT/'standalone/orbslam3/neon_sfm.py'),'--prepared',str(prepared),'--output',str(target),'--start_n',str(start),'--max_n','900','--stage','reconstruct','--resume']
                execution=run_command(command,out/f'{name}.log');execution['wall_seconds_including_copy']=time.monotonic()-wall
                row=dict(run_id=name,evaluation_set=window,condition=condition,status='completed' if execution['returncode']==0 else 'failed',execution=execution)
                if execution['returncode']==0:
                    poses,extra=collect_orb(target,frames,recording) if condition=='O' else collect_colmap(target,frames,recording)
                    row['metrics']=evaluate(poses,frames);row['diagnostics']=extra;write_poses(target/'common_poses.csv',poses);write_json(target/'comparison_metrics.json',row)
                results.append(row);write_json(out/'results.json',results);summarize(out)
                if sha(cache/'database.db')!=cache_sha:raise RuntimeError('Frozen cache database changed')
                print('DONE',name,row.get('metrics',{}).get('registered_unique'),'status',row['status'],flush=True)
    return summarize(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    experiment(args.prepared.resolve(),args.output.resolve(),args.repeats)

if __name__=='__main__':main()
