#!/usr/bin/env python3
"""Summarize a fixed alternating tie-policy experiment, including censoring."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .compare_visual_maps import sim3
from .neon_sfm import write_json


def read_points(path):
    with path.open() as f:return {r['observations']:r for r in csv.DictReader(f)}


def summarize(root):
    runs=json.loads((root/'results.json').read_text());output={};details=[]
    for row in runs:
        folder=root/row['name'];events=[json.loads(x) for x in (folder/'slam/map_events.jsonl').read_text().splitlines()];initial=[x for x in events if x['event']=='IMU_INITIALIZED'];resets=[x for x in events if x['event']=='RESET_APPLIED']
        first=initial[0] if initial else None;followup=10052-first['source_frame_id'] if first else None
        details.append(dict(name=row['name'],condition=row['condition'],any_reset=bool(resets),resets=len(resets),initialization_source=first['source_frame_id'] if first else None,post_initialization_followup_frames=followup,right_censored_20_frames=followup is None or followup<20,usable_frames=row['pipeline']['usable_frames'],coverage=row['pipeline']['coverage']))
    for condition in ['A','B']:
        selected=[r for r in details if r['condition']==condition];pool=defaultdict(set);presence=Counter();topologies=set();shape=[];available=0;reference=None
        for row in selected:
            folder=root/row['name']/'slam';file=folder/'light_10004_ProcessNewKeyFrame_after_points.csv'
            if file.exists():
                pts=read_points(file)
                for key,p in pts.items():pool[key].add(p['descriptor_hash']);presence[key]+=1
            file=folder/'light_10004_SearchInNeighbors_after_points.csv'
            if file.exists():
                pts=read_points(file);available+=1;topologies.add(hashlib.sha256('\n'.join(sorted(pts)).encode()).hexdigest())
                if reference is None:reference=pts
                keys=sorted(reference.keys()&pts.keys())
                if len(keys)>=30:
                    a=np.array([[reference[k][c] for c in ['x','y','z']] for k in keys],float);b=np.array([[pts[k][c] for c in ['x','y','z']] for k in keys],float);fit=sim3(a,b);radius=np.sqrt(np.mean(np.sum((b-b.mean(0))**2,axis=1)));shape.append(fit['rms']/radius)
        repeated=[k for k in pool if presence[k]>=2]
        output[condition]=dict(runs=len(selected),any_reset_runs=sum(r['any_reset'] for r in selected),right_censored_runs=sum(r['right_censored_20_frames'] for r in selected),usable_frames=[r['usable_frames'] for r in selected],usable_frames_median=float(np.median([r['usable_frames'] for r in selected])),initialization_source=[r['initialization_source'] for r in selected],repeated_descriptor_observation_sets=len(repeated),descriptor_sets_with_multiple_choices=sum(len(pool[k])>1 for k in repeated),fusion_available_runs=available,distinct_fusion_topologies=len(topologies),fusion_point_relative_rms_vs_first=shape)
    write_json(root/'analysis.json',dict(summary=output,runs=details,interpretation='Any-reset counts include pre-initialization failures. The older after-first-estimate early_reset field alone misses these and is censored by the 65-frame window. Metric flags are not metric accuracy.'))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args();print(json.dumps(summarize(args.root),indent=2))
if __name__=='__main__':main()
