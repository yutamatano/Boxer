#!/usr/bin/env python3
"""Replay upstream MapPoint descriptor medoid selection without SLAM threads.

The upstream rule includes self distance, takes floor((N-1)/2), and keeps the
first strict minimum. Snapshot observation order is the native std::map pointer
order. Canonical source/feature ordering is a separate diagnostic condition,
never a change to the live estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .benchmark_neon_correspondences import read_features
except ImportError:
    from benchmark_neon_correspondences import read_features


def select_descriptor(descriptors):
    descriptors=np.asarray(descriptors,np.uint8)
    if descriptors.ndim!=2 or descriptors.shape[1]!=32 or not len(descriptors):
        raise ValueError('Expected at least one ORB descriptor of 32 bytes')
    distances=np.unpackbits(descriptors[:,None,:]^descriptors[None,:,:],axis=2).sum(2)
    medians=np.sort(distances,axis=1)[:,(len(descriptors)-1)//2]
    index=int(np.argmin(medians))
    return index,int(np.count_nonzero(medians==medians[index]))


def replay(snapshot,features):
    rows=[]
    for point in snapshot['points']:
        obs=[(int(o['source_frame_id']),int(o['feature'])) for o in point['observations'] if int(o['feature'])>=0]
        if any(a not in features for a,b in obs):raise ValueError('Missing observed keyframe features')
        descriptors=np.array([features[a]['descriptor'][b] for a,b in obs],np.uint8)
        index,ties=select_descriptor(descriptors)
        order=sorted(range(len(obs)),key=lambda i:obs[i]);canonical,_=select_descriptor(descriptors[order])
        actual=point['matching_state']['descriptor_hex']
        native=descriptors[index].tobytes().hex();stable=descriptors[order[canonical]].tobytes().hex()
        rows.append(dict(point_id=int(point['point_id']),observations=obs,native_descriptor=native,actual_descriptor=actual,
                         native_matches=actual==native,chosen_observation=obs[index],tied_minima=ties,
                         canonical_descriptor=stable,canonical_differs=stable!=native))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('snapshot',type=Path)
    parser.add_argument('--features',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();features={}
    for path in args.features:
        data=read_features(path)
        if features.keys()&data.keys():raise ValueError('Duplicate feature sources')
        features.update(data)
    result=replay(json.loads(args.snapshot.read_text()),features)
    args.output.write_text(json.dumps(dict(points=result,matched=sum(r['native_matches'] for r in result),total=len(result),ties=sum(r['tied_minima']>1 for r in result)),indent=2)+'\n')

if __name__=='__main__':main()
