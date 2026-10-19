#!/usr/bin/env python3
"""Fixed-pair native ORB / deduplicated multiscale ORB / forward-backward KLT audit.

This is an independent correspondence test, not a metric SLAM trajectory estimator.
No learned inference is performed. OpenCV's feature/geometry routines use CPU.
"""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import cv2
import numpy as np
from scipy.spatial import cKDTree

DTYPE = np.dtype([('xy','<f4',(2,)), ('angle','<f4'), ('octave','<i4'),
                  ('descriptor','u1',(32,))])


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def read_features(path):
    features = {}
    with Path(path).open('rb') as f:
        while header := f.read(12):
            if len(header) != 12:
                raise ValueError('Truncated feature header')
            i, n = struct.unpack('<qI', header)
            payload = f.read(n*48)
            if len(payload) != n*48 or i in features:
                raise ValueError('Invalid feature payload')
            features[i] = np.frombuffer(payload, dtype=DTYPE).copy()
    return features


def deduplicate(features):
    """Keep the finest observation within 3*scale pixels of an existing point."""
    if not len(features):
        return features
    tree = cKDTree(features['xy'])
    scales=1.2**features["octave"]
    blocked = np.zeros(len(features), bool)
    keep = []
    for i in np.argsort(features['octave'], kind='stable'):
        if blocked[i]:
            continue
        keep.append(i)
        nearby=np.asarray(tree.query_ball_point(features['xy'][i],3*scales.max()),int)
        distances=np.linalg.norm(features['xy'][nearby]-features['xy'][i],axis=1)
        blocked[nearby[distances<=3*np.maximum(scales[i],scales[nearby])]]=True
    return features[keep]


def multiscale(a, b):
    if min(len(a), len(b)) < 2:
        return np.empty((0,2), int)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(np.ascontiguousarray(a['descriptor']),
                             np.ascontiguousarray(b['descriptor']), k=2)
    reverse = matcher.match(np.ascontiguousarray(b['descriptor']),
                            np.ascontiguousarray(a['descriptor']))
    accepted = [(m.queryIdx, m.trainIdx) for pair in pairs if len(pair)==2
                for m,n in [pair] if m.distance <= 50 and m.distance < .9*n.distance
                and reverse[m.trainIdx].trainIdx == m.queryIdx]
    return np.asarray(accepted, int).reshape(-1,2)


def flow(a, b, points):
    if len(points)==0:
        return np.empty((0,2), np.float32), np.empty(0, int)
    p = np.asarray(points, np.float32).reshape(-1,1,2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3,
                                      criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    if q is None:
        return np.empty((0,2), np.float32), np.empty(0, int)
    back, rev, _ = cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    h,w = b.shape
    good = (ok.ravel()!=0)&(rev.ravel()!=0)&np.isfinite(q.reshape(-1,2)).all(axis=1)
    good &= np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)<=1.
    good &= (q[:,0,0]>=0)&(q[:,0,0]<w)&(q[:,0,1]>=0)&(q[:,0,1]<h)
    idx = np.flatnonzero(good)
    return q[idx,0],idx


def geometry(a, b, sigma_a, sigma_b, K, shape):
    """Identical robust geometry gates for all methods, with octave uncertainty.

    A conservative 1.5px RANSAC threshold is shared by every method. Octave
    uncertainty affects triangulated reprojection checks, not candidate counts.
    """
    empty = dict(matches=len(a), essential_inliers=0, pose_inliers=0, triangulated=0,
                 grid_cells=0, parallax_deg=0., median_reprojection_px=None)
    if len(a)<8:
        return empty, np.empty(0,int)
    cv2.setRNGSeed(0)
    E, mask = cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=.999,threshold=1.5)
    if E is None or E.shape!=(3,3):
        return empty,np.empty(0,int)
    empty['essential_inliers']=int(mask.sum())
    _, R, t, posemask = cv2.recoverPose(E,a,b,K,mask=mask.copy())
    idx = np.flatnonzero(posemask.ravel())
    empty['pose_inliers']=len(idx)
    if not len(idx):
        return empty,idx
    p = cv2.undistortPoints(a[idx].reshape(-1,1,2),K,None).reshape(-1,2)
    q = cv2.undistortPoints(b[idx].reshape(-1,1,2),K,None).reshape(-1,2)
    Xh = cv2.triangulatePoints(np.column_stack((np.eye(3),np.zeros(3))),
                               np.column_stack((R,t)),p.T,q.T)
    with np.errstate(divide='ignore',invalid='ignore'):
        X = Xh[:3]/Xh[3]
        Y = R@X+t
        proj1 = K@X;proj1=(proj1[:2]/proj1[2]).T
        proj2 = K@Y;proj2=(proj2[:2]/proj2[2]).T
        e1=np.linalg.norm(proj1-a[idx],axis=1);e2=np.linalg.norm(proj2-b[idx],axis=1)
        center=(-R.T@t).ravel()
        ray2=X.T-center
        cos=(X.T*ray2).sum(axis=1)/(np.linalg.norm(X,axis=0)*np.linalg.norm(ray2,axis=1))
        angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
    valid=np.isfinite(X).all(axis=0)&(X[2]>0)&(Y[2]>0)&(e1<=3*sigma_a[idx])&(e2<=3*sigma_b[idx])&(angle>=1.)
    valid_idx=idx[valid]
    empty['triangulated']=len(valid_idx)
    if len(valid_idx):
        h,w=shape
        cells=np.minimum((a[valid_idx]/[w/6,h/4]).astype(int),[5,3])
        empty['grid_cells']=len(np.unique(cells,axis=0))
        empty['parallax_deg']=float(np.median(angle[valid]))
        empty['median_reprojection_px']=float(np.median(np.maximum(e1[valid],e2[valid])))
    return empty,valid_idx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True,help='Existing branch-audit directory with matching_replay.csv')
    p.add_argument('--prepared',type=Path,required=True,help='Full prepared input containing good/hard windows')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native',type=Path,default=Path(__file__).resolve().parents[2]/
                   'output/orbslam3_standalone/build_diagnostics/correspondence_native')
    p.add_argument('--reuse-native',action='store_true',help='Reuse this output directory native binary results after identity checks')
    p.add_argument('--controls',type=int,default=100)
    args=p.parse_args();cv2.setNumThreads(2)
    if args.output.exists() and not args.reuse_native:
        p.error('Output exists; use a new directory or --reuse-native')
    args.output.mkdir(parents=True,exist_ok=args.reuse_native)
    fixed=rows(args.audit/'matching_replay.csv')
    if len(fixed)!=370:
        raise ValueError('Expected the fixed 370 failed pairs')
    frames={int(r['frame_id']):r for r in rows(args.prepared/'frames.csv')}
    good_states=rows(args.audit/'pace1_1/slam/states.csv')
    good={int(r['frame_id']) for r in good_states if r['state']=='2' and r['imu_initialized']=='1'}
    candidates=sorted(i for i in good if all(i+d in good for d in range(7)))
    control_ids=[candidates[i] for i in np.linspace(0,len(candidates)-1,args.controls,dtype=int)]
    cases=[]
    for group, refs in [('hard',[int(r['reference_frame_id']) for r in fixed]),('good',control_ids)]:
        for ref in refs:
            for delta in (1,2,3,5):
                target=ref+delta
                if target+1 not in frames:
                    raise ValueError('Missing third frame')
                cases.append(dict(group=group,reference=ref,target=target,third=target+1,delta=delta))
    pairs=sorted({(c['reference'],c['target']) for c in cases}|
                 {(c['target'],c['third']) for c in cases})
    ids=sorted({i for pair in pairs for i in pair})
    native_hash=hashlib.sha256(args.native.read_bytes()).hexdigest()
    identity=dict(native_sha256=native_hash,library_sha256=hashlib.sha256((args.native.parent/'libstandalone_orb.so').read_bytes()).hexdigest(),
                  fixed_pairs_sha256=hashlib.sha256((args.audit/'matching_replay.csv').read_bytes()).hexdigest(),pairs=pairs,controls=control_ids,opencv_python=cv2.__version__)
    if args.reuse_native:
        if json.loads((args.output/'identity.json').read_text())!=json.loads(json.dumps(identity)):
            raise ValueError('Native reuse identity mismatch')
    else:
        (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        with (args.output/'images.csv').open('w') as f:
            for i in ids:
                f.write(f"{i},{(args.prepared/frames[i]['image']).resolve()}\n")
        (args.output/'pairs.csv').write_text(''.join(f'{a},{b}\n' for a,b in pairs))
        env=os.environ.copy();env['LD_LIBRARY_PATH']=str(args.native.resolve().parent.parent/'prefix/lib')+':'+env.get('LD_LIBRARY_PATH','')
        subprocess.run([str(args.native.resolve()),str((args.prepared/'camera.yaml').resolve()),
                        str((args.output/'images.csv').resolve()),str((args.output/'pairs.csv').resolve()),
                        str((args.output/'features.bin').resolve()),str((args.output/'native_matches.csv').resolve())],env=env,check=True)
    features=read_features(args.output/'features.bin')
    matches=defaultdict(list);counts={}
    for r in rows(args.output/'native_matches.csv'):
        key=(int(r['reference']),int(r['target']));counts[key]=int(r['native_count'])
        if int(r['index_reference'])>=0:
            matches[key].append((int(r['index_reference']),int(r['index_target'])))
    for r in fixed:
        key=int(r['reference_frame_id']),int(r['source_frame_id'])
        if counts[key]!=int(r['native_matches']):
            raise ValueError(f'Native matcher replay mismatch: {key}: {counts[key]} != {r["native_matches"]}')
    fs=cv2.FileStorage(str(args.prepared/'camera.yaml'),cv2.FILE_STORAGE_READ)
    K=np.array([[fs.getNode('Camera1.fx').real(),0,fs.getNode('Camera1.cx').real()],
                [0,fs.getNode('Camera1.fy').real(),fs.getNode('Camera1.cy').real()],[0,0,1.]])
    @lru_cache(maxsize=32)
    def image(i):
        im=cv2.imread(str(args.prepared/frames[i]['image']),cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError('Unreadable benchmark image')
        return im
    dedup={i:deduplicate(features[i]) for i in ids}
    @lru_cache(maxsize=64)
    def multi(a,b,level0=False):
        fa,fb=dedup[a],dedup[b]
        if level0:fa,fb=fa[fa['octave']==0],fb[fb['octave']==0]
        return multiscale(fa,fb)
    result=[]
    for j,c in enumerate(cases):
        a,b,t=c['reference'],c['target'],c['third'];ia,ib,it=image(a),image(b),image(t)
        for method in ('native_orb','level0_orb_control','multiscale_orb','klt'):
            if method=='klt':
                pts=cv2.goodFeaturesToTrack(ia,maxCorners=2000,qualityLevel=.01,minDistance=5,blockSize=7)
                pts=np.empty((0,2),np.float32) if pts is None else pts.reshape(-1,2)
                qb,keep=flow(ia,ib,pts);pa=pts[keep];pb=qb
                _,continued=flow(ib,it,pb)
                sa=sb=np.ones(len(pa))
            else:
                fa,fb=(features[a],features[b]) if method=='native_orb' else (dedup[a],dedup[b])
                control=method=='level0_orb_control'
                if control:fa,fb=fa[fa['octave']==0],fb[fb['octave']==0]
                ij=np.asarray(matches[a,b],int).reshape(-1,2) if method=='native_orb' else multi(a,b,control)
                jk=np.asarray(matches[b,t],int).reshape(-1,2) if method=='native_orb' else multi(b,t,control)
                pa=fa['xy'][ij[:,0]];pb=fb['xy'][ij[:,1]]
                sa=1.2**fa['octave'][ij[:,0]];sb=1.2**fb['octave'][ij[:,1]]
                continued=np.flatnonzero(np.isin(ij[:,1],jk[:,0]))
            stats,valid=geometry(pa,pb,sa,sb,K,ia.shape)
            stats['continued_three_frames']=int(np.intersect1d(valid,continued).size)
            stats['common_gate_pass']=bool(stats['essential_inliers']>=100 and stats['triangulated']>=50 and stats['grid_cells']>=6 and stats['continued_three_frames']>=50)
            result.append(dict(**c,method=method,**stats))
        if (j+1)%50==0:print(f'Geometry pairs {j+1}/{len(cases)}',flush=True)
    with (args.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary=[]
    for group in ('hard','good'):
        for delta in (1,2,3,5):
            for method in ('native_orb','level0_orb_control','multiscale_orb','klt'):
                rs=[r for r in result if (r['group'],r['delta'],r['method'])==(group,delta,method)]
                summary.append(dict(group=group,delta=delta,method=method,pairs=len(rs),passes=sum(r['common_gate_pass'] for r in rs),
                                    matches_median=float(np.median([r['matches'] for r in rs])),
                                    triangulated_median=float(np.median([r['triangulated'] for r in rs])),
                                    continued_median=float(np.median([r['continued_three_frames'] for r in rs]))))
    report=dict(native_failed_pair_counts_verified=370,summary=summary,
                matcher_ablation='level0_orb_control uses the same deduplication/global mutual ratio matcher as multiscale_orb, restricted to octave 0. Only their contrast isolates extra octave eligibility; comparison with native_orb also changes search/assignment/orientation filtering.',
                limitation='Independent relative-geometry proxy, not trajectory coverage or absolute accuracy. Essential-only pose can remain ambiguous on planar/rotational scenes. Temporal continuation is a correspondence check, not a full three-view bundle adjustment.',
                thresholds=dict(ransac_px=1.5,parallax_deg=1,reprojection_sigma=3,essential_inliers=100,triangulated=50,grid_cells=6,continued=50),
                uncertainty='Multiscale coordinates stay in original 800x600 pixels. NMS prefers fine octaves, pairwise radius 3*max(1.2^octave). Reprojection gates use octave sigma; robust essential fit uses the same conservative pixel threshold for every method.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
