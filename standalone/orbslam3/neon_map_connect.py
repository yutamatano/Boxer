#!/usr/bin/env python3
"""Audit pairwise SfM overlap with predeclared correspondence and Sim3 gates."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from .compare_visual_maps import sim3,match_points
from .neon_sfm import setup,write_json,ROOT


def source(im):return int(im.name[6:-4])

def points(model):
    result=[]
    for pid,p in sorted(model.points3D.items()):
        result.append(dict(point_id=pid,xyz=p.xyz.tolist(),observations=[dict(source_frame_id=source(model.images[e.image_id]),feature=e.point2D_idx) for e in p.track.elements]))
    return result


def robust_fit(a,b):
    radius=float(np.sqrt(np.mean(np.sum((b-b.mean(0))**2,axis=1))));threshold=.01*radius
    rng=np.random.default_rng(0);best=np.zeros(len(a),bool)
    for _ in range(300):
        idx=rng.choice(len(a),4,replace=False)
        try:f=sim3(a[idx],b[idx])
        except ValueError:continue
        R=np.array(f['rotation']);t=np.array(f['translation']);good=np.linalg.norm(f['scale']*a@R.T+t-b,axis=1)<threshold
        if good.sum()>best.sum():best=good
    if best.sum()<10:return None
    fit=sim3(a[best],b[best]);fit.update(inlier_count=int(best.sum()),training_points=len(a),threshold=threshold,target_radius=radius)
    return fit


def audit(models,out,timestamps=None):
    ps={i:points(m) for i,m in models.items()};images={i:{source(im):im for im in m.images.values() if im.has_pose} for i,m in models.items()};rows=[]
    for a,b in combinations(sorted(models),2):
        shared=sorted(images[a].keys()&images[b].keys());pairs=match_points(ps[a],ps[b])
        row=dict(a=a,b=b,shared_frames=shared,shared_frame_count=len(shared),common_points=len(pairs),time_span_s=((timestamps[shared[-1]]-timestamps[shared[0]])*1e-9 if timestamps else (shared[-1]-shared[0])*.1) if shared else 0,accepted=False)
        if shared:
            for name,i in [('a',a),('b',b)]:
                C=np.array([images[i][n].cam_from_world().inverse().translation for n in shared]);row['centres_'+name]=dict(rank=int(np.linalg.matrix_rank(C-C.mean(0))),singular_values=np.linalg.svd(C-C.mean(0),compute_uv=False).tolist(),extent=np.ptp(C,axis=0).tolist())
                first=images[i][shared[0]].cam_from_world().rotation.matrix()
                row['rotation_spread_'+name+'_deg']=max(float(np.degrees(np.arccos(np.clip((np.trace(images[i][n].cam_from_world().rotation.matrix()@first.T)-1)/2,-1,1)))) for n in shared)
        if len(pairs)>=30 and len(shared)>=3:
            X=np.array([ps[a][i]['xyz'] for i,j in pairs]);Y=np.array([ps[b][j]['xyz'] for i,j in pairs]);row['point_spread_a']=np.linalg.svd(X-X.mean(0),compute_uv=False).tolist();row['point_spread_b']=np.linalg.svd(Y-Y.mean(0),compute_uv=False).tolist();order=np.random.default_rng(0).permutation(len(pairs));test=order[::5];train=np.setdiff1d(order,test);fit=robust_fit(X[train],Y[train])
            if fit:
                R=np.array(fit['rotation']);t=np.array(fit['translation']);s=fit['scale'];error=np.linalg.norm(s*X[test]@R.T+t-Y[test],axis=1)
                angles=[];cams=[]
                for n in shared:
                    A=images[a][n].cam_from_world();B=images[b][n].cam_from_world();d=B.rotation.matrix()@R@A.rotation.matrix().T
                    angles.append(float(np.degrees(np.arccos(np.clip((np.trace(d)-1)/2,-1,1)))))
                    cams.append(float(np.linalg.norm(s*A.inverse().translation@R.T+t-B.inverse().translation)))
                row.update(transform=fit,heldout_points=len(test),heldout_p95=float(np.percentile(error,95)),heldout_relative_p95=float(np.percentile(error,95)/fit['target_radius']),camera_rotation_p95_deg=float(np.percentile(angles,95)),camera_centre_p95=float(np.percentile(cams,95)))
                row['accepted']=bool(row['heldout_relative_p95']<=.02 and row['camera_rotation_p95_deg']<=2 and fit['inlier_count']/fit['training_points']>=.8)
        rows.append(row)
    write_json(out/'pairwise.json',dict(protocol=dict(common_frames_min=3,common_points_min=30,training_inlier_radius_fraction=.01,heldout_fraction=.2,heldout_p95_radius_fraction_max=.02,rotation_p95_deg_max=2,training_inlier_fraction_min=.8),pairs=rows));return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();setup(ROOT/'output/neon_sfm_runtime')
    import pycolmap
    args.output.mkdir(exist_ok=False,parents=True)
    models={int(d.name):pycolmap.Reconstruction(str(d)) for d in args.baseline.iterdir() if d.is_dir() and d.name.isdigit()}
    identity=json.loads((args.baseline.parent/'identity.json').read_text());timestamps={r['frame_id']:r['time_ns'] for r in identity['frames']}
    rows=audit(models,args.output,timestamps)
    for r in rows:
        if r['shared_frame_count']:print(r['a'],r['b'],r['shared_frame_count'],r['common_points'],r['accepted'],r.get('heldout_relative_p95'),flush=True)

if __name__=='__main__':main()
