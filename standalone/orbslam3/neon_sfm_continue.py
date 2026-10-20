#!/usr/bin/env python3
"""Preserve an existing SfM database, add boundary pairs, and continue a map."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .neon_sfm import ROOT,setup,write_json,read_rows,intervals,reconstruction_report


def prepare(baseline,out):
    out.mkdir(parents=True,exist_ok=False)
    identity=json.loads((baseline/'identity.json').read_text());write_json(out/'identity.json',identity)
    (out/'features').symlink_to((baseline/'features').resolve(),target_is_directory=True)
    (out/'matches').mkdir()
    for path in (baseline/'matches').glob('*.npz'):(out/'matches'/path.name).symlink_to(path.resolve())
    shutil.copy2(baseline/'database.db',out/'database.db')
    write_json(out/'database_seed.json',dict(source=str(baseline.resolve()),sha256=hashlib.sha256((baseline/'database.db').read_bytes()).hexdigest()))
    ids={r['frame_id'] for r in identity['frames']}
    prior=baseline/'pair_manifest.json'
    base={tuple(pair) for pair in json.loads(prior.read_text())['pairs']} if prior.exists() else {(a,a+d) for a in ids for d in (1,2,3,5) if a+d in ids}
    extra=set()
    for left,right in [(26169,26176),(26441,26447)]:
        for a in range(left-10,left+1):
            for d in [6,7,8,10]:
                b=a+d
                if right<=b<=right+10 and a in ids and b in ids:extra.add((a,b))
    write_json(out/'pair_manifest.json',dict(schema=1,pairs=sorted(base|extra),boundaries=[[26169,26176],[26441,26447]],added_pairs=sorted(extra-base)))
    return len(extra-base)


def observation_errors(model):
    result={}
    for point in model.points3D.values():
        signature=frozenset((int(model.images[e.image_id].name[6:-4]),e.point2D_idx) for e in point.track.elements)
        for e in point.track.elements:
            im=model.images[e.image_id];q=im.cam_from_world()*point.xyz
            error=float(np.linalg.norm(im.camera.img_from_cam(q)-im.points2D[e.point2D_idx].xy)) if q[2]>0 else None
            result[(int(im.name[6:-4]),e.point2D_idx)]=(error,signature)
    return result


def compare_quality(before,after):
    a=observation_errors(before);b=observation_errors(after);common=a.keys()&b.keys();new=b.keys()-a.keys();lost=a.keys()-b.keys()
    def stats(values):
        data=[v for v in values if v is not None]
        return dict(count=len(values),positive=len(data),negative=len(values)-len(data),p50=float(np.percentile(data,50)) if data else None,p95=float(np.percentile(data,95)) if data else None)
    return dict(original_total=len(a),retained=len(common),removed=len(lost),track_membership_changed=sum(a[k][1]!=b[k][1] for k in common),old_before=stats([a[k][0] for k in common]),old_after=stats([b[k][0] for k in common]),added=stats([b[k][0] for k in new]))


def continuation(database,images,seed,out,frames):
    import pycolmap as p
    out.mkdir(exist_ok=False,parents=True)
    options=p.IncrementalPipelineOptions();options.image_names=[f"frame_{int(r['frame_id']):08}.png" for r in frames]
    options.multiple_models=False;options.num_threads=4;options.random_seed=0;options.mapper.random_seed=0
    options.ba_refine_focal_length=options.ba_refine_principal_point=options.ba_refine_extra_params=False;options.ba_refine_sensor_from_rig=False
    options.mapper.abs_pose_refine_focal_length=options.mapper.abs_pose_refine_extra_params=False;options.ba_use_gpu=False;options.mapper.init_min_tri_angle=1
    source_sha=hashlib.sha256(database.read_bytes()).hexdigest();working=out/'input_database.db';shutil.copy2(database,working)
    (out/'options.txt').write_text(options.summary());models=p.incremental_mapping(str(working),str(images),str(out),options=options,input_path=str(seed))
    if hashlib.sha256(database.read_bytes()).hexdigest()!=source_sha:raise RuntimeError('Source database changed during continuation')
    baseline=p.Reconstruction(str(seed));reports=[]
    for i,model in models.items():
        folder=out/str(i);folder.mkdir(exist_ok=True);model.write(str(folder));report=reconstruction_report(model,frames,folder);report['comparison_to_seed']=compare_quality(baseline,model);report['model_id']=i;reports.append(report)
    write_json(out/'summary.json',dict(models=reports,metric=False,seed=str(seed),database_sha256=source_sha,working_database_sha256=hashlib.sha256(working.read_bytes()).hexdigest(),source_database_unchanged=True,largest_single_map=max((r['registered'] for r in reports),default=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['prepare','continue'],required=True);parser.add_argument('--database',type=Path);parser.add_argument('--images',type=Path)
    args=parser.parse_args();setup(ROOT/'output/neon_sfm_runtime')
    if args.stage=='prepare':print('added pairs',prepare(args.baseline,args.output));return
    identity=json.loads((args.baseline/'identity.json').read_text());frames=read_rows(Path(identity['prepared'])/'frames.csv');ids={r['frame_id'] for r in identity['frames']};frames=[r for r in frames if int(r['frame_id']) in ids]
    continuation(args.database or args.baseline/'database.db',args.images or args.baseline/'images',args.baseline/'mapping_auto_900/8',args.output,frames)

if __name__=='__main__':main()
