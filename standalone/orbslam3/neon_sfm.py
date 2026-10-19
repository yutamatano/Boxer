#!/usr/bin/env python3
"""Independent, arbitrary-scale Neon SfM from CUDA SuperPoint + LightGlue.

No metric Boxer pipeline is emitted. Camera intrinsics are fixed. The pose
export preserves original int64 timestamps and inverts COLMAP camera-from-world.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LIGHTGLUE_REVISION='eb42fee2d71449efb0aa5c10549752b5d75384d8'


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def read_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def intervals(ids):
    result=[]
    for i in sorted(set(ids)):
        if not result or i!=result[-1][1]+1:result.append([i,i])
        else:result[-1][1]=i
    return result


def setup(runtime):
    sys.path[:0]=[str(runtime/'deps'),str(ROOT/'output/neon_lightglue_runtime/deps'),str(ROOT/'output/neon_lightglue_runtime/source')]
    os.environ['MPLCONFIGDIR']=str(runtime/'mpl')


def camera_cv(prepared):
    fs=cv2.FileStorage(str(prepared/'camera.yaml'),cv2.FILE_STORAGE_READ)
    K=np.array([[fs.getNode('Camera1.fx').real(),0,fs.getNode('Camera1.cx').real()],
                [0,fs.getNode('Camera1.fy').real(),fs.getNode('Camera1.cy').real()],[0,0,1.]])
    return K,int(fs.getNode('Camera.width').real()),int(fs.getNode('Camera.height').real())


def load_feature(out,i):
    with np.load(out/'features'/f'{i:08}.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    if any(v.dtype!=np.float32 for v in data.values()):raise ValueError('Learned features must be Float32')
    return data


def load_match(out,a,b):
    reverse=a>b;a,b=sorted((a,b))
    with np.load(out/'matches'/f'{a:08}_{b:08}.npz',allow_pickle=False) as f:m=f['matches'].astype(np.int64)
    return m[:,::-1] if reverse else m


def gpu_models():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU inference fallback')
    source=ROOT/'output/neon_lightglue_runtime/source'
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=LIGHTGLUE_REVISION:raise ValueError('Unexpected LightGlue source')
    torch.hub.set_dir(str(source.parent/'torch_hub'));torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from lightglue import LightGlue,SuperPoint
    return torch,SuperPoint(max_num_keypoints=2048).eval().cuda().float(),LightGlue(features='superpoint',flash=False,mp=False).eval().cuda().float()


def extract(out,frames,prepared):
    torch,extractor,_=gpu_models();(out/'features').mkdir(exist_ok=True)
    with torch.inference_mode():
        for n,r in enumerate(frames,1):
            i=int(r['frame_id']);path=out/'features'/f'{i:08}.npz'
            if path.exists():load_feature(out,i);continue
            image=cv2.imread(str(prepared/r['image']),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError('Missing prepared image')
            data=extractor.extract(torch.from_numpy(image.astype(np.float32)/255.)[None].cuda(),resize=None)
            if data['descriptors'].dtype!=torch.float32 or data['descriptors'].device.type!='cuda':raise RuntimeError('Feature dtype/device mismatch')
            arrays={k:v[0].detach().cpu().numpy().astype(np.float32) for k,v in data.items()}
            np.savez_compressed(path,**arrays)
            if n%25==0 or n==len(frames):print(f'[features] {n}/{len(frames)} ({100*n/len(frames):.1f}%)',flush=True)
    write_json(out/'features_done.json',dict(frames=len(frames),device='cuda',dtype='float32',gpu=torch.cuda.get_device_name(0),source_revision=LIGHTGLUE_REVISION))


def match(out,frames):
    torch,_,matcher=gpu_models();(out/'matches').mkdir(exist_ok=True)
    ids={int(r['frame_id']) for r in frames};pairs=[(a,a+d) for a in sorted(ids) for d in (1,2,3,5) if a+d in ids]
    @lru_cache(maxsize=16)
    def feature(i):return {k:torch.from_numpy(v)[None].cuda() for k,v in load_feature(out,i).items()}
    with torch.inference_mode():
        for n,(a,b) in enumerate(pairs,1):
            path=out/'matches'/f'{a:08}_{b:08}.npz'
            if path.exists():continue
            result=matcher({'image0':feature(a),'image1':feature(b)})
            np.savez_compressed(path,matches=result['matches'][0].cpu().numpy().astype(np.uint32),scores=result['scores'][0].cpu().numpy().astype(np.float32))
            if n%50==0 or n==len(pairs):print(f'[matching] {n}/{len(pairs)} ({100*n/len(pairs):.1f}%)',flush=True)
    write_json(out/'matches_done.json',dict(pairs=len(pairs),offsets=[1,2,3,5],device='cuda',dtype='float32'))


def database(out,frames,prepared):
    import pycolmap as p
    if (out/'database_done.json').exists():return
    db_path=out/'database.db'
    if db_path.exists():raise ValueError('Incomplete database exists; preserve it and use a new output')
    image_dir=out/'images';image_dir.mkdir(exist_ok=True)
    for r in frames:
        link=image_dir/f"frame_{int(r['frame_id']):08}.png"
        if not link.exists():link.symlink_to((prepared/r['image']).resolve())
    K,w,h=camera_cv(prepared)
    # Both coordinates and principal point move by +0.5 exactly once. Normalized
    # rays stay identical to OpenCV's pixel-center-at-zero convention.
    params=[K[0,0],K[1,1],K[0,2]+.5,K[1,2]+.5]
    options=p.ImageReaderOptions(camera_model='PINHOLE',camera_params=','.join(format(x,'.17g') for x in params))
    with p.Database.open(str(db_path)):
        pass
    p.import_images(str(db_path),str(image_dir),camera_mode=p.CameraMode.SINGLE,options=options)
    with p.Database.open(str(db_path)) as db:
        images={int(im.name[6:-4]):im for im in db.read_all_images()}
        for i,im in images.items():db.write_keypoints(im.image_id,load_feature(out,i)['keypoints']+.5)
        pairs=[]
        for a in sorted(images):
            for delta in (1,2,3,5):
                b=a+delta
                if b not in images:continue
                m=load_match(out,a,b).astype(np.uint32);db.write_matches(images[a].image_id,images[b].image_id,m)
                pairs.append(f'{images[a].name} {images[b].name}')
    (out/'pairs.txt').write_text('\n'.join(pairs)+'\n')
    options=p.TwoViewGeometryOptions();options.ransac.max_error=1.5;options.ransac.random_seed=0
    print('[verification] starting common calibrated database geometry',flush=True)
    p.verify_matches(str(db_path),str(out/'pairs.txt'),options=options)
    write_json(out/'database_done.json',dict(images=len(images),pairs=len(pairs),camera_model='PINHOLE',camera_params=params,feature_offset=.5,principal_point_offset=.5,pycolmap=p.__version__,has_cuda=p.has_cuda,database_sha256=hashlib.sha256(db_path.read_bytes()).hexdigest()))


def reconstruction_report(model,frames,path):
    wanted={int(r['frame_id']):r for r in frames};registered=[];export=[];negative=0;observations=0;errors=[];observation_errors=[];tracks=[]
    for im in model.images.values():
        if not im.has_pose:continue
        source=int(im.name[6:-4]);registered.append(source);T=im.cam_from_world().inverse();matrix=T.matrix();q=T.rotation.quat
        export.append(dict(frame_id=source,time_ns=int(wanted[source]['time_ns']),tx=float(matrix[0,3]),ty=float(matrix[1,3]),tz=float(matrix[2,3]),qx=float(q[0]),qy=float(q[1]),qz=float(q[2]),qw=float(q[3])))
    for point in model.points3D.values():
        tracks.append(point.track.length());errors.append(point.error)
        for el in point.track.elements:
            im=model.images[el.image_id];xyz=im.cam_from_world()*point.xyz;observations+=1;negative+=int(xyz[2]<=0)
            if xyz[2]>0:observation_errors.append(float(np.linalg.norm(im.camera.img_from_cam(xyz)-im.points2D[el.point2D_idx].xy)))
    export.sort(key=lambda r:r['frame_id'])
    if export:
        with (path/'visual_camera_to_world.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
        np.savez_compressed(path/'visual_trajectory.npz',frame_ids=np.array([r['frame_id'] for r in export],np.int64),timestamps_ns=np.array([r['time_ns'] for r in export],np.int64),camera_to_world=np.array([model.images[next(i for i,im in model.images.items() if int(im.name[6:-4])==r['frame_id'])].cam_from_world().inverse().matrix() for r in export],np.float64),metric=np.array(False))
    spans=intervals(registered);missing=intervals(set(wanted)-set(registered))
    summary=dict(registered=len(registered),total=len(frames),coverage=len(registered)/len(frames),registered_intervals=spans,unregistered_intervals=missing,longest_contiguous_frames=max((b-a+1 for a,b in spans),default=0),points3D=len(tracks),track_length_median=float(np.median(tracks)) if tracks else None,reprojection_error_mean=model.compute_mean_reprojection_error() if tracks else None,point_mean_reprojection_error_p95=float(np.percentile(errors,95)) if errors else None,observation_reprojection_error_p95=float(np.percentile(observation_errors,95)) if observation_errors else None,outlier_observations_over_3px=sum(e>3 for e in observation_errors),negative_depth_observations=negative,observations=observations,metric=False,units='arbitrary_visual',pose_convention='camera_to_world',timestamp_dtype='int64',uses_future_images=True)
    write_json(path/'quality.json',summary);return summary


def reconstruct(out,frames,seed_policy,limit=None,single_map=False):
    import pycolmap as p
    if not p.has_cuda:raise RuntimeError('CUDA-enabled PyCOLMAP required')
    if seed_policy!='auto' and not single_map:raise ValueError('Forced initial pair comparisons require --single-map, preventing repeated reuse for subsequent maps')
    subset=frames[:limit] if limit else frames
    suffix='_single' if single_map else ''
    target=out/f'mapping_{seed_policy}_{len(subset)}{suffix}';target.mkdir(exist_ok=False)
    options=p.IncrementalPipelineOptions();options.image_names=[f"frame_{int(r['frame_id']):08}.png" for r in subset]
    options.num_threads=4;options.random_seed=0;options.mapper.random_seed=0
    options.ba_refine_focal_length=options.ba_refine_principal_point=options.ba_refine_extra_params=False
    options.ba_refine_sensor_from_rig=False;options.mapper.abs_pose_refine_focal_length=options.mapper.abs_pose_refine_extra_params=False
    # The pinned CUDA wheel supplies neural/feature acceleration but its Ceres
    # build has no CUDA solver. Request CPU explicitly rather than silent fallback.
    options.ba_use_gpu=False
    options.ba_local_backend=p.BundleAdjustmentBackend.CERES;options.ba_global_backend=p.BundleAdjustmentBackend.CERES
    # Same angular criterion already exercised in the correspondence benchmark;
    # the normal COLMAP initializer assumes much wider temporal baselines (16deg).
    options.mapper.init_min_tri_angle=1.;options.multiple_models=not single_map
    if seed_policy!='auto':
        candidates=read_rows(ROOT/'output/neon_Htou_lightglue_audit_01/results.csv')
        ids={int(r['frame_id']) for r in subset};start=min(ids)
        candidates=[r for r in candidates if r['group']=='hard' and r['common_gate_pass']=='True' and int(r['reference']) in ids and int(r['target']) in ids and int(r['reference'])<start+60]
        if not candidates:
            write_json(target/'summary.json',dict(failed=True,reason='No seed satisfies predefined rule',models=[]));return
        key=(lambda r:(int(r['reference']),int(r['target']))) if seed_policy=='earliest' else (lambda r:(-int(r['triangulated']),int(r['reference']),int(r['target'])))
        chosen=min(candidates,key=key)
        with p.Database.open(str(out/'database.db')) as db:
            lookup={int(im.name[6:-4]):im.image_id for im in db.read_all_images()}
        options.init_image_id1=lookup[int(chosen['reference'])];options.init_image_id2=lookup[int(chosen['target'])]
        write_json(target/'seed.json',dict(rule=seed_policy,reference=int(chosen['reference']),target=int(chosen['target'])))
    (target/'options.txt').write_text(options.summary())
    started=time.monotonic();progress={'registered_callbacks':0}
    def callback():
        progress['registered_callbacks']+=1
        if progress['registered_callbacks']%10==0:print(f'[mapping {seed_policy}] registration callbacks {progress["registered_callbacks"]}, elapsed {time.monotonic()-started:.1f}s',flush=True)
    models=p.incremental_mapping(str(out/'database.db'),str(out/'images'),str(target),options=options,next_image_callback=callback)
    summaries=[];all_registered=set()
    for index,model in models.items():
        path=target/str(index);path.mkdir(exist_ok=True);model.write(str(path));summary=reconstruction_report(model,subset,path);summary['model_id']=index;summaries.append(summary)
        all_registered.update(int(im.name[6:-4]) for im in model.images.values() if im.has_pose)
    write_json(target/'summary.json',dict(models=summaries,num_models=len(models),registered_unique=len(all_registered),total=len(subset),largest_single_map=max((r['registered'] for r in summaries),default=0),elapsed_seconds=time.monotonic()-started,metric=False,uses_future_images=True,calibration_fixed=True,ba_backend="CERES_CPU",neural_backend="CUDA_FLOAT32",multiple_models=not single_map))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start_n',type=int,default=25834);parser.add_argument('--max_n',type=int,default=900)
    parser.add_argument('--stage',choices=['all','extract','match','database','triplets','reconstruct'],default='all')
    parser.add_argument('--single-map',action='store_true',help='Compare initial-pair policies with one map and identical settings')
    parser.add_argument('--seed-policy',choices=['auto','earliest','most_points'],default='auto');parser.add_argument('--reconstruct_n',type=int)
    parser.add_argument('--runtime',type=Path,default=ROOT/'output/neon_sfm_runtime');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();setup(args.runtime)
    frames=[r for r in read_rows(args.prepared/'frames.csv') if args.start_n<=int(r['frame_id'])<args.start_n+args.max_n]
    if len(frames)!=args.max_n:parser.error('Requested complete contiguous frame interval is unavailable')
    identity=dict(prepared=str(args.prepared.resolve()),frames=[dict(frame_id=int(r['frame_id']),time_ns=int(r['time_ns']),image=r['image']) for r in frames],camera_sha256=hashlib.sha256((args.prepared/'camera.yaml').read_bytes()).hexdigest(),feature_dtype='float32',lightglue_revision=LIGHTGLUE_REVISION,metric=False)
    if args.output.exists():
        if not args.resume or json.loads((args.output/'identity.json').read_text())!=identity:parser.error('Output reuse requires --resume and matching input identity')
    else:args.output.mkdir(parents=True);write_json(args.output/'identity.json',identity)
    stage=args.stage
    if stage in ['all','extract']:extract(args.output,frames,args.prepared)
    if stage in ['all','match']:match(args.output,frames)
    if stage in ['all','triplets']:
        from neon_multiview import run_cases
        run_cases(args.output,frames,args.prepared)
    if stage in ['all','database']:database(args.output,frames,args.prepared)
    if stage=='all':reconstruct(args.output,frames,'auto',40)
    if stage in ['all','reconstruct']:reconstruct(args.output,frames,args.seed_policy,args.reconstruct_n,args.single_map)


if __name__=='__main__':main()
