"""Three/four-view PnP and joint BA using shared learned point identities."""
from pathlib import Path
import json
import cv2
import numpy as np

try:
    from .neon_sfm import load_feature,load_match,camera_cv,read_rows,write_json,ROOT
except ImportError:
    from neon_sfm import load_feature,load_match,camera_cv,read_rows,write_json,ROOT


def triangulate(a,b,K):
    cv2.setRNGSeed(0)
    E,mask=cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=.999,threshold=1.5)
    if E is None or E.shape!=(3,3):return None
    H,hm=cv2.findHomography(a,b,cv2.RANSAC,1.5)
    _,R,t,pm=cv2.recoverPose(E,a,b,K,mask=mask.copy())
    na=cv2.undistortPoints(a.reshape(-1,1,2),K,None).reshape(-1,2)
    nb=cv2.undistortPoints(b.reshape(-1,1,2),K,None).reshape(-1,2)
    # Keep all four E hypotheses to expose cheirality ambiguity. This is a
    # diagnostic, not success-based reselection of initial pairs.
    R_a,R_b,t_e=cv2.decomposeEssentialMat(E)
    cheirality=[]
    for rr in (R_a,R_b):
        for tt in (t_e,-t_e):
            hh=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[rr,tt],na.T,nb.T)
            with np.errstate(divide='ignore',invalid='ignore'):
                xx=(hh[:3]/hh[3]).T;yy=xx@rr.T+tt.ravel()
            cheirality.append(int(((mask.ravel()!=0)&np.isfinite(xx).all(1)&(xx[:,2]>0)&(yy[:,2]>0)).sum()))
    Xh=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[R,t],na.T,nb.T)
    with np.errstate(divide='ignore',invalid='ignore'):
        X=(Xh[:3]/Xh[3]).T;Y=X@R.T+t.ravel()
        p1=X@K.T;p1=p1[:,:2]/p1[:,2:];p2=Y@K.T;p2=p2[:,:2]/p2[:,2:]
        c=(-R.T@t).ravel();rays=X-c
        cosine=(X*rays).sum(1)/(np.linalg.norm(X,axis=1)*np.linalg.norm(rays,axis=1))
        angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    good=(pm.ravel()!=0)&np.isfinite(X).all(1)&(X[:,2]>0)&(Y[:,2]>0)&(np.linalg.norm(p1-a,axis=1)<=3)&(np.linalg.norm(p2-b,axis=1)<=3)&(angles>=1)
    eig=np.linalg.svd(X[good]-X[good].mean(0),compute_uv=False) if good.sum()>=3 else np.zeros(3)
    return dict(R=R,t=t.ravel(),xyz=X[good],indices=np.flatnonzero(good),essential_inliers=int(mask.sum()),homography_inliers=int(hm.sum()) if hm is not None else 0,
                essential_cheirality_counts=cheirality,median_valid_parallax_deg=float(np.median(angles[good])) if good.any() else None,point_planarity_ratio=float(eig[-1]/eig[0]) if eig[0]>0 else None)


def pnp(xyz,xy,K):
    if len(xyz)<15:return None
    cv2.setRNGSeed(0)
    ok,r,t,indices=cv2.solvePnPRansac(np.asarray(xyz,np.float64),np.asarray(xy,np.float64),K,None,iterationsCount=1000,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_AP3P)
    if not ok or indices is None or len(indices)<15:return None
    ii=indices.ravel();r,t=cv2.solvePnPRefineLM(np.asarray(xyz[ii],np.float64),np.asarray(xy[ii],np.float64),K,None,r,t)
    return cv2.Rodrigues(r)[0],t.ravel(),ii


def new_model(K,width,height,features,poses):
    import pycolmap as p
    model=p.Reconstruction();camera=p.Camera(model='PINHOLE',width=width,height=height,params=[K[0,0],K[1,1],K[0,2]+.5,K[1,2]+.5],camera_id=1)
    model.add_camera(camera);rig=p.Rig(rig_id=1);rig.add_ref_sensor(p.sensor_t(p.SensorType.CAMERA,1));model.add_rig(rig)
    for image_id,(feature,(R,t)) in enumerate(zip(features,poses),1):
        im=p.Image(name=f'view_{image_id}.png',keypoints=np.asarray(feature['keypoints']+.5,np.float64),camera_id=1,image_id=image_id)
        model.add_image_with_trivial_frame(im,p.Rigid3d(p.Rotation3d(R),t))
    return model


def reprojection(model):
    errors=[];negative=0
    for point in model.points3D.values():
        for el in point.track.elements:
            im=model.images[el.image_id];q=im.cam_from_world()*point.xyz
            negative+=int(q[2]<=0)
            if q[2]>0:errors.append(float(np.linalg.norm(im.camera.img_from_cam(q)-im.points2D[el.point2D_idx].xy)))
    return dict(observations=len(errors)+negative,negative_depth=negative,median_reprojection_px=float(np.median(errors)) if errors else None,p95_reprojection_px=float(np.percentile(errors,95)) if errors else None)


def ba(model):
    import pycolmap as p
    options=p.BundleAdjustmentOptions();options.backend=p.BundleAdjustmentBackend.CERES;options.refine_focal_length=options.refine_principal_point=options.refine_extra_params=False;options.refine_sensor_from_rig=False;options.print_summary=False
    options.ceres.use_gpu=False
    options.ceres.solver_options.num_threads=2
    config=p.BundleAdjustmentConfig()
    for i in model.reg_image_ids():config.add_image(i)
    config.set_constant_cam_intrinsics(1);config.fix_gauge(p.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    before=model.cameras[1].params.copy();quality_before=reprojection(model)
    adjuster=p.create_default_bundle_adjuster(options,config,model);summary=adjuster.solve()
    if not np.array_equal(before,model.cameras[1].params):raise RuntimeError('Intrinsics unexpectedly changed')
    return dict(usable=summary.is_solution_usable(),report=summary.brief_report(),before=quality_before,after=reprojection(model),gpu_requested=False,backend='CERES_CPU',gauge='first pose and one translation component; TWO_CAMS_FROM_WORLD')


def run_case(out,case,prepared):
    import pycolmap as p
    a,b,c,d=case['frames'];fs=[load_feature(out,i) for i in [a,b,c,d]];K,w,h=camera_cv(prepared)
    ab=load_match(out,a,b);bc=load_match(out,b,c);bd=load_match(out,b,d)
    report=dict(**case,success=False,metric=False,units='arbitrary_visual',pose_convention='camera_to_world')
    if len(ab)<8:report['reason']='fewer_than_8_initial_matches';return report
    tri=triangulate(fs[0]['keypoints'][ab[:,0]],fs[1]['keypoints'][ab[:,1]],K)
    if tri is None:report['reason']='no_unique_essential_matrix';return report
    report['two_view']={k:v for k,v in tri.items() if k not in ['R','t','xyz','indices']};report['initial_triangulated']=len(tri['xyz'])
    link3=dict(bc.tolist());valid_ab=ab[tri['indices']];which=np.array([j for j,x in enumerate(valid_ab[:,1]) if int(x) in link3],int)
    third_indices=np.array([link3[int(valid_ab[j,1])] for j in which],int)
    pose=pnp(tri['xyz'][which],fs[2]['keypoints'][third_indices],K)
    report['pnp3_correspondences']=len(which)
    if pose is None:report['reason']='pnp3_failed';return report
    R3,t3,in3=pose;report['pnp3_inliers']=len(in3)
    keep=which[in3];third_indices=third_indices[in3]
    model=new_model(K,w,h,fs[:3],[(np.eye(3),np.zeros(3)),(tri['R'],tri['t']),(R3,t3)])
    point_ids=[]
    for j,idx in enumerate(keep):
        track=p.Track();track.add_element(1,int(valid_ab[idx,0]));track.add_element(2,int(valid_ab[idx,1]));track.add_element(3,int(third_indices[j]))
        point_ids.append(model.add_point3D(tri['xyz'][idx],track))
    report['ba3']=ba(model)
    if not report['ba3']['usable']:report['reason']='ba3_unusable';return report
    link4=dict(bd.tolist());which4=[j for j,idx in enumerate(keep) if int(valid_ab[idx,1]) in link4]
    xy4=np.array([fs[3]['keypoints'][link4[int(valid_ab[keep[j],1])]] for j in which4],np.float64).reshape(-1,2)
    xyz4=np.array([model.points3D[point_ids[j]].xyz for j in which4],np.float64).reshape(-1,3)
    pose4=pnp(xyz4,xy4,K);report['pnp4_correspondences']=len(which4)
    if pose4 is None:report['reason']='pnp4_failed';return report
    R4,t4,in4=pose4;report['pnp4_inliers']=len(in4)
    im=p.Image(name='view_4.png',keypoints=np.asarray(fs[3]['keypoints']+.5,np.float64),camera_id=1,image_id=4)
    model.add_image_with_trivial_frame(im,p.Rigid3d(p.Rotation3d(R4),t4))
    for index in in4:
        j=which4[int(index)];feature_index=link4[int(valid_ab[keep[j],1])];model.add_observation(point_ids[j],p.TrackElement(4,feature_index))
    report['ba4']=ba(model);q=report['ba4']['after']
    report['success']=bool(report['ba4']['usable'] and q['median_reprojection_px'] is not None and q['median_reprojection_px']<=3 and q['negative_depth']<=.05*q['observations'])
    report['reason']='four_view_joint_ba_passed' if report['success'] else 'four_view_joint_ba_failed'
    folder=out/'triplets'/case['name'];folder.mkdir(exist_ok=True);model.write(str(folder))
    report['camera_to_world']={str(frame):model.images[i].cam_from_world().inverse().matrix().tolist() for i,frame in enumerate([a,b,c,d],1)}
    return report


def run_cases(out,frames,prepared):
    folder=out/'triplets';folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():return
    rows=read_rows(ROOT/'output/neon_Htou_lightglue_audit_01/results.csv');ids={int(r['frame_id']) for r in frames}
    rows=[r for r in rows if r['group']=='hard' and all(i in ids for i in [int(r['reference']),int(r['target']),int(r['target'])+1,int(r['target'])+2])]
    adjacent={int(r['reference']):r['common_gate_pass']=='True' for r in rows if r['delta']=='1'}
    groups={'adjacent_pass':[r for r in rows if r['delta']=='1' and r['common_gate_pass']=='True'],
            'delayed_rescue':[r for r in rows if r['delta']!='1' and r['common_gate_pass']=='True' and not adjacent.get(int(r['reference']),False)],
            'failed_gate':[r for r in rows if r['delta']=='1' and r['common_gate_pass']=='False']}
    cases=[]
    for group,rs in groups.items():
        chosen=[];seen=set()
        for r in sorted(rs,key=lambda r:(int(r['reference']),int(r['target']))):
            a,b=int(r['reference']),int(r['target'])
            if a in seen:continue
            seen.add(a);chosen.append(dict(name=f'{group}_{len(chosen):02}',group=group,frames=[a,b,b+1,b+2]))
            if len(chosen)==5:break
        cases.extend(chosen)
    write_json(folder/'plan.json',dict(selection='First 5 distinct references per fixed group, sorted source IDs; first valid delayed offset; no success-based reselection',cases=cases,pnp_min_inliers=15,reprojection_max_px=3.,minimum_parallax_deg=1.,joint_ba_gauge='TWO_CAMS_FROM_WORLD',metric=False))
    results=[]
    for case in cases:
        try:result=run_case(out,case,prepared)
        except Exception as error:result=dict(**case,success=False,reason=type(error).__name__+': '+str(error))
        results.append(result);write_json(folder/'partial_results.json',results);print('[multiview]',case['name'],result['success'],result['reason'],flush=True)
    write_json(folder/'results.json',results)
