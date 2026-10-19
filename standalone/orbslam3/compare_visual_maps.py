#!/usr/bin/env python3
"""Compare observed visual maps using source IDs and feature observations.

Point correspondence never uses a run-local MapPoint ID. Sim(3) is fit jointly
on shared camera centres and shared observed points; relative rotation errors
are separately reported to distinguish scale gauge from map deformation.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sim3(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3:
        raise ValueError('Sim3 needs at least three paired 3D positions')
    aa=a-a.mean(0);bb=b-b.mean(0);variance=np.sum(aa*aa)
    if variance<1e-24:raise ValueError('Degenerate Sim3 input')
    U,s,Vt=np.linalg.svd(bb.T@aa)
    D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vt)
    R=U@D@Vt;scale=float(np.sum(s*np.diag(D))/variance)
    translation=b.mean(0)-scale*R@a.mean(0)
    residual=np.linalg.norm(scale*a@R.T+translation-b,axis=1)
    return dict(scale=scale,rotation=R.tolist(),translation=translation.tolist(),
                rms=float(np.sqrt(np.mean(residual**2))),max=float(residual.max()),
                input_rank=int(np.linalg.matrix_rank(aa)),singular_values=s.tolist())


def observations(point):
    return frozenset((int(o['source_frame_id']),int(o['feature']))
                     for o in point.get('observations',[]) if int(o['feature'])>=0)


def match_points(a,b):
    # Require two shared feature observations and a unique mutual best overlap.
    lookup=defaultdict(set)
    for j,p in enumerate(b):
        for key in observations(p):lookup[key].add(j)
    choices=[]
    for i,p in enumerate(a):
        counts=Counter(j for key in observations(p) for j in lookup[key])
        best=counts.most_common(2)
        if best and best[0][1]>=2 and (len(best)==1 or best[0][1]>best[1][1]):
            choices.append((i,*best[0]))
    reverse_lookup=defaultdict(set)
    for i,p in enumerate(a):
        for key in observations(p):reverse_lookup[key].add(i)
    reverse_best={}
    for j,p in enumerate(b):
        best=Counter(i for key in observations(p) for i in reverse_lookup[key]).most_common(2)
        if best and best[0][1]>=2 and (len(best)==1 or best[0][1]>best[1][1]):
            reverse_best[j]=best[0][0]
    return [(i,j) for i,j,n in choices if reverse_best.get(j)==i]


def state(snapshot,result=None):
    if 'vertices_insertion_order' not in snapshot:return snapshot
    vertices={int(v['id']):v for v in (result['vertices'] if result else snapshot['vertices_insertion_order'])}
    offset=max((int(v['id']) for v in snapshot['vertices_insertion_order'] if v['type']=='pose'),default=-1)+1
    keyframes=[];points=[]
    for k in snapshot.get('keyframes',[]):
        v=vertices.get(int(k['kf_id']))
        if not v:continue
        R=np.array(v['Rcw'],float).reshape(3,3);t=np.array(v['tcw'],float)
        keyframes.append(dict(**k,Rcw=R.ravel().tolist(),center=(-R.T@t).tolist()))
    for p in snapshot.get('map_points_identity',[]):
        v=vertices.get(int(p['point_id'])+offset)
        if v:points.append(dict(**{k:value for k,value in p.items() if k!='xyz'},xyz=v['xyz']))
    return dict(keyframes=keyframes,points=points)


def compare(a,b):
    ak={int(k['source_frame_id']):k for k in a.get('keyframes',[])}
    bk={int(k['source_frame_id']):k for k in b.get('keyframes',[])}
    shared=sorted(ak.keys()&bk.keys());pairs=match_points(a.get('points',[]),b.get('points',[]))
    ca=np.array([ak[i]['center'] for i in shared],float).reshape(-1,3)
    cb=np.array([bk[i]['center'] for i in shared],float).reshape(-1,3)
    pa=np.array([a['points'][i]['xyz'] for i,j in pairs],float).reshape(-1,3)
    pb=np.array([b['points'][j]['xyz'] for i,j in pairs],float).reshape(-1,3)
    matching_counts={}
    for field in ['descriptor_hex','reference_source','normal','min_distance','max_distance']:
        available=[(a['points'][i]['matching_state'][field],b['points'][j]['matching_state'][field])
                   for i,j in pairs if 'matching_state' in a['points'][i] and 'matching_state' in b['points'][j]]
        if available:matching_counts[field]=sum(x!=y for x,y in available)
    result=dict(matching_state_mismatches=matching_counts,observation_set_mismatches=sum(observations(a['points'][i])!=observations(b['points'][j]) for i,j in pairs),common_keyframes=shared,common_points=len(pairs),keyframes_a=len(ak),keyframes_b=len(bk),
                points_a=len(a.get('points',[])),points_b=len(b.get('points',[])))
    if len(ca)+len(pa)>=3:
        fit=sim3(np.concatenate([ca,pa]),np.concatenate([cb,pb]));result['joint_sim3']=fit
        R=np.array(fit['rotation']);s=fit['scale'];t=np.array(fit['translation'])
        for name,x,y in [('camera',ca,cb),('point',pa,pb)]:
            if len(x):
                error=np.linalg.norm(s*x@R.T+t-y,axis=1)
                result[name+'_aligned_rms']=float(np.sqrt(np.mean(error**2)))
                result[name+'_unaligned_max']=float(np.max(np.abs(x-y)))
        extent=float(np.sqrt(np.mean(np.sum((pb-pb.mean(0))**2,axis=1)))) if len(pb) else 0
        result['point_relative_rms']=result.get('point_aligned_rms',0)/extent if extent>0 else None
    if len(ca)>=3:
        result['camera_only_sim3']=sim3(ca,cb)
    rotations=[]
    for i in shared[1:]:
        ref=shared[0]
        ra=np.array(ak[i]['Rcw'],float).reshape(3,3)@np.array(ak[ref]['Rcw'],float).reshape(3,3).T
        rb=np.array(bk[i]['Rcw'],float).reshape(3,3)@np.array(bk[ref]['Rcw'],float).reshape(3,3).T
        rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(ra@rb.T)-1)/2,-1,1)))))
    result['relative_rotation_max_deg']=max(rotations,default=0.)
    return result


def graph_quality(snapshot,result=None):
    vertices={int(v['id']):v for v in (result['vertices'] if result else snapshot['vertices_insertion_order'])}
    errors=[];negative=0;cached=[];adj=defaultdict(set)
    for e in snapshot['edges_insertion_order']:
        pi,ci=int(e['point_vertex']),int(e['pose_vertex']);adj[pi].add(ci);adj[ci].add(pi)
        point=np.array(vertices[pi]['xyz'],float);pose=vertices[ci]
        q=np.array(pose['Rcw'],float).reshape(3,3)@point+np.array(pose['tcw'],float)
        negative+=int(q[2]<=0)
        fx,fy,cx,cy=map(float,e['camera'])
        residual=np.array([fx*q[0]/q[2]+cx,fy*q[1]/q[2]+cy])-np.array(e['xy'],float)
        errors.append(float(residual@np.array(e['information'],float).reshape(2,2)@residual))
    components=0;remaining=set(adj)
    while remaining:
        components+=1;queue=[remaining.pop()]
        while queue:
            for v in adj[queue.pop()]&remaining:remaining.remove(v);queue.append(v)
    if result:
        cached=[float(e['chi2']) for e in result.get('edge_results',[])]
    fixed=[int(v['id']) for v in vertices.values() if v['type']=='pose' and str(v['fixed']).lower()=='true']
    return dict(fixed_pose_vertices=fixed,connected_components=components,edges=len(errors),negative_depth=negative,
                actual_chi2=float(sum(errors)),actual_outliers_5_991=sum(e>5.991 for e in errors),
                cached_chi2=sum(cached) if cached else None,
                cached_actual_max_difference=max((abs(a-b) for a,b in zip(cached,errors)),default=None))


def load_sequence(folder):
    counters=Counter();result={}
    for path in sorted(Path(folder).glob('visual_*.json'),key=lambda p:int(p.name.split('_')[1])):
        if path.name.endswith('.result.json'):continue
        data=read(path);key=(int(data['source_frame_id']),data['stage']);n=counters[key];counters[key]+=1
        result[(*key,n)]=(path,data)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('a',type=Path);parser.add_argument('b',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();aa=load_sequence(args.a);bb=load_sequence(args.b);rows=[]
    for key in sorted(aa.keys()&bb.keys()):
        ap,a=aa[key];bp,b=bb[key]
        if 'associations' in a:
            af={int(p['feature']):p for p in a['associations']};bf={int(p['feature']):p for p in b['associations']}
            common=af.keys()&bf.keys()
            rows.append(dict(source_frame_id=key[0],stage=key[1],occurrence=key[2],a=str(ap),b=str(bp),
                             pose_max_difference=float(np.max(np.abs(np.array(a['Tcw'],float)-np.array(b['Tcw'],float)))),
                             associations_a=len(af),associations_b=len(bf),feature_set_difference=len(af.keys()^bf.keys()),
                             common_feature_xyz_max_difference=max((float(np.max(np.abs(np.array(af[i]['xyz'],float)-np.array(bf[i]['xyz'],float)))) for i in common),default=None)))
            continue
        if a['stage']=='two_view_reconstruction':
            ma={(int(m['feature1']),int(m['feature2'])):np.array(m['xyz'],float) for m in a['matches']}
            mb={(int(m['feature1']),int(m['feature2'])):np.array(m['xyz'],float) for m in b['matches']}
            shared=ma.keys()&mb.keys()
            rows.append(dict(source_frame_id=key[0],stage=key[1],a=str(ap),b=str(bp),
                             common_matches=len(shared),matches_a=len(ma),matches_b=len(mb),
                             pose_max_difference=float(np.max(np.abs(np.array(a['Tcw'],float)-np.array(b['Tcw'],float)))),
                             points_max_difference=max((float(np.max(np.abs(ma[i]-mb[i]))) for i in shared),default=None)))
            continue
        if not ('keyframes' in a or 'vertices_insertion_order' in a):continue
        row=dict(source_frame_id=key[0],stage=key[1],occurrence=key[2],a=str(ap),b=str(bp),input=compare(state(a),state(b)))
        if 'vertices_insertion_order' in a:
            ar=read(str(ap)+'.result.json');br=read(str(bp)+'.result.json')
            row.update(result=compare(state(a,ar),state(b,br)),quality_a=graph_quality(a,ar),quality_b=graph_quality(b,br),
                       actual_iterations_a=int(ar['iterations_actual']),actual_iterations_b=int(br['iterations_actual']))
        rows.append(row)
    args.output.write_text(json.dumps(dict(comparisons=rows,only_a=[list(k) for k in sorted(aa.keys()-bb.keys())],only_b=[list(k) for k in sorted(bb.keys()-aa.keys())]),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
