"""Geometric contracts for the independent visual audit; no optional wheel needed."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from standalone.orbslam3.neon_sfm import intervals, load_feature
from standalone.orbslam3.neon_multiview import pnp, triangulate
from standalone.orbslam3.compare_visual_maps import sim3, match_points, compare, graph_quality


def test_sim3_separates_scale_from_deformation():
    rng=np.random.default_rng(7);a=rng.normal(size=(50,3))
    R=cv2.Rodrigues(np.array([.1,-.2,.4]))[0];b=3*a@R.T+[4,2,-3]
    fit=sim3(a,b)
    assert fit['scale']==pytest.approx(3)
    assert fit['max']<1e-13
    b[0]+=[1,0,0]
    assert sim3(a,b)['rms']>.1
    with pytest.raises(ValueError,match='Degenerate'):sim3(np.zeros((3,3)),np.ones((3,3)))


def test_point_identity_uses_source_features_not_native_ids():
    def point(i,obs):return dict(point_id=i,observations=[dict(source_frame_id=a,feature=b) for a,b in obs])
    a=[point(1,[(101,5),(104,9)]),point(2,[(101,7),(104,8)])]
    b=[point(999,[(101,7),(104,8),(107,1)]),point(1,[(101,5),(104,9)])]
    assert sorted(match_points(a,b))==[(0,1),(1,0)]
    # A single overlap is insufficient, and ambiguous targets are rejected.
    assert match_points(a,[point(1,[(101,5)])])==[]


def test_half_pixel_conversion_preserves_rays():
    K=np.array([[430.,0,399.5],[0,431.,299.5],[0,0,1.]])
    xy=np.array([[0.,0.],[30.25,42.75],[799,599]])
    shifted=K.copy();shifted[:2,2]+=.5
    rays=(np.c_[xy,np.ones(3)]@np.linalg.inv(K).T)
    converted=(np.c_[xy+.5,np.ones(3)]@np.linalg.inv(shifted).T)
    np.testing.assert_allclose(rays,converted,atol=1e-15)


def test_real_three_dimensional_pnp_recovers_third_camera():
    rng=np.random.default_rng(8);X=rng.uniform([-2,-1,4],[2,1,9],(300,3))
    K=np.array([[440.,0,400.],[0,440.,300.],[0,0,1.]])
    def xy(R,t):
        q=(X@R.T+t)@K.T;return q[:,:2]/q[:,2:]
    R1=np.eye(3);t1=np.zeros(3);R2=cv2.Rodrigues(np.array([.01,.03,0]))[0];t2=np.array([.4,.02,.01])
    tri=triangulate(xy(R1,t1),xy(R2,t2),K)
    assert len(tri['xyz'])>200
    R3=cv2.Rodrigues(np.array([.02,.06,.01]))[0];t3=np.array([.8,.03,.02])
    result=pnp(tri['xyz'],xy(R3,t3)[tri['indices']],K)
    assert result is not None
    np.testing.assert_allclose(result[0],R3,atol=2e-5)
    # Initial two-view translation is unit length, so PnP shares that scale.
    np.testing.assert_allclose(result[1],t3/np.linalg.norm(t2),atol=2e-4)


def test_missing_intervals_and_float32_cache_contract(tmp_path):
    assert intervals([9,4,3,8,4])==[[3,4],[8,9]]
    (tmp_path/'features').mkdir()
    np.savez(tmp_path/'features/00000003.npz',keypoints=np.ones((3,2),np.float64))
    with pytest.raises(ValueError,match='Float32'):load_feature(tmp_path,3)


def test_visual_flags_require_diagnostics(tmp_path):
    root=Path(__file__).resolve().parents[1]
    process=subprocess.run([sys.executable,str(root/'standalone/orbslam3/run.py'),'--visual-snapshots','--output',str(tmp_path/'unused')],capture_output=True,text=True)
    assert process.returncode==2
    assert '--visual-snapshots requires --diagnostics' in process.stderr
    assert not (tmp_path/'unused').exists()


def test_graph_checks_actual_error_without_mutating_cached_native_errors():
    graph=dict(vertices_insertion_order=[dict(id=0,type='pose',fixed=True,Rcw=np.eye(3).ravel().tolist(),tcw=[0,0,0]),dict(id=1,type='point',fixed=False,xyz=[0,0,2])],edges_insertion_order=[dict(point_vertex=1,pose_vertex=0,camera=[400,400,400,300],xy=[401,300],information=[1,0,0,1])])
    result=dict(vertices=graph['vertices_insertion_order'],edge_results=[dict(chi2=9)])
    quality=graph_quality(graph,result)
    assert quality['actual_chi2']==1
    assert quality['cached_chi2']==9
    assert quality['cached_actual_max_difference']==8
    assert quality['fixed_pose_vertices']==[0]
    assert quality['connected_components']==1


def test_visual_export_preserves_nanoseconds_and_inverts_pose(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    deps=root/'output/neon_sfm_runtime/deps'
    if not deps.exists():pytest.skip('Optional independent COLMAP runtime absent')
    monkeypatch.syspath_prepend(str(deps))
    pytest.importorskip('pycolmap')
    from standalone.orbslam3.neon_multiview import new_model
    from standalone.orbslam3.neon_sfm import reconstruction_report
    K=np.array([[400.,0,400.],[0,400.,300.],[0,0,1.]])
    R=cv2.Rodrigues(np.array([.1,.2,-.3]))[0];t=np.array([1.,2.,3.])
    model=new_model(K,800,600,[dict(keypoints=np.zeros((0,2),np.float32))],[(R,t)])
    model.images[1].name='frame_00000103.png'
    timestamp=1760000000123456789
    report=reconstruction_report(model,[dict(frame_id=103,time_ns=timestamp)],tmp_path)
    with np.load(tmp_path/'visual_trajectory.npz') as a:
        assert a['timestamps_ns'].dtype==np.int64
        assert a['timestamps_ns'][0]==timestamp
        assert not bool(a['metric'])
        np.testing.assert_allclose(a['camera_to_world'][0,:,:3],R.T,atol=1e-14)
        np.testing.assert_allclose(a['camera_to_world'][0,:,3],-R.T@t,atol=1e-14)
    assert report['registered']==1
    assert not (tmp_path/'pipeline.json').exists()


def test_descriptor_tie_changes_with_order_and_canonical_order_removes_it():
    from standalone.orbslam3.replay_point_descriptors import select_descriptor
    a=np.zeros(32,np.uint8);b=np.full(32,255,np.uint8)
    first,ties=select_descriptor(np.array([a,b]))
    reversed_first,reversed_ties=select_descriptor(np.array([b,a]))
    assert ties==reversed_ties==2
    np.testing.assert_array_equal(np.array([a,b])[first],a)
    np.testing.assert_array_equal(np.array([b,a])[reversed_first],b)


def test_native_visual_anchor_constrains_baseline_without_fixing_second_pose(tmp_path):
    """Synthetic gauge unit check; no inference about the real-data failure."""
    import os
    root=Path(__file__).resolve().parents[1]
    runtime=root/'output/orbslam3_standalone';binary=runtime/'build_diagnostics/replay_visual_ba'
    if not binary.exists():pytest.skip('Optional native visual replay absent')
    rng=np.random.default_rng(12);X=rng.uniform([-1,-1,4],[1,1,7],(35,3))
    R=cv2.Rodrigues(np.array([.01,.02,-.015]))[0];C=np.array([1.,.03,.01]);t=-R@C
    poses=[dict(id=0,type='pose',fixed=True,marginalized=False,Rcw=np.eye(3).ravel().tolist(),tcw=[0,0,0]),
           dict(id=1,type='pose',fixed=False,marginalized=False,Rcw=R.ravel().tolist(),tcw=t.tolist())]
    vertices=poses+[dict(id=i+2,type='point',fixed=False,marginalized=True,xyz=(x+[.005,.004,.01]).tolist()) for i,x in enumerate(X)]
    edges=[]
    for i,x in enumerate(X):
        for j,centre in enumerate([np.zeros(3),np.array([1.,0,0])]):
            q=x-centre;xy=[400*q[0]/q[2]+400,400*q[1]/q[2]+300]
            edges.append(dict(point_vertex=i+2,pose_vertex=j,xy=xy,information=[1,0,0,1],level=0,huber_delta=0,camera=[400,400,400,300]))
    graph=tmp_path/'graph.json';graph.write_text(json.dumps(dict(vertices_insertion_order=vertices,edges_insertion_order=edges,lambda_init=-1)))
    Path(str(graph)+'.result.json').write_text(json.dumps(dict(iterations_actual=30)))
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(binary.parent)+':'+str(runtime/'prefix/lib')
    target=tmp_path/'result.json'
    subprocess.run([str(binary),str(graph),str(target),'--anchor-scale'],check=True,env=env,capture_output=True)
    result=json.loads(target.read_text());pose=next(v for v in result['vertices'] if int(v['id'])==1)
    refined_R=np.array(pose['Rcw'],float).reshape(3,3);refined_t=np.array(pose['tcw'],float)
    assert np.linalg.norm(refined_t)==pytest.approx(np.linalg.norm(C),rel=1e-5)
    assert np.linalg.norm(refined_R-R)>.01


def test_point_identity_rejects_non_mutual_overlap_after_fusion():
    def point(ids):return dict(observations=[dict(source_frame_id=i,feature=0) for i in ids])
    a=[point([1,2]),point([3,4,5,6,7,8,9])]
    b=[point([1,2,3,4,5]),point([6,7,8,9])]
    assert match_points(a,b)==[(1,1)]
