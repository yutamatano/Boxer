"""Guard the diagnostic experiment's identities and geometry interpretation."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from standalone.orbslam3.compare_imu_initialization import (
    numeric_difference, max_estimate_error, scale_distances,
)
from standalone.orbslam3.benchmark_neon_correspondences import (
    DTYPE, deduplicate, flow, geometry, read_features,
)
from standalone.orbslam3.instrument import instrument


def test_snapshot_cli_requires_diagnostic_binary(tmp_path):
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,str(root/'standalone/orbslam3/run.py'),
                      '--output',str(tmp_path/'unused'),'--imu-snapshots'],capture_output=True,text=True)
    assert p.returncode==2
    assert '--imu-snapshots requires --diagnostics' in p.stderr
    assert not (tmp_path/'unused').exists()


def test_optimizer_capture_occurs_after_actual_edge_addition():
    root=Path(__file__).resolve().parents[1]
    p=root/'output/orbslam3_standalone/source/src/Optimizer.cc'
    if not p.exists():pytest.skip('Pinned native sources not installed')
    original=p.read_text();generated=instrument('Optimizer',original)
    begin=generated.index('void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg,')
    end=generated.index('void Optimizer::InertialOptimization(',begin+10)
    part=generated[begin:end]
    assert part.index('diagnosticSnapshot.Edge(ei)')<part.index('diagnosticSnapshot.Capture(')<part.index('optimizer.initializeOptimization()')
    assert part.index('optimizer.optimize(its)')<part.index('diagnosticSnapshot.Result(optimizer)')
    assert 'vpKFs.front()' in part and 'int its = 200;' in part
    assert p.read_text()==original


def test_scale_comparison_uses_input_distance():
    graph={'vertices_insertion_order':[{'id':'5','twb':['0','0','0']},{'id':'8','twb':['3','0','0']}],
           'keyframes_map_order':[{'kf_id':'8','time_s':'2','source_frame_id':'102'},
                                  {'kf_id':'5','time_s':'1','source_frame_id':'101'}]}
    result={'vertices':[{'type':'scale','value':'2'}]}
    rows=scale_distances(graph,result)
    assert rows==[dict(source_a=101,source_b=102,before=3.,scale=2.,after_scale_only=6.)]
    assert numeric_difference({'a':['1','2']},{'a':['1','3']})['path']=='/a/1'


def test_replay_comparison_checks_all_variables():
    a={'vertices':[{'id':1,'type':'velocity','fixed':False,'value':['1','2','3']}]}
    b=json.loads(json.dumps(a));b['vertices'][0]['value'][2]='3.25'
    assert max_estimate_error(a,b)==.25


def test_multiscale_points_not_counted_twice():
    f=np.zeros(4,dtype=DTYPE);f['xy']=[[20,20],[25,20],[60,60],[120,120]];f['octave']=[0,4,2,0]
    kept=deduplicate(f)
    assert len(kept)==3
    assert not np.any(kept['octave']==4)


def test_corrupt_feature_cache_rejected(tmp_path):
    p=tmp_path/'bad.bin';p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='Truncated'):
        read_features(p)


def test_common_geometry_accepts_translation_with_depth():
    rng=np.random.default_rng(12)
    X=np.column_stack([rng.uniform(-2,2,250),rng.uniform(-1.5,1.5,250),rng.uniform(3,8,250)])
    K=np.array([[440.,0,400],[0,440.,300],[0,0,1.]])
    def project(x):
        p=x@K.T
        return np.asarray(p[:,:2]/p[:,2:],np.float32)
    a=project(X);b=project(X+[.25,0,0])
    result,valid=geometry(a,b,np.ones(len(a)),np.ones(len(b)),K,(600,800))
    assert result['triangulated']>200
    assert result['parallax_deg']>1
    assert len(valid)==result['triangulated']


def test_klt_tracks_translation_and_checks_reverse():
    rng=np.random.default_rng(4)
    image=rng.integers(0,256,(160,200),dtype=np.uint8)
    moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,2]]),(200,160))
    points=cv2.goodFeaturesToTrack(image,150,.01,7).reshape(-1,2)
    target,idx=flow(image,moved,points)
    assert len(idx)>60
    assert np.median(np.linalg.norm(target-points[idx]-[3,2],axis=1))<.2


def test_real_graph_replay_matches_native_solver_when_capture_installed(tmp_path):
    """Integration check using the local real-data capture, never synthetic priors."""
    import hashlib
    import os
    from standalone.orbslam3.compare_imu_initialization import read
    root=Path(__file__).resolve().parents[1]
    capture=root/'output/neon_Htou_optimizer_audit_02/trial_01/slam/imu_snapshot_1.json'
    runtime=root/'output/orbslam3_standalone'
    binary=runtime/'build_diagnostics/replay_imu_initialization'
    if not capture.exists() or not binary.exists():
        pytest.skip('Optional native real-data capture not installed')
    execution=read(capture.parents[1]/'execution.json')
    if execution['library_sha256']!=hashlib.sha256((binary.parent/'libstandalone_orb.so').read_bytes()).hexdigest():
        binary=runtime/'frozen_optimizer_audit_02/replay_imu_initialization'
        if not binary.exists() or execution['library_sha256']!=hashlib.sha256((binary.parent/'libstandalone_orb.so').read_bytes()).hexdigest():
            pytest.skip('Matching frozen native capture build not installed')
    graph=read(capture)
    for edge in graph['edges_insertion_order']:
        if edge['type'].startswith('prior'):
            assert np.array_equal(np.asarray(edge['measurement'],float),np.zeros(3))
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(binary.parent)+':'+str(runtime/'prefix/lib')+':'+env.get('LD_LIBRARY_PATH','')
    outputs=[]
    for i in range(2):
        path=tmp_path/f'replay_{i}.json'
        subprocess.run([str(binary),str(capture),str(path)],check=True,env=env)
        outputs.append(read(path))
    original=read(str(capture)+'.result.json')
    assert all(max_estimate_error(original,r)<1e-9 for r in outputs)
    assert max_estimate_error(*outputs)==0


def test_learned_benchmark_refuses_cpu_fallback(tmp_path):
    import os
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.run([sys.executable,str(root/'standalone/orbslam3/benchmark_neon_lightglue.py'),
                      '--classical',str(tmp_path/'absent'),'--prepared',str(tmp_path/'absent'),
                      '--runtime',str(tmp_path/'runtime'),'--output',str(tmp_path/'result')],
                     env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'CUDA is required; CPU fallback is forbidden' in p.stderr
    assert not (tmp_path/'result').exists()
