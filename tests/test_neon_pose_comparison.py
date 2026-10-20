"""Availability is a union; coordinate continuity must not cross map boundaries."""
import pytest
from standalone.orbslam3.pose_comparison import evaluate


def pose(i,world='a',metric='arbitrary_scale'):
    return dict(run_id='trial',frame_id=i,world_id=world,pose_available=True,
                metric_status=metric,quality_status='not_evaluated',tx=0,ty=0,tz=0,
                qx=0,qy=0,qz=0,qw=1)


def frames(n):return [dict(frame_id=i,time_ns=1791958200000000000+i*100000001) for i in range(n)]


def test_union_singletons_and_world_boundaries():
    rows=[pose(0),pose(1),pose(1,'b'),pose(2,'b'),pose(3,'b'),pose(5,'c')]
    r=evaluate(rows,frames(6))
    assert r['registered_unique']==5 and r['pose_rows']==6
    assert r['largest_single_world']==3 and r['independent_worlds']==3
    assert r['longest_contiguous']['frames']==3
    assert r['longest_contiguous']['duration_s']==pytest.approx(.200000002)
    assert r['missing_intervals']==[[4,4]]
    assert r['quality_counts']['pass']==0


def test_no_pose_is_not_execution_failure_and_metric_status_separate():
    r=evaluate([],frames(4));assert r['output_rate']==0 and r['longest_contiguous'] is None
    r=evaluate([pose(0),pose(1,metric='imu_metric_candidate')],frames(4))
    assert r['registered_unique']==2 and r['metric_candidate_unique']==1


def test_invalid_pose_and_duplicate_world_rejected():
    with pytest.raises(ValueError,match='duplicate'):evaluate([pose(0),pose(0)],frames(2))
    bad=pose(0);bad['qw']=0
    with pytest.raises(ValueError,match='quaternion'):evaluate([bad],frames(2))
    with pytest.raises(ValueError,match='Unknown'):evaluate([pose(4)],frames(2))


def test_orb_adapter_keeps_nonmetric_singletons_and_separates_runtime(tmp_path):
    import csv
    from standalone.orbslam3.pose_comparison import collect_orb
    folder=tmp_path/'O_01'/'slam';folder.mkdir(parents=True)
    with (folder/'camera_trajectory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['slam_time_s','map_id','metric','tx','ty','tz','qx','qy','qz','qw']);w.writeheader()
        w.writerows([dict(slam_time_s='0.1',map_id='0',metric='0',tx=0,ty=0,tz=0,qx=0,qy=0,qz=0,qw=1),dict(slam_time_s='0.2',map_id='1',metric='1',tx=0,ty=0,tz=0,qx=0,qy=0,qz=0,qw=1)])
    (folder/'states.csv').write_text('state,imu_initialized\n1,0\n2,1\n')
    inputs=[dict(frame_id=i,time_ns=1791958200000000000+i*100000000,slam_time_s=str(.1+.1*i)) for i in range(2)]
    rows,info=collect_orb(folder.parent,inputs,'recording')
    r=evaluate(rows,inputs)
    assert r['registered_unique']==2 and r['independent_worlds']==2
    assert r['longest_contiguous']['frames']==1 and r['metric_candidate_unique']==1
    assert rows[0]['source_timestamp']==1791958200000000000
    assert info['runtime_metric_ok']==1 and info['runtime_tracking_ok']==1


def test_colmap_adapter_preserves_original_integer_time_and_checks_identity(tmp_path):
    import json
    from standalone.orbslam3.pose_comparison import collect_colmap
    target=tmp_path/'C0_01';folder=target/'mapping_auto_1'/'0';folder.mkdir(parents=True)
    (folder.parent/'summary.json').write_text(json.dumps(dict(models=[])))
    path=folder/'visual_camera_to_world.csv'
    path.write_text('frame_id,time_ns,tx,ty,tz,qx,qy,qz,qw\n0,1791958200000000001,0,0,0,0,0,0,1\n')
    inputs=[dict(frame_id=0,time_ns='1791958200000000001')]
    rows,_=collect_colmap(target,inputs,'recording')
    assert rows[0]['source_timestamp']=='1791958200000000001'
    inputs[0]['time_ns']='1791958200000000002'
    with pytest.raises(ValueError,match='timestamp mismatch'):collect_colmap(target,inputs,'recording')
