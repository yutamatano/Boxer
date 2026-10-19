"""Verify diagnostic identity/epochs and fail-closed pinned source injection."""
import csv
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from standalone.orbslam3.instrument import instrument, replace

ROOT=Path(__file__).resolve().parents[1]


def test_changed_upstream_location_fails_closed():
    with pytest.raises(ValueError,match='location changed'):
        replace('changed source','missing marker','diagnostic')
    with pytest.raises(ValueError,match='location changed'):
        replace('marker marker','marker','diagnostic')


@pytest.mark.parametrize('name',['Tracking','TwoViewReconstruction','System','Atlas','LocalMapping','LoopClosing'])
def test_pinned_instrumentation_preserves_source(name):
    path=ROOT/'output/orbslam3_standalone/source/src'/f'{name}.cc'
    if not path.exists():pytest.skip('Pinned upstream checkout not installed')
    original=path.read_text()
    generated=instrument(name,original)
    assert path.read_text()==original
    assert 'diagnostics.h' in generated
    if name=='Tracking':
        assert 'mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0' in generated
        assert 'if(nmatches<100)' in generated
        assert 'mCurrentFrame.mvKeys.size()>100' in generated
        assert 'CURRENT_KEYPOINTS_LOW' in generated and 'PAIR_TIMEOUT' in generated
    if name=='System':
        assert 'if(!mbResetActiveMap) neon_diag::RequestReset();' in generated


def test_native_log_uses_source_ids_and_counts_actual_resets(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    source=ROOT/'standalone/orbslam3'
    binary=tmp_path/'test_diagnostics'
    subprocess.run([compiler,'-std=c++14','-pthread','-I',str(source),
                    str(source/'diagnostics.cc'),str(source/'test_diagnostics.cc'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(tmp_path)],check=True)
    rows=list(csv.DictReader((tmp_path/'init_diagnostics.csv').open()))
    assert [int(r['source_frame_id']) for r in rows]==[9988,9989,9990]
    assert rows[0]['initial_reference_frame_id']==''
    assert rows[1]['initial_reference_frame_id']=='9988'
    assert rows[1]['triangulated_points']=='60'
    assert rows[1]['geometry_result']=='PARALLAX_LOW'
    assert rows[2]['initialization_result']=='TRACKING'
    assert rows[2]['current_keypoints']=='250'
    events=[json.loads(line) for line in (tmp_path/'map_events.jsonl').read_text().splitlines()]
    applied=[e for e in events if e['event']=='RESET_APPLIED']
    assert [e['map_epoch'] for e in applied]==[1,2]
    assert [e['request_id'] for e in applied]==[1,2]
    assert [e['actual_reset_index'] for e in applied]==[1,2]
    created=[e for e in events if e['event']=='MAP_CREATED']
    assert all(e['source_frame_id'] is None for e in created)
    assert created[1]['map_epoch']==0
    state=next(e for e in events if e['event']=='TRACK_STATUS')
    assert state['queued_keyframes']==3
    assert state['source_frame_id']==9989
    estimate=next(e for e in events if e['event']=='IMU_OPTIMIZATION_ESTIMATE')
    assert estimate['gyro_bias']==[1,2,3]
    assert estimate['accel_bias']==[4,5,6]
    assert created[0]['scale_estimate'] is None
    assert next(e for e in events if e['event']=='NEGATIVE_SCALE_TEST')['scale_estimate']==-1
