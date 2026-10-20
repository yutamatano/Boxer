"""Safety and geometry contracts for bounded map-connection experiments."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from standalone.orbslam3.neon_sfm import image_pairs,database
from standalone.orbslam3.neon_map_connect import robust_fit
from standalone.orbslam3.instrument import instrument
from pathlib import Path


def test_pair_manifest_shared_and_validated(tmp_path):
    frames=[dict(frame_id=i) for i in range(12)]
    pairs,sha=image_pairs(tmp_path,frames)
    assert (0,5) in pairs and (0,6) not in pairs
    p=tmp_path/'pair_manifest.json';p.write_text(json.dumps(dict(pairs=[[0,6],[1,8]])))
    extra,new_sha=image_pairs(tmp_path,frames)
    assert extra==[(0,6),(1,8)] and new_sha!=sha
    p.write_text(json.dumps(dict(pairs=[[0,6],[0,6]])))
    with pytest.raises(ValueError,match='Invalid'):image_pairs(tmp_path,frames)


def test_database_cache_rejects_changed_pair_manifest(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'pycolmap',SimpleNamespace())
    (tmp_path/'pair_manifest.json').write_text(json.dumps(dict(pairs=[[0,6]])))
    (tmp_path/'database_done.json').write_text(json.dumps(dict(pair_manifest_sha256='old')))
    with pytest.raises(ValueError,match='manifest changed'):database(tmp_path,[dict(frame_id=i) for i in range(7)],tmp_path)


def test_sim3_ransac_resists_correspondence_outliers():
    rng=np.random.default_rng(5);a=rng.normal(size=(100,3));b=2*a+[1,3,4];b[-20:]=rng.normal(0,30,(20,3))
    fit=robust_fit(a,b)
    assert fit is not None and fit['inlier_count']>=80
    assert fit['scale']==pytest.approx(2,rel=1e-8)
    np.testing.assert_allclose(fit['translation'],[1,3,4],atol=1e-8)


def test_tie_patch_preserves_normals_and_does_not_allocate_candidate_keys():
    path=Path('output/orbslam3_standalone/source/src/MapPoint.cc')
    if not path.exists():pytest.skip('Optional pinned sources absent')
    original=path.read_text();result=instrument('MapPoint',original)
    assert 'stableKey(i)<stableKey(BestIdx)' in result
    assert 'candidateKeys' not in result
    assert result[result.index('void MapPoint::UpdateNormalAndDepth()'):]==original[original.index('void MapPoint::UpdateNormalAndDepth()'):]


def test_partial_database_import_reverifies_missing_geometry(tmp_path,monkeypatch):
    """Interrupted after raw matching must not silently omit geometric checks."""
    import shutil
    from standalone.orbslam3.neon_sfm import read_rows
    root=Path(__file__).resolve().parents[1]
    deps=root/'output/neon_sfm_runtime/deps';source=root/'output/neon_Htou_pair_import_verify_01'
    if not deps.exists() or not (source/'database.db').exists():pytest.skip('Optional real-data DB absent')
    monkeypatch.syspath_prepend(str(deps));p=pytest.importorskip('pycolmap')
    for name in ['database.db','pair_manifest.json','database_seed.json']:shutil.copy2(source/name,tmp_path/name)
    with p.Database.open(str(tmp_path/'database.db')) as db:
        images={int(im.name[6:-4]):im for im in db.read_all_images()};ia,ib=images[26168].image_id,images[26178].image_id
        original=db.read_matches(ia,ib).copy();db.delete_two_view_geometry(ia,ib)
        assert db.exists_matches(ia,ib) and not db.exists_two_view_geometry(ia,ib)
    prepared=root/'output/neon_Htou_standalone_03/prepared'
    frames=[r for r in read_rows(prepared/'frames.csv') if 25834<=int(r['frame_id'])<=26733]
    database(tmp_path,frames,prepared)
    result=json.loads((tmp_path/'database_done.json').read_text())
    assert result['imported_pairs']==0 and result['verified_pairs']==1
    with p.Database.open(str(tmp_path/'database.db')) as db:
        assert db.exists_two_view_geometry(ia,ib)
        assert len(db.read_two_view_geometry(ia,ib).inlier_matches)>0
        np.testing.assert_array_equal(original,db.read_matches(ia,ib))
