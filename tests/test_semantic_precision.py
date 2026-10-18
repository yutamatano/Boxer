"""Cached detector features and newly encoded labels must mix safely."""

from types import SimpleNamespace

import pytest
import torch

from test_track_3d_boxes import _make_det
from utils import fuse_3d_boxes as fusion
from utils.track_3d_boxes import BoundingBox3DTracker, TrackState


def embedder(dtype):
    return SimpleNamespace(
        forward=lambda texts: torch.tensor([[2., 0., 0.]] * len(texts), dtype=dtype)
    )


@pytest.mark.parametrize("cached_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("new_dtype", [torch.bfloat16, torch.float32])
def test_visible_cached_and_invisible_new_label_can_merge(
    monkeypatch, cached_dtype, new_dtype
):
    monkeypatch.setattr(fusion, "_load_cached_text_embeddings", lambda: {
        "chair": torch.tensor([2., 0., 0.], dtype=cached_dtype)
    })
    tracker = BoundingBox3DTracker(verbose=False)
    tracker._embed_model = embedder(new_dtype)
    detections = torch.stack([
        _make_det([0., 0., .5], text="chair"),
        _make_det([.05, 0., .5], text="uncached label"),
    ])
    tracker.update(detections, 0)
    for track in tracker.tracks:
        track.state = TrackState.ACTIVE
    tracker.tracks[1].last_visible = False
    # This is the failing V x N operation: only cached features in V,
    # but both cached and freshly encoded features in N.
    tracker._merge_duplicate_tracks()
    assert len(tracker.tracks) == 1
    for value in tracker._embedding_cache.values():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value.norm(), torch.tensor(1.))


def test_batch_generated_tracker_features_are_float32(monkeypatch):
    monkeypatch.setattr(fusion, "_load_cached_text_embeddings", lambda: {})
    tracker = BoundingBox3DTracker(verbose=False, merge_interval=1)
    tracker._embed_model = embedder(torch.bfloat16)
    detections = torch.stack([_make_det([0., 0., .5])])
    tracker.update(detections, 0)
    tracker.update(detections, 1)
    assert tracker._embedding_cache["chair"].dtype == torch.float32
    torch.testing.assert_close(tracker._embedding_cache["chair"].norm(), torch.tensor(1.))


def test_old_caches_are_float32_and_csv_case_matches(tmp_path, monkeypatch):
    legacy = tmp_path / "a.pt"
    labelled = tmp_path / "b.pt"
    torch.save(torch.ones(1, 3), legacy)
    torch.save(
        {"prompts": ["iPod"], "embeddings": torch.ones(1, 3, dtype=torch.bfloat16)},
        labelled,
    )
    monkeypatch.setattr("glob.glob", lambda pattern: [str(labelled), str(legacy)])
    result = fusion._load_cached_text_embeddings()
    assert result["ipod"].dtype == torch.float32
    torch.testing.assert_close(result["ipod"], result["iPod"])
    assert torch.load(labelled, weights_only=False)["embeddings"].dtype == torch.bfloat16


@pytest.mark.parametrize("cache_hit", [True, False])
def test_fusion_features_have_same_normalization_and_dtype(monkeypatch, cache_hit):
    monkeypatch.setattr(fusion, "_load_cached_text_embeddings", lambda: {
        "chair": torch.tensor([2., 0., 0.], dtype=torch.bfloat16)
    } if cache_hit else {})
    monkeypatch.setattr("owl.clip_tokenizer.TextEmbedder", lambda: embedder(torch.bfloat16))
    obbs = torch.stack([_make_det([0., 0., .5])])
    result = fusion.precompute_semantic_embeddings(obbs)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result @ result.t(), torch.ones(1, 1))
