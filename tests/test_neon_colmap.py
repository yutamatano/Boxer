"""COLMAP evaluation must not hide failures or relabel arbitrary scale as metric."""

import numpy as np
import pytest

from run_neon_colmap import split_frames, summarize_errors


def test_training_holdout_split_preserves_integer_source_ids():
    ids = np.arange(0, 618, 3, dtype=np.int64)
    train, holdout = split_frames(ids, 5)
    assert len(train) == 165 and len(holdout) == 41
    assert set(train).isdisjoint(holdout)
    np.testing.assert_array_equal(np.sort(np.r_[train, holdout]), ids)
    assert train.dtype == holdout.dtype == np.int64
    with pytest.raises(ValueError):
        split_frames(ids, 1)


def test_reprojection_summary_counts_failed_projections():
    result = summarize_errors([1, 2, np.inf, np.nan])
    assert result["count"] == 4 and result["finite_count"] == 2
    assert result["fraction_within_4px"] == 0.5
    assert result["median_px"] == 1.5
    assert summarize_errors([])["median_px"] is None
    assert summarize_errors([np.inf])["fraction_within_4px"] == 0
