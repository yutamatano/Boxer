"""Ensure online tracking consumes raw detections rather than track history."""

import sys

import pytest

import view_tracker


def test_prefers_raw_detections_over_tracked_history(tmp_path, monkeypatch):
    raw = tmp_path / "boxer_3dbbs.csv"
    tracked = tmp_path / "boxer_3dbbs_tracked.csv"
    raw.touch()
    tracked.touch()
    (tmp_path / "owl_2dbbs.csv").touch()
    loaded = []
    monkeypatch.setattr(sys, "argv", ["view_tracker.py", "--input", "nym10_gen1"])
    monkeypatch.setattr(
        view_tracker,
        "load_common",
        lambda args: (
            "sample_data/nym10_gen1", "aria", "nym10_gen1",
            str(tmp_path), "", None,
        ),
    )
    monkeypatch.setattr(
        view_tracker, "read_obb_csv", lambda path: loaded.append(path) or {}
    )
    monkeypatch.setattr(view_tracker, "build_seq_ctx", lambda *args: {})
    monkeypatch.setattr(view_tracker, "launch_viewer", lambda cls: None)
    view_tracker.main()
    assert loaded == [str(raw)]

    raw.unlink()
    with pytest.raises(IOError, match="boxer_3dbbs.csv"):
        view_tracker.main()
    assert loaded == [str(raw)]
