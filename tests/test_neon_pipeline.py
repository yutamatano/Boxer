"""Pipeline boundaries that must not join unrelated maps or incompatible modes."""

from types import SimpleNamespace

import run_neon_pipeline as pipeline
from utils.neon_vio import recording_path


def test_boxer_intervals_have_separate_outputs_and_fusion_stage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "run_logged", lambda command, log: calls.append(command)
    )
    summary = {
        "recording": str(tmp_path / "recording"),
        "segments": [
            {"geometry": str(tmp_path / "a.npz"), "start_n": 30, "max_n": 40},
            {"geometry": str(tmp_path / "b.npz"), "start_n": 90, "max_n": 50},
        ],
    }
    args = SimpleNamespace(labels="chair", force_cpu=True, fuse=True)
    pipeline.run_boxer_segments(summary, tmp_path, args)
    assert len(calls) == 4
    for i, start in enumerate((30, 90)):
        tracking, fusion = calls[2 * i : 2 * i + 2]
        assert "--track" in tracking and "--fuse" not in tracking
        assert tracking[tracking.index("--start_n") + 1] == start
        assert (
            tracking[tracking.index("--output_dir") + 1] == tmp_path / f"boxer_{i:03d}"
        )
        assert fusion[1:3] == ["-m", "utils.fuse_3d_boxes"]
        assert str(tmp_path / f"boxer_{i:03d}") in str(fusion[-1])
    assert summary["segments"][0]["results"] != summary["segments"][1]["results"]


def test_resolve_single_imported_recording_without_guessing(tmp_path):
    import pytest

    nested = tmp_path / "recording"
    nested.mkdir()
    (nested / "Neon Scene Camera v1 ps1.mp4").touch()
    assert recording_path(str(tmp_path)) == nested
    second = tmp_path / "other_recording"
    second.mkdir()
    (second / "Neon Scene Camera v1 ps1.mp4").touch()
    with pytest.raises(ValueError, match="Choose one"):
        recording_path(str(tmp_path))
