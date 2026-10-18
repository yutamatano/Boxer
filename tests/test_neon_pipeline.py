"""Pipeline boundaries that must not join unrelated maps or incompatible modes."""

from types import SimpleNamespace

import run_neon_pipeline as pipeline
from utils.neon_vio import recording_path


def test_boxer_intervals_have_separate_outputs_and_fusion_stage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "run_logged", lambda command, log, **kwargs: calls.append(command)
    )
    summary = {
        "recording": str(tmp_path / "recording"),
        "segments": [
            {"geometry": str(tmp_path / "a.npz"), "start_n": 30, "max_n": 40},
            {"geometry": str(tmp_path / "b.npz"), "start_n": 90, "max_n": 50},
        ],
    }
    args = SimpleNamespace(labels="chair", force_cpu=True, fuse=True,
                           gaze_targets=True, gaze_max_delta_ms=5., point_cloud="off")
    pipeline.run_boxer_segments(summary, tmp_path, args)
    assert len(calls) == 4
    for i, start in enumerate((30, 90)):
        tracking, fusion = calls[2 * i : 2 * i + 2]
        assert "--track" in tracking and "--fuse" not in tracking
        assert "--gaze_targets" in tracking
        assert tracking[tracking.index("--gaze_max_delta_ms") + 1] == 5.
        assert tracking[tracking.index("--point_cloud") + 1] == "off"
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


def test_untracked_intervals_preserve_source_gaps_and_reject_overlap():
    import pytest

    assert pipeline.uncovered_intervals(range(9), [{"start_n": 2, "max_n": 3}]) == [
        {"start_n": 0, "max_n": 2},
        {"start_n": 5, "max_n": 4},
    ]
    assert pipeline.uncovered_intervals([0, 1, 3, 4], []) == [
        {"start_n": 0, "max_n": 2},
        {"start_n": 3, "max_n": 2},
    ]
    with pytest.raises(ValueError, match="overlap"):
        pipeline.uncovered_intervals(
            range(9),
            [
                {"start_n": 2, "max_n": 3},
                {"start_n": 4, "max_n": 2},
            ],
        )


def test_frame_estimates_never_receive_world_poses_or_tracking(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "run_logged", lambda command, log, **kwargs: calls.append(command)
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "frames.csv").write_text("frame_id\n0\n1\n2\n3\n4\n")
    summary = {
        "recording": str(tmp_path / "recording"),
        "prepared": str(prepared),
        "segments": [{"start_n": 2, "max_n": 2}],
        "usable_frames": 2,
    }
    pipeline.run_frame_estimates(
        summary, tmp_path, SimpleNamespace(labels="chair", force_cpu=False, gaze_targets=True)
    )
    assert len(calls) == 2
    for command in calls:
        assert "--neon_geometry" not in command
        assert "--gaze_targets" in command
        assert "--track" not in command and "--fuse" not in command
        assert command[command.index("--point_cloud") + 1] == "off"
    assert summary["total_estimated_frames"] == 5
    assert summary["frame_only_frames"] == 3
    assert all(
        not r["camera_trajectory_available"] for r in summary["frame_only_intervals"]
    )
