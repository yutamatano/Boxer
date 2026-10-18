"""GPU-only requests must fail before loading data on a CPU-only machine."""

import sys
from types import SimpleNamespace

import pytest
import torch

import run_boxer
import run_neon_pipeline as pipeline


@pytest.mark.parametrize("module", [run_boxer, pipeline])
@pytest.mark.parametrize("force_cpu", [False, True])
def test_require_gpu_fails_before_data_loading(module, force_cpu, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = [module.__name__, "--input", "missing", "--require_gpu"]
    if module is pipeline:
        argv += ["--output", "unused"]
    if force_cpu:
        argv += ["--force_cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    message = capsys.readouterr().err
    expected = "cannot be combined" if force_cpu else "CPU fallback is disabled"
    assert expected in message


def test_pipeline_forwards_gpu_requirement_to_all_intervals(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_logged", lambda command, log: calls.append(command))
    monkeypatch.setattr(pipeline, "selected_frame_ids", lambda path: [0, 1, 2, 3])
    summary = dict(
        recording=str(tmp_path / "recording"), prepared="unused",
        segments=[dict(start_n=1, max_n=2, geometry="geometry.npz")],
        usable_frames=2,
    )
    args = SimpleNamespace(force_cpu=False, require_gpu=True, labels="chair", fuse=False)
    pipeline.run_boxer_segments(summary, tmp_path, args)
    pipeline.run_frame_estimates(summary, tmp_path, args)
    assert len(calls) == 3
    assert all(
        "--require_gpu" in command and "--force_cpu" not in command
        for command in calls
    )
