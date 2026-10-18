"""GPU-only requests must fail before loading data on a CPU-only machine."""

import sys
from types import SimpleNamespace

import pytest
import torch

import run_boxer
import run_neon_pipeline as pipeline
import view_tracker
from utils.viewer_3d import configure_gpu_viewer, validate_gpu_renderer


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
    monkeypatch.setattr(pipeline, "run_logged", lambda command, log, **kwargs: calls.append(command))
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


def test_viewer_requires_cuda_before_loading_data(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["view_tracker.py", "--input", "missing", "--require_gpu"])
    with pytest.raises(SystemExit) as exc:
        view_tracker.main()
    assert exc.value.code == 2
    assert "CPU fallback is disabled" in capsys.readouterr().err


def test_gpu_viewer_selects_nvidia_glx(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    configure_gpu_viewer()
    import os
    assert os.environ["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"


@pytest.mark.parametrize("renderer", ["llvmpipe (LLVM 20)", "softpipe", "GDI Generic", ""])
def test_gpu_viewer_rejects_software_gl(renderer):
    with pytest.raises(RuntimeError, match="Hardware OpenGL is required"):
        validate_gpu_renderer(SimpleNamespace(info={"GL_RENDERER": renderer}))


def test_gpu_viewer_accepts_nvidia_gl():
    validate_gpu_renderer(SimpleNamespace(info={"GL_RENDERER": "NVIDIA GeForce RTX 3070"}))


def test_pipeline_viewer_command_keeps_gpu_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "run_boxer_segments", lambda *args: None)
    monkeypatch.setattr(pipeline, "run_frame_estimates", lambda *args: None)
    monkeypatch.setattr(pipeline, "assemble_replay", lambda *args: tmp_path / "viewer")
    summary = {"recording": "recording"}
    pipeline.run_inference(summary, tmp_path, SimpleNamespace(require_gpu=True))
    assert "--require_gpu" in summary["viewer_command"]
