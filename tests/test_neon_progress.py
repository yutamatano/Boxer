"""Progress reflects completed frames, including worker failures and restarts."""

import sys

import pytest

from run_neon_pipeline import run_logged
from utils.neon_progress import StageProgress


def test_cr_tqdm_is_local_and_cumulative(tmp_path, capsys):
    log = tmp_path / "boxer.log"
    log.write_text("startup\n\rBoxerNet:  50%|x | 2/4 [00:01<00:01]\r")
    progress = StageProgress(tmp_path, "推論", 10, offset=3, source="inference", interval=0)
    progress.poll(log)
    text = capsys.readouterr().out
    assert "50.0% (5/10)" in text
    assert "区間 50.0% (2/4)" in text
    assert (tmp_path / "progress.log").read_text() == text


def test_slam_counts_rows_not_source_ids_and_does_not_finish_early(tmp_path, capsys):
    states = tmp_path / "states.csv"
    states.write_text("frame_id,state\n150,2\n151,2\n")
    progress = StageProgress(tmp_path, "SLAM", 10, offset=4, source="slam", states=states, interval=0)
    progress.poll(None)
    with states.open("a") as stream:
        stream.write("153,2\n")
    progress.poll(None)
    progress.finish(True)
    assert progress.current == 7
    text = capsys.readouterr().out
    assert "60.0% (6/10)" in text
    assert "70.0% (7/10) セッション完了" in text
    assert "100.0%" not in text


@pytest.mark.parametrize("code", [0, 3, 1])
def test_real_child_log_is_preserved_and_errors_do_not_reach_100(tmp_path, code, capsys):
    log = tmp_path / "prepare.log"
    source = f"print('Prepared 1/2 frames', flush=True); raise SystemExit({code})"
    progress = StageProgress(tmp_path, "前処理", None, source="prepare")
    command = [sys.executable, "-c", source]
    if code == 1:
        with pytest.raises(RuntimeError, match="Command failed"):
            run_logged(command, log, progress=progress)
    else:
        assert run_logged(command, log, allowed=(0, 3), progress=progress) == code
    assert log.read_text().strip() == "Prepared 1/2 frames"
    text = capsys.readouterr().out
    if code == 1:
        assert "50.0% (1/2) 失敗" in text
        assert "100.0%" not in text
    else:
        assert "100.0% (2/2) 完了" in text


def test_progress_is_visible_while_child_is_running(tmp_path, monkeypatch):
    log = tmp_path / "prepare.log"
    progress = StageProgress(tmp_path, "前処理", None, source="prepare", interval=0)
    original = progress.poll
    observed_running = []

    def poll(path, **kwargs):
        original(path, **kwargs)
        if progress.current == 1:
            observed_running.append(True)
            (tmp_path / "continue").touch()

    monkeypatch.setattr(progress, "poll", poll)
    code = (
        "from pathlib import Path; import time; "
        "print('Prepared 1/2 frames',flush=True); "
        f"p=Path({str(tmp_path / 'continue')!r}); "
        "start=time.monotonic()\n"
        "while not p.exists() and time.monotonic()-start<8: time.sleep(.05)\n"
        "assert p.exists(), 'parent did not report live progress'\n"
        "print('Prepared 2/2 frames',flush=True)"
    )
    run_logged([sys.executable, "-c", code], log, progress=progress)
    assert observed_running


def test_stage_context_records_failure(tmp_path, capsys):
    with pytest.raises(ValueError):
        with StageProgress(tmp_path, "結果統合", 1):
            raise ValueError("broken input")
    text = capsys.readouterr().out
    assert "0.0% (0/1) 失敗" in text
    assert "100.0%" not in text
