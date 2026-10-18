"""Read existing worker progress without changing analysis or frame selection."""

import re
import time
from datetime import datetime
from pathlib import Path


class StageProgress:
    """Report frame/interval counts to stdout and one pipeline progress log."""

    def __init__(self, out, stage, total, *, offset=0, detail="", source=None,
                 states=None, interval=5.):
        self.path = Path(out) / "progress.log"
        self.stage = stage
        self.total = total
        self.offset = offset
        self.detail = detail
        self.source = source
        self.states = Path(states) if states else None
        self.interval = interval
        self.current = offset
        self.local_current = 0
        self.local_total = None
        self._last_report = float("-inf")
        self._state_position = self._state_lines = 0

    def update(self, current, *, status="処理中", force=False):
        self.current = max(self.current, current)
        if self.total is not None:
            self.current = min(self.total, self.current)
        now = time.monotonic()
        if not force and now - self._last_report < self.interval:
            return
        self._last_report = now
        if self.total is None:
            percent = 0.
        else:
            percent = 100 * self.current / self.total if self.total else 100.
        # Avoid rounding an incomplete stage up to 100%.
        if self.total is not None and self.current < self.total:
            percent = min(percent, 99.9)
        extra = f" | {self.detail}" if self.detail else ""
        if self.local_total and self.source == "inference":
            local_percent = min(100 * self.local_current / self.local_total, 100.)
            extra += f" | 区間 {local_percent:.1f}% ({self.local_current}/{self.local_total})"
        total_text = self.total if self.total is not None else "?"
        message = (
            f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] "
            f"[{self.stage}] {percent:.1f}% ({self.current}/{total_text}) "
            f"{status}{extra}"
        )
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    def __enter__(self):
        self.update(self.offset, status="開始", force=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(exc_type is None)

    def poll(self, log, *, force=False):
        if self.source == "slam":
            if self.states.is_file():
                with self.states.open("rb") as stream:
                    stream.seek(self._state_position)
                    while data := stream.read(65536):
                        self._state_lines += data.count(b"\n")
                    self._state_position = stream.tell()
                self.local_current = max(0, self._state_lines - 1)  # CSV header
        elif self.source in ("prepare", "inference"):
            with Path(log).open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                tail = stream.read().decode("utf-8", errors="replace")
            pattern = (
                r"Prepared (\d+)/(\d+) frames"
                if self.source == "prepare"
                else r"BoxerNet:[^\r\n]*?\|\s*(\d+)/(\d+)\s*\["
            )
            matches = re.findall(pattern, tail)
            if matches:
                self.local_current, self.local_total = map(int, matches[-1])
                if self.source == "prepare":
                    self.total = self.local_total
        status = "初期化中（進捗通知待ち）" if self.local_current == 0 else "処理中"
        if self.local_total and self.local_current == self.local_total:
            status = "フレーム処理済み・終了処理中"
        self.update(self.offset + self.local_current, status=status, force=force)

    def finish(self, success, *, count=None):
        if success:
            if count is not None:
                self.local_current = count
            elif self.local_total is not None:
                self.local_current = self.local_total
            # SLAM sessions can finish before the full recording is processed.
            current = self.offset + self.local_current if self.source else self.total
            status = "セッション完了" if self.source == "slam" else "完了"
            self.update(current, status=status, force=True)
        else:
            self.update(self.current, status="失敗（工程別ログを確認）", force=True)
