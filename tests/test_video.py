"""Real encoder checks for Remote SSH result artifacts."""

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from utils.video import make_mp4

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)


def test_mp4_handles_spaces_and_limits_resolution(tmp_path):
    frames = tmp_path / "frames with spaces"
    frames.mkdir()
    for i in range(2):
        cv2.imwrite(
            str(frames / f"image{i}.png"), np.full((960, 2880, 3), i * 100, np.uint8)
        )
    output = make_mp4(
        str(frames), 10, output_name="result with spaces.mp4", preset="ultrafast"
    )
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", output]
        )
    )["streams"][0]
    assert (info["width"], info["height"], info["pix_fmt"]) == (1920, 640, "yuv420p")
    assert int(info["nb_frames"]) == 2
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", output, "-f", "null", "-"], check=True
    )


def test_failed_encode_preserves_existing_video(tmp_path):
    target = tmp_path / "out.mp4"
    target.write_bytes(b"previous completed video")
    with pytest.raises(subprocess.CalledProcessError):
        make_mp4(str(tmp_path), 10)
    assert target.read_bytes() == b"previous completed video"
    assert list(tmp_path.iterdir()) == [target]
