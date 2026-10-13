# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import shutil
import subprocess
import tempfile


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def safe_delete_folder(folder, extensions=None, keep_folder=False, recursive=False):
    """Deletion utility that will only delete files with matching extension,
    making it safer than shutil.rmtree() which could delete some important
    system files on accident."""
    if extensions is None:
        extensions = [".jpg", ".mp4", ".wav", ".mp3", ".png"]
    if not os.path.exists(folder):
        return
    found = os.listdir(folder)
    for item in found:
        full_path = os.path.join(folder, item)
        if item.endswith(tuple(extensions)):
            os.remove(full_path)
        if recursive and os.path.isdir(full_path):
            safe_delete_folder(full_path, extensions, keep_folder, recursive)
    if not keep_folder:
        if len(os.listdir(folder)) == 0:
            os.rmdir(folder)
        else:
            print(
                "Warning, was unable to remove directory %s because it wasn't empty"
                % folder
            )


def find_ffmpeg(home, ffmpeg_binary=None):
    options = [
        "/usr/bin/ffmpeg",  # Better version on AWS than conda
        "ffmpeg",  # Normal installation will have this in the path (e.g. local machine)
        "/usr/local/fbprojects/fb-motion2/ffmpeg/bin/ffmpeg",
        "/usr/local/fbprojects/ffmpeg-ref/ffmpeg/bin/ffmpeg",
        "%s/ffmpeg/ffmpeg/bin/ffmpeg" % home,
    ]  # Devserver README instruction location.
    if ffmpeg_binary is None:
        for option in options:
            if cmd_exists(option):
                ffmpeg_binary = option
                break
    else:
        assert cmd_exists(ffmpeg_binary), "Provided --ffmpeg_binary does not exist"

    if ffmpeg_binary is None:
        raise IOError("Cannot find ffmpeg; see the README for installation help")
    return ffmpeg_binary


def get_video_codec(ffmpeg_binary=None):
    ffmpeg_binary = find_ffmpeg(os.path.expanduser("~"), ffmpeg_binary)
    result = subprocess.check_output(
        [ffmpeg_binary, "-hide_banner", "-encoders"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    if "libx264" not in result:
        raise ValueError("FFmpeg needs the libx264 video encoder to write MP4")
    return "libx264"


def make_mp4(
    input_dir,
    framerate,
    ffmpeg_binary=None,
    output_dir=None,
    image_glob="image*.png",
    output_name="out.mp4",
    crf=18,
    preset="slow",
) -> str:
    """Create a mp4 from a directory of images. Output to the input_dir by default."""
    ffmpeg_binary = find_ffmpeg(os.path.expanduser("~"), ffmpeg_binary)
    if output_dir is None:
        output_dir = input_dir
    mp4_path = os.path.join(output_dir, output_name)
    video_codec = get_video_codec(ffmpeg_binary)
    os.makedirs(output_dir, exist_ok=True)
    # Keep a prior completed video if encoding fails. Avoid shell interpretation
    # of recording names, and let x264 choose a level valid for the resolution/FPS.
    fd, temporary = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
    os.close(fd)
    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(framerate),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(input_dir, image_glob),
        "-vf",
        "scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        temporary,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(temporary, mp4_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Video: {mp4_path}")
    return mp4_path
