"""Compile the adapter's stop policy without requiring an ORB-SLAM3 build."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_recovery_keeps_map_until_reset_or_terminal_loss(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "policy.cc"
    source.write_text(
        '#include "neon_tracking_policy.h"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "  using neon::shouldFinalizeMetricMap;\n"
        "  assert(!shouldFinalizeMetricMap(2, true, false));\n"
        "  assert(!shouldFinalizeMetricMap(3, true, false));\n"
        "  assert(shouldFinalizeMetricMap(3, true, true));\n"
        "  assert(shouldFinalizeMetricMap(2, true, true));\n"
        "  assert(shouldFinalizeMetricMap(4, true, false));\n"
        "  assert(shouldFinalizeMetricMap(1, false, false));\n"
        "  assert(shouldFinalizeMetricMap(2, false, false));\n"
        "  assert(shouldFinalizeMetricMap(3, false, false));\n"
        "  using neon::nextMetricSessionIndex;\n"
        "  assert(nextMetricSessionIndex(391, true, 340) == 341);\n"
        "  assert(nextMetricSessionIndex(391, false, 340) == 391);\n"
        "  assert(nextMetricSessionIndex(341, true, 340) == 341);\n"
        "}\n"
    )
    binary = tmp_path / "policy"
    include = Path(__file__).resolve().parents[1] / "slam"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True)
