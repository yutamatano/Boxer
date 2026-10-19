#!/usr/bin/env bash
# All generated files stay inside Boxer/output; existing results are read-only.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runtime="$task_root/output/orbslam3_standalone"
upstream="$task_root/output/neon_vio_deps/ORB_SLAM3"
revision=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
mkdir -p "$runtime"
if [[ ! -d "$runtime/source" ]]; then
    mkdir "$runtime/source"
    git -C "$upstream" archive "$revision" | tar -x -C "$runtime/source"
    # Only System's observation/export API is added. Core algorithms stay exact.
    patch -d "$runtime/source" -p1 < "$task_root/slam/export_neon.patch"
    cp -a "$task_root/output/neon_vio_deps/prefix/usr" "$runtime/prefix"
    printf '%s\n' "$revision" > "$runtime/upstream_revision.txt"
fi
for source_file in Tracking.cc LocalMapping.cc Optimizer.cc ORBextractor.cc ORBmatcher.cc LoopClosing.cc TwoViewReconstruction.cc KeyFrame.cc; do
    git -C "$upstream" show "$revision:src/$source_file" | cmp - "$runtime/source/src/$source_file"
done
if [[ ! -f "$runtime/source/Vocabulary/ORBvoc.txt" ]]; then
    tar -xzf "$runtime/source/Vocabulary/ORBvoc.txt.tar.gz" -C "$runtime/source/Vocabulary"
fi
if [[ ! -x "$runtime/.venv/bin/python" ]]; then
    python3 -m venv --without-pip "$runtime/.venv"
fi
if ! "$runtime/.venv/bin/python" -c 'import numpy' 2>/dev/null; then
    # A wheel path allows a fully offline installation too.
    uv --no-config --cache-dir "$runtime/uv-cache" pip install \
        --python "$runtime/.venv/bin/python" "${NEON_NUMPY_WHEEL:-numpy==2.5.2}"
fi
task_diagnostics="${NEON_DIAGNOSTICS:-OFF}"
case "${task_diagnostics^^}" in
    ON|TRUE|YES|Y|1) task_diagnostics=ON ;;
    OFF|FALSE|NO|N|0) task_diagnostics=OFF ;;
    *) echo "NEON_DIAGNOSTICS must be ON or OFF" >&2; exit 2 ;;
esac
task_build="$runtime/build"
if [[ "$task_diagnostics" == ON ]]; then task_build="$runtime/build_diagnostics"; fi
cmake -S "$task_root/standalone/orbslam3" -B "$task_build" \
    -DORB_SLAM3_ROOT="$runtime/source" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native' \
    -DNEON_ORPHAN_GUARD="${NEON_ORPHAN_GUARD:-OFF}" \
    -DNEON_DIAGNOSTICS="$task_diagnostics" \
    -DCMAKE_PREFIX_PATH="$runtime/prefix" -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
cmake --build "$task_build" --parallel "${NEON_BUILD_JOBS:-2}"
echo "Python: $runtime/.venv/bin/python"
echo "Runner: $task_build/neon_standalone"
