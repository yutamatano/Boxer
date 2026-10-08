#!/usr/bin/env bash
# Build the pinned offline adapter; dependencies stay outside Git in output/.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
deps_dir="$project_dir/output/neon_vio_deps"
orb_dir=${1:-"$deps_dir/ORB_SLAM3"}
pangolin_dir=${2:-"$deps_dir/Pangolin"}
prefix_dir=${3:-"$deps_dir/prefix/usr"}
build_dir=${NEON_VIO_BUILD_DIR:-"$project_dir/output/neon_vio_build"}
orb_commit=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
pangolin_commit=aff6883c83f3fd7e8268a9715e84266c42e2efe3
if [[ $(git -C "$orb_dir" rev-parse HEAD) != "$orb_commit" ]]; then
    echo "ORB-SLAM3 must be checked out at $orb_commit" >&2; exit 1
fi
if [[ $(git -C "$pangolin_dir" rev-parse HEAD) != "$pangolin_commit" ]]; then
    echo "Pangolin must be checked out at $pangolin_commit (v0.8)" >&2; exit 1
fi
for patch in export_neon.patch build_compat.patch; do
    if ! git -C "$orb_dir" apply --reverse --check "$project_dir/slam/$patch" 2>/dev/null; then
        git -C "$orb_dir" apply --check "$project_dir/slam/$patch"
        git -C "$orb_dir" apply "$project_dir/slam/$patch"
    fi
done
cmake_bin=$(command -v cmake || true)
if [[ -z "$cmake_bin" ]]; then cmake_bin="$prefix_dir/bin/cmake"; fi
export LD_LIBRARY_PATH="$prefix_dir/lib/x86_64-linux-gnu:$prefix_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$prefix_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
core_cxx=${CXX:-$(command -v clang++ || true)}
if [[ -z "$core_cxx" && -x "$prefix_dir/bin/clang++" ]]; then core_cxx="$prefix_dir/bin/clang++"; fi
if [[ -z "$core_cxx" ]]; then core_cxx=$(command -v g++); fi
core_cxx=$(command -v "$core_cxx")
core_flags=${CXXFLAGS:-}
if [[ "$core_cxx" == "$prefix_dir/bin/clang++" ]]; then
    core_flags="$core_flags --gcc-toolchain=/usr"
fi
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
    if [[ -n "$cached_cxx" && $(readlink -f "$cached_cxx") != $(readlink -f "$core_cxx") ]]; then
        echo "Compiler changed; choose a fresh NEON_VIO_BUILD_DIR (existing build preserved)." >&2; exit 1
    fi
fi
"$cmake_bin" -S "$pangolin_dir" -B "$deps_dir/build_pangolin" \
    -DCMAKE_INSTALL_PREFIX="$prefix_dir" -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_LIBPNG=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF \
    -DBUILD_PANGOLIN_LIBTIFF=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_LIBRAW=OFF \
    -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_FFMPEG=OFF
"$cmake_bin" --build "$deps_dir/build_pangolin" --parallel "${NEON_BUILD_JOBS:-1}"
"$cmake_bin" --install "$deps_dir/build_pangolin"
"$cmake_bin" -S "$project_dir/slam" -B "$build_dir" \
    -DORB_SLAM3_ROOT="$orb_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$core_cxx" -DCMAKE_CXX_FLAGS="$core_flags" \
    '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG' -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
"$cmake_bin" --build "$build_dir" --parallel "${NEON_BUILD_JOBS:-1}"
if [[ ! -f "$orb_dir/Vocabulary/ORBvoc.txt" ]]; then
    tar -xzf "$orb_dir/Vocabulary/ORBvoc.txt.tar.gz" -C "$orb_dir/Vocabulary"
fi
echo "Built: $build_dir/mono_inertial_neon"
echo "Local runtime libraries: $prefix_dir/lib/x86_64-linux-gnu:$prefix_dir/lib"
