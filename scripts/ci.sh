#!/usr/bin/env bash
# Run the same checks locally and in GitHub Actions.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
PYTHON_BIN="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

git diff --check
git diff --cached --check

"$PYTHON_BIN" - <<'PY'
import py_compile
import subprocess

files = subprocess.check_output(
    ["git", "ls-files", "-z", "--", "*.py"]
).decode().split("\0")
for filename in filter(None, files):
    py_compile.compile(filename, doraise=True)
print("Python syntax checks passed")
PY

while IFS= read -r -d '' filename; do
    bash -n "$filename"
done < <(git ls-files -z -- '*.sh')

mkdir -p output/ci
"$PYTHON_BIN" -m pytest tests -q -ra \
    --junitxml=output/ci/junit.xml 2>&1 | tee output/ci/pytest.log
