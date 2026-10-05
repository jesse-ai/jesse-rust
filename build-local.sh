#!/bin/bash
# Build into the selected development environment without upgrading its dependencies.
set -euo pipefail
cd "$(dirname "$0")"
JESSE_BUILD_PYTHON="${JESSE_BUILD_PYTHON:-python3}"
command -v rustc >/dev/null || { echo "Rust is required for a source build."; exit 1; }
"$JESSE_BUILD_PYTHON" -c 'import maturin, numpy'
# Building a wheel works for Conda and venv alike and does not install build tools.
JESSE_BUILD_OUTPUT=$(mktemp -d)
trap 'rm -rf "$JESSE_BUILD_OUTPUT"' EXIT
PYO3_PYTHON="$JESSE_BUILD_PYTHON" "$JESSE_BUILD_PYTHON" -m maturin build --release --locked --interpreter "$JESSE_BUILD_PYTHON" --out "$JESSE_BUILD_OUTPUT"
"$JESSE_BUILD_PYTHON" -m pip install --no-deps --no-index --force-reinstall "$JESSE_BUILD_OUTPUT"/*.whl
"$JESSE_BUILD_PYTHON" scripts/check_coordinator.py
"$JESSE_BUILD_PYTHON" -c 'import jesse_rust, numpy; print("Jesse Rust indicators and coordinator installed")'
