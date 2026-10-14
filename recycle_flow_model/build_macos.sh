#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

target_arch="${1:-}"
case "$target_arch" in
  "") unset PYINSTALLER_TARGET_ARCH || true ;;
  arm64|x86_64|universal2) export PYINSTALLER_TARGET_ARCH="$target_arch" ;;
  *)
    echo "Usage: ./build_macos.sh [arm64|x86_64|universal2]" >&2
    exit 2
    ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This build script must run on macOS." >&2
  exit 1
fi

if ! python3 -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller is missing. Install build dependencies first:" >&2
  echo "  python3 -m pip install -r requirements-build.txt" >&2
  exit 1
fi

rm -rf build "dist/CHE031 Crystallization.app" dist/CHE031_Crystallization_App
python3 -m PyInstaller --clean --noconfirm combined_app_macos.spec

echo
echo "Build complete:"
echo "  $(pwd)/dist/CHE031 Crystallization.app"
