#!/bin/bash
# Start the DarSIA FluidFlower GUI on macOS from this clone.
#
# Double-click this file in Finder, or run it in Terminal:
#   ./start_darsia.command              the native DarSIA window (default)
#   ./start_darsia.command studio       the Phase Model Studio
#   ./start_darsia.command experiments  the FluidFlower experiments window
#
# The first start creates .venv-fluidflower next to this file and installs
# DarSIA into it (pip install -e .); later starts reinstall only after
# pyproject.toml changes. Set DARSIA_PYTHON to choose the Python (3.12 or
# newer). Output is also written to ~/Library/Logs/DarSIA.
# See docs/mac_setup_station.md.

set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
environment="$repository/.venv-fluidflower"
application="${1:-native}"
# Read on demand by the GUI but not declared in pyproject.toml (camera RAW files).
extra_packages=(rawpy)

fail() {
  echo
  echo "DarSIA did not start: $1"
  if [ -t 0 ]; then
    read -r -p "Press Return to close this window. " _ || true
  fi
  exit 1
}

case "$application" in
  native) module="darsia.fluidflower.native_darsia_app" ;;
  studio) module="darsia.fluidflower.phase_model_app" ;;
  experiments) module="darsia.fluidflower.app" ;;
  *) fail "unknown window '$application' (use native, studio or experiments)." ;;
esac

find_python() {
  local candidate
  for candidate in "${DARSIA_PYTHON:-}" python3.13 python3.12 python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -x "$environment/bin/python" ]; then
  python="$(find_python)" || fail "Python 3.12 or newer was not found. Install it (for example from python.org) and start again."
  echo "Creating $environment with $python"
  "$python" -m venv "$environment" || fail "the Python environment could not be created."
fi

stamp="$environment/.darsia-install"
signature="$(cksum < "$repository/pyproject.toml") ${extra_packages[*]}"
if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$signature" ]; then
  echo "Installing DarSIA and its packages into $environment (this takes a few minutes the first time)"
  "$environment/bin/python" -m pip install --upgrade pip || fail "pip could not be updated."
  "$environment/bin/python" -m pip install -e "$repository" "${extra_packages[@]}" \
    || fail "DarSIA could not be installed; see the messages above."
  printf '%s' "$signature" > "$stamp"
fi

# Use only this clone's environment, as the Windows launcher does.
unset PYTHONPATH PYTHONHOME
if [ "${QT_QPA_PLATFORM:-}" = "offscreen" ]; then
  unset QT_QPA_PLATFORM
fi
export VIRTUAL_ENV="$environment"
export DARSIA_FLUIDFLOWER_PYTHON="$environment/bin/python"
export PATH="$environment/bin:$PATH"
export PYTHONFAULTHANDLER=1

working="${DARSIA_FLUIDFLOWER_CACHE:-$HOME/Library/Caches/DarSIA}"
logs="$HOME/Library/Logs/DarSIA"
mkdir -p "$working" "$logs"
log="$logs/$(date +%Y%m%d-%H%M%S)-$application.log"
cd "$working"
echo "Starting DarSIA ($application); output is also written to $log"
set +e
"$environment/bin/python" -m "$module" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
  fail "the window closed with an error (exit status $status). The output is in $log"
fi
