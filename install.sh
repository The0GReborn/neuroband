#!/bin/bash
# install.sh — full NeuroBand environment setup.
# Run once, from the repo root:
#   bash install.sh
#
# What it does (in order, all idempotent — safe to re-run):
#   1. Installs system packages (espeak-ng, python3-venv, python3-rpi.gpio)
#   2. Creates .venv if it doesn't exist yet
#   3. Installs required Python packages into .venv
#   4. Sets up the `neuroband` command on PATH (via docs/install.sh)
#   5. Offers to launch the first-time setup wizard
#
# Safe to run on a machine with no hardware attached — steps 1-4 don't
# touch any GPIO/hardware, and step 5 is asked, never forced, and can be
# run in --simulate mode.
#
# Flags:
#   --yes          Don't prompt before installing system packages or
#                   launching the wizard — assume yes to both.
#   --no-wizard    Skip the "launch wizard now?" step entirely (just sets
#                   up the environment).
#   --simulate     If the wizard is launched, launch it with --simulate.

set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

ASSUME_YES=false
SKIP_WIZARD=false
SIMULATE=false
for arg in "$@"; do
  case "$arg" in
    --yes) ASSUME_YES=true ;;
    --no-wizard) SKIP_WIZARD=true ;;
    --simulate) SIMULATE=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

confirm() {
  # confirm "question" -> 0 (yes) or 1 (no)
  if [ "$ASSUME_YES" = true ]; then
    return 0
  fi
  read -r -p "$1 [Y/n]: " reply
  [ -z "$reply" ] || [[ "$reply" =~ ^[Yy] ]]
}

# Use sudo only when not already root (avoids failing in containers/sandboxes
# where sudo isn't installed but the script is already running as root).
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "=== NeuroBand setup ==="
echo

# ── 1. System packages ───────────────────────────────────────────────────────
if command -v apt-get >/dev/null 2>&1; then
  if confirm "Install system packages (espeak-ng, python3-venv, python3-rpi.gpio) via apt?"; then
    echo "--- Installing system packages ---"
    # Don't let a failing THIRD-PARTY repo (unrelated to the packages we
    # need) hard-abort the whole installer — `apt-get update` exits non-zero
    # if ANY configured repo fails to fetch, even when the repos we actually
    # need succeeded. Try the real install regardless; if it also fails,
    # that failure is real and should stop the script.
    $SUDO apt-get update || echo "WARNING: apt-get update reported errors (possibly an unrelated repo) — continuing anyway."
    $SUDO apt-get install -y espeak-ng python3-venv python3-pip python3-rpi.gpio
  else
    echo "Skipping system package install — make sure espeak-ng and python3-venv"
    echo "are available yourself, or the wizard's audio step and venv creation"
    echo "below may fail."
  fi
else
  echo "No apt-get found (not Debian/Raspberry Pi OS) — skipping system package"
  echo "install. Make sure espeak-ng is installed some other way before running"
  echo "the wizard."
fi
echo

# ── 2. Virtual environment ───────────────────────────────────────────────────
# --system-site-packages: RPi.GPIO is installed via apt (above) into the
# SYSTEM Python's site-packages, not pip. A normal venv is isolated from
# system packages, so without this flag RPi.GPIO would silently be
# invisible inside .venv and the code would fall back to simulation mode
# even with real hardware connected and apt reporting a successful install.
if [ -f ".venv/bin/activate" ]; then
  echo "--- .venv already exists, reusing it ---"
else
  echo "--- Creating .venv (with access to system packages, for RPi.GPIO) ---"
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo

# ── 3. Python packages ───────────────────────────────────────────────────────
echo "--- Installing Python packages into .venv ---"
pip install --quiet --upgrade pip
pip install --quiet \
  flask flask-socketio \
  numpy scipy scikit-learn joblib \
  pygame rich
echo "Python packages installed."
echo

# ── 4. neuroband CLI command ─────────────────────────────────────────────────
bash docs/install.sh
echo

# ── 5. Optional: launch the wizard ───────────────────────────────────────────
if [ "$SKIP_WIZARD" = true ]; then
  echo "=== Setup complete. Run 'neuroband --setup' when you're ready. ==="
  exit 0
fi

echo "=== Environment setup complete. ==="
if confirm "Launch the first-time NeuroBand setup wizard now?"; then
  ARGS=(--setup)
  [ "$SIMULATE" = true ] && ARGS+=(--simulate)
  # New shell so the freshly-appended PATH change is picked up even if this
  # script was run in the current shell rather than sourced.
  exec python3 main.py "${ARGS[@]}"
else
  echo "OK — run 'neuroband --setup' (or 'python main.py --setup') whenever you're ready."
fi
