#!/bin/bash
# docs/install.sh — one-time setup: puts the `neuroband` command on PATH.
#
# Run once, from the repo root:
#   bash docs/install.sh
#
# What it does:
#   1. Symlinks bin/neuroband into ~/.local/bin/neuroband
#   2. Adds ~/.local/bin to PATH in ~/.bashrc, if it isn't already there
#
# After this, open a new shell (or `source ~/.bashrc`) and `neuroband --help`
# should work from any directory.

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$HOME/.local/bin"
ln -sf "$REPO_ROOT/bin/neuroband" "$HOME/.local/bin/neuroband"
chmod +x "$REPO_ROOT/bin/neuroband"

echo "Linked: $HOME/.local/bin/neuroband -> $REPO_ROOT/bin/neuroband"

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  echo "Added \$HOME/.local/bin to PATH in ~/.bashrc — run: source ~/.bashrc"
else
  echo "\$HOME/.local/bin is already on PATH."
fi

echo
echo "Done. Open a new shell (or 'source ~/.bashrc'), then try:"
echo "  neuroband --help"
