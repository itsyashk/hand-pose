#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv311"
VENV_PY_WIN="$VENV_DIR/Scripts/python.exe"
VENV_PY_POSIX="$VENV_DIR/bin/python"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

create_venv() {
  if has_cmd py && py -3.11 -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
    echo "Creating Python 3.11 venv via 'py -3.11'..."
    py -3.11 -m venv "$VENV_DIR"
  elif has_cmd python3.11; then
    echo "Creating Python 3.11 venv via 'python3.11'..."
    python3.11 -m venv "$VENV_DIR"
  else
    echo "ERROR: Python 3.11 not found. Install it or ensure 'py -3.11' works." >&2
    exit 1
  fi
}

if [[ ! -d "$VENV_DIR" ]]; then
  create_venv
fi

if [[ -x "$VENV_PY_WIN" ]]; then
  PY="$VENV_PY_WIN"
else
  PY="$VENV_PY_POSIX"
fi

"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt --no-input

exec "$PY" main.py
#exec "$PY" main_test.py