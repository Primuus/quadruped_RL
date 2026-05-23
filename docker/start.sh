#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaacgym') else 1)"; then
  if [ ! -d /opt/isaacgym/python ]; then
    echo "Missing Isaac Gym mount at /opt/isaacgym/python" >&2
    exit 1
  fi
  python -m pip install --no-cache-dir -e /opt/isaacgym/python --no-deps
fi

exec bash
