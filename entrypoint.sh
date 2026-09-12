#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$PWD}"

if [[ ! -d .venv ]]; then
  uv venv --system-site-packages --no-managed-python .venv
fi

uv sync --no-managed-python --all-extras
source .venv/bin/activate

if [[ $# -gt 0 ]]; then
  exec "$@"
fi
exec bash
