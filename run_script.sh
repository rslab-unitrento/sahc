#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_script.sh scripts/tuning/lightning_tuning.py [suffix]" >&2
  exit 2
fi

script_path=$1
script_name=$(basename "$script_path")
script_dir=$(dirname "$script_path")
config_name=${script_name%.py}
if [[ $# -ge 2 ]]; then
  config_name="${config_name}_$2"
fi

cd "${REPO_DIR:-$PWD}"
config_path="$script_dir/config_files/$config_name.yaml"
if [[ ! -f "$config_path" ]]; then
  echo "configuration file not found: $config_path" >&2
  exit 2
fi

uv run --no-managed-python python "$script_path" --conf "$config_path"
