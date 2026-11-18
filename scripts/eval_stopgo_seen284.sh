#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$root_dir/src/vlnce_src/eval_stopgo_seen284.py" "$@"
