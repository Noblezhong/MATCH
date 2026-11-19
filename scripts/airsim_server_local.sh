#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python3}" -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --root_path "${AIRSIM_ENV_ROOT:-/HDD2/AeroDuo_envs}" \
  --port "${AIRSIM_CONTROL_PORT:-25000}" --gpus "${AIRSIM_GPUS:-0}" \
  --clock_speed "${AIRSIM_CLOCK_SPEED:-10}" "$@"
