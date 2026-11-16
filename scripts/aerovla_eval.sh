#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${RUN_ID:-$(date +%m%d-%H%M%S)}"
simulator_port="${SIMULATOR_PORT:-30000}"
ddp_port="${DDP_MASTER_PORT:-20016}"
max_waypoints="${MAX_WAYPOINTS:-200}"
data_root="${DATA_ROOT:-${root_dir}/dataset/TravelUAV_data}"
aerovla_root="${AEROVLA_ROOT:-/home/zt/code/AeroVLA}"
eval_json_path="${EVAL_JSON_PATH:-${data_root}/data/uav_dataset/seen_valset.json}"
eval_save_path="${EVAL_SAVE_PATH:-${root_dir}/eval_aerovla_edge_${run_id}_fast_x10}"
source "${CONDA_SH:-/home/smc/miniconda3/etc/profile.d/conda.sh}"
conda activate aerovla
cd "${root_dir}"
export PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/aerovla_stopgo_eval.py \
    --run_type eval --name AeroVLAEdge --gpu_id 0 \
    --simulator_tool_port "${simulator_port}" --DDP_MASTER_PORT "${ddp_port}" \
    --batchSize 1 --always_help False --use_gt False --maxWaypoints "${max_waypoints}" \
    --enable_comm_delay True --fast_eval True --fast_eval_speedup 10 \
    --max_episodes_per_scene 80 \
    --comm_trace_csv_path "${COMM_TRACE_CSV_PATH:-${root_dir}/bandwidth/ucc4g_bandwidth_trace.csv}" \
    --dataset_path "${data_root}/" --eval_save_path "${eval_save_path}" \
    --model_path "${AEROVLA_LORA_PATH:-${aerovla_root}/checkpoints/aero_vla}" \
    --base_model_path "${OPENVLA_BASE_PATH:-${aerovla_root}/openvla-7b}" \
    --eval_json_path "${eval_json_path}" \
    --map_spawn_area_json_path "${data_root}/data/meta/map_spawnarea_info.json" \
    --object_name_json_path "${data_root}/data/meta/object_description.json" \
    "$@"
