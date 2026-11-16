#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_dir="${AERODPO_GGUF_DIR:-/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0}"
run_id="${RUN_ID:-$(date +%m%d-%H%M)}"
simulator_port="${SIMULATOR_PORT:-25000}"
ddp_port="${DDP_MASTER_PORT:-20017}"
max_waypoints="${MAX_WAYPOINTS:-200}"
text_model="${LLAMA_TEXT_MODEL:-${artifact_dir}/language-base-q8_0.gguf}"
lora_model="${AERODPO_LORA_MODEL:-${artifact_dir}/aerodpo-language-lora-f16.gguf}"
mmproj_model="${AERODPO_MMPROJ_MODEL:-${artifact_dir}/aerodpo-mmproj-bf16.gguf}"
bridge_path="${AERODPO_LOCAL_BRIDGE:-${root_dir}/native/aerodpo_llamacpp_local/build/libaerodpo_llamacpp_local.so}"
context_size="${LLAMA_LOCAL_CONTEXT:-2048}"
eval_json_path="${EVAL_JSON_PATH:-/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json}"
eval_save_path="${EVAL_SAVE_PATH:-${root_dir}/eval_aerodpo_llamacpp_local_q8_seen1418_${run_id}}"

for required in "${text_model}" "${lora_model}" "${mmproj_model}" "${bridge_path}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing local llama.cpp artifact: ${required}" >&2
        exit 2
    fi
done
if [[ -e "${eval_save_path}" ]]; then
    echo "Refusing to overwrite existing result directory: ${eval_save_path}" >&2
    exit 2
fi

source /home/zt/miniconda3/etc/profile.d/conda.sh
conda activate aerodpo

cd "${root_dir}"
export PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/aerodpo_stopgo_eval.py \
    --run_type eval \
    --name AeroDPOLlamaCppLocalStopGo \
    --gpu_id 0 \
    --simulator_tool_port "${simulator_port}" \
    --DDP_MASTER_PORT "${ddp_port}" \
    --batchSize 1 \
    --always_help False \
    --use_gt False \
    --maxWaypoints "${max_waypoints}" \
    --enable_comm_delay True \
    --fast_eval True \
    --fast_eval_speedup 10 \
    --max_episodes_per_scene 80 \
    --comm_trace_csv_path "${root_dir}/bandwidth/ucc4g_bandwidth_trace.csv" \
    --dataset_path /HDD2/TravelUAV_dataset/TravelUAV_data/ \
    --eval_save_path "${eval_save_path}" \
    --model_path "${lora_model}" \
    --aerodpo_backend llamacpp_local \
    --llama_text_model_path "${text_model}" \
    --llama_mmproj_path "${mmproj_model}" \
    --llama_local_bridge_path "${bridge_path}" \
    --llama_local_context "${context_size}" \
    --eval_json_path "${eval_json_path}" \
    --map_spawn_area_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/map_spawnarea_info.json \
    --object_name_json_path /HDD2/TravelUAV_dataset/TravelUAV_data/data/meta/object_description.json \
    "$@"

