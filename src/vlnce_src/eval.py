import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from env_uav import AirVLNENV
from src.common.param import args, model_args, data_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    EvalBatchState,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    setup,
)
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_outputs,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    action_timing,
    configure_fast_eval_output,
)
from src.vlnce_src.eval_contract import execute_stop_waypoint_chunks
from utils.logger import logger
from utils.utils import *


def _metric_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _print_profile_line(step_idx, profile_info):
    message = (
        f"[eval] step={step_idx} "
        f"llm={profile_info['llm_latency_ms']:.3f}ms "
        f"traj={profile_info['traj_latency_ms']:.3f}ms "
        f"action={profile_info['airsim_action_latency_ms']:.3f}ms "
        f"obs={profile_info['obs_latency_ms']:.3f}ms "
        f"dino={profile_info['groundingdino_latency_ms']:.3f}ms "
        f"uplink={profile_info['uplink_latency_ms']:.3f}ms "
        f"bw={profile_info['uplink_bandwidth_mbps']:.3f}Mbps "
        f"payload={profile_info['uplink_payload_mb']:.3f}MB "
        f"llm_output={profile_info['llm_output']}"
    )
    logger.info(message)


def eval(
    model_wrapper: ProfileTravelModelWrapper,
    assist: Assist,
    eval_env: AirVLNENV,
    eval_save_dir,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
):
    from src.vlnce_src.continuous_runtime import evaluate
    return evaluate(model_wrapper, assist, eval_env, eval_save_dir, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay, 5, mode="stopgo")


if __name__ == "__main__":
    configure_fast_eval_output(args, "stop_and_go")
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

    enable_comm_delay = bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    if not trace_path.exists():
        raise FileNotFoundError(f"bandwidth trace not found: {trace_path}")
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), enable_comm_delay={enable_comm_delay}")

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = ProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)

    profile_log_path = os.path.join(profile_log_dir, f"eval_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"eval_{args.make_dir_time}_summary.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
    )

    eval_env.delete_VectorEnvUtil()
