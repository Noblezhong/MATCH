"""Random scheduling policy: every motion or one-second stop ends a decision."""
import sys, random
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.vlnce_src import continue_eval as base
globals().update({k: v for k, v in vars(base).items() if not k.startswith('__') and k != 'eval'})
from src.vlnce_src.continuous_runtime import evaluate

class RandomScheduler:
    ACTIONS = ('CONTINUE_ONLY', 'CONTINUE_AND_REQUEST', 'STOP_ONLY', 'STOP_AND_REQUEST')

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, buffer_remaining, active_state_drift_m=0, planner_has_inflight=False):
        action = self.rng.choice(self.ACTIONS)
        move = action.startswith('CONTINUE') and buffer_remaining > 0
        return SimpleNamespace(action=action, request='REQUEST' in action, move=move, idle_ms=0 if move else 1000)

def eval(model_wrapper, assist, eval_env, eval_save_dir, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay, chunk_waypoints):
    return evaluate(model_wrapper, assist, eval_env, eval_save_dir, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay, chunk_waypoints, mode='random')
if __name__ == '__main__':
    _set_console_log_message_only()
    configure_fast_eval_output(args, 'random')
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, 'profile_logs')
    os.makedirs(profile_log_dir, exist_ok=True)
    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    if not trace_path.exists():
        raise FileNotFoundError(f'bandwidth trace not found: {trace_path}')
    chunk_waypoints = max(1, int(args.chunk_waypoints))
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(f'Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), enable_comm_delay={enable_comm_delay}, scheduler=random')
    setup()
    assert CheckPort(), 'error port'
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False
    model_wrapper = ProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    print('Assist setting: always_help --', args.always_help, '    use_gt --', args.use_gt)
    profile_log_path = os.path.join(profile_log_dir, f'random_{args.make_dir_time}.jsonl')
    summary_path = os.path.join(profile_log_dir, f'random_{args.make_dir_time}_summary.json')
    eval(model_wrapper=model_wrapper, assist=assist, eval_env=eval_env, eval_save_dir=eval_save_path, profile_log_path=profile_log_path, summary_path=summary_path, bandwidth_trace=bandwidth_trace, enable_comm_delay=enable_comm_delay, chunk_waypoints=chunk_waypoints)
    eval_env.delete_VectorEnvUtil()
