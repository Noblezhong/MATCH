# AeroDPO / TravelUAV benchmark static audit

Date: 2026-08-26  
Scope: the `eval/aerodpo-stopgo` worktree on 4090, compared read-only with
TravelUAV at 5090-LSQ (`f90c07a6d3c71b186fc1b23f59fec07bbeb1da64`).

## Execution status

The stale 4090 LLaMA-UAV edge server was stopped because it is not part of an
AeroDPO evaluation.  A real three-map smoke test then passed on Carla,
NewYorkCity, and TropicalIsland with real weights, UE, port 25000, Fast Eval
x10, native actions, LAND termination, upload-delay logging, and
`compute_metrics.py`.  It produced three unique end records and nine action
records; every model-generation and original-rate action-simulation latency
was positive.  The only metric warning was the expected smoke size (3 rather
than 1,418 episodes).

## Result categories

| Status | Meaning |
| --- | --- |
| Aligned | Same benchmark boundary is implemented. |
| Inherent | Required by the model's own interface. |
| Allowed API difference | Explicitly approved native AeroDPO control difference. |
| Limitation | Affects strict apples-to-apples interpretation and must accompany a result table. |
| Must fix | Unintended implementation difference. |

## Implementation comparison

| Item | AeroDPO Stop-and-go | LLaMA-UAV Stop-and-go | LLaMA-UAV Continuous | LLaMA-UAV Random | Review |
| --- | --- | --- | --- | --- | --- |
| Model | AeroVLA 2B merged base plus AeroDPO LoRA | LLaMA-UAV plus trajectory model | Same LLaMA-UAV / trajectory family | Same LLaMA-UAV / trajectory family when request is sampled | Inherent |
| Visual input | Front and down RGB, each resized to 384, vertically mosaicked | Five RGB views with the TravelUAV model input path | Same request/model input path, asynchronously captured | Request-dependent TravelUAV input | Inherent; limitation |
| State / prompt | Current pose and target position are reduced to semantic relative direction; object wording comes from the dataset instruction | TravelUAV history, relative target state and Assist notices | Same when an edge request is made | Same when scheduler requests | AeroDPO target direction is GT-derived; limitation to state explicitly |
| Model output | Quantized `fwd`, `down`, `yaw`, or `LAND` | Coarse waypoint proposal, then trajectory model | Latest-only edge plan / waypoint buffering | Random action decides request/continue/stop; planner used on requests | Inherent |
| Control API | Native AirSim rotate + velocity / vertical movement + zero-velocity hover | Waypoint path execution | One buffered waypoint/control step | Same buffered waypoint machinery | Allowed API difference |
| One decision executes | One AeroDPO command; `control_steps` counts the issued command | A stop-and-go plan may execute waypoint chunks | Asynchronous execution of buffered waypoints | Scheduler-dependent | Report AeroDPO actions, model decisions, and TravelUAV steps separately |
| Stop decision | `LAND` or zero action; <=20 m success, >20 m immediate `early_end` | GroundingDINO; >20 m sets early-end flag in reference behavior | GroundingDINO in async path | GroundingDINO when request produces a result | Inherent plus approved immediate-AeroDPO rule |
| Collision / regression / timeout / oracle | Physical collision, 10-frame non-decreasing NE, max 200 decisions, passive oracle | Same shared evaluation contract | Same shared contract, control limit 1000 | Same shared contract, control limit 1000 | Aligned except explicitly different step budget/unit |
| Fast evaluation | `FastEvalClock`, x10, original-rate action simulation time | Same clock / x10 machinery | Same | Same | Aligned |
| RGB-D upload / bandwidth | Reuses TravelUAV output payload estimator, UCC 4G trace and logical uplink delay | Same | Same | Same when request occurs | Aligned implementation; its relevance to AeroDPO is a protocol limitation |
| Time Shift / State Shift | N/A: no asynchronous guidance application | N/A in stop-and-go | First actual asynchronous guidance application | First actual asynchronous guidance application | Aligned metric interpretation |
| Waiting Time | logical E2E minus original-rate action simulation time | Same stop-and-go residual definition in current metric script | Same residual metric | Same residual metric | Aligned script; use the selected paper definition consistently |
| Inference latency | CUDA-synchronised `model.generate` only; preprocessing and full decision also logged | Existing LLaMA/trajectory timings | Asynchronous edge timing is not directly equivalent | Request-dependent | Do not compare a synchronous AeroDPO `generate` mean to asynchronous E2E as if identical |

## Checked AeroDPO data and control flow

1. `seen_valset.json` contains 75,374 annotations and 1,418 distinct
   `json` episode paths.  The evaluator rejects a different unique count unless
   the smoke-only `--allow_nonstandard_eval_count True` flag is supplied.
2. `AeroDPOEvalModelWrapper` retains the official two-view mosaic, Qwen
   processor, bfloat16 / FlashAttention2 loading, merged LoRA, deterministic
   generation, original action dequantisation and LAND parsing.
3. `move_path_by_aerodpo_actions` retains AeroDPO's yaw threshold: for a
   large yaw action the native policy changes yaw and vertical position rather
   than also applying the forward component.  This is intentionally not
   converted to a TravelUAV waypoint.
4. The evaluator uses batch size 1, 200 decisions, retry-at-most-three, scene
   reuse at most 80, port 25000, Fast Eval x10, and buffers retries so rejected
   attempts never enter final JSONL.
5. Action records contain the generated text, parsed action, LAND flag,
   preparation time, synchronised generation time, full-decision time,
   original-rate action simulation time, observation/upload timing and the
   fixed `termination_backend="aerodpo_land"` marker.  End records include
   final NE, collision, oracle success, termination reason, action count and
   decision count.
6. The native action worker has a 15-second wall-time guard only for an
   AirSim wedge.  Normal commands retain AeroDPO's original API calls and
   values; a timed-out worker returns a failed attempt so the evaluator's
   three-attempt retry path can run rather than blocking forever.
7. This server's Record cameras cannot return DepthPerspective.  AeroDPO mode
   therefore reuses the standard front/down policy RGB-D as trajectory-record
   frames and never requests that broken Record-depth stream.  It does not run
   GroundingDINO.  The actual RGB-D payload remains logged; its two-view input
   difference is an AeroDPO model-interface limitation, not a LLaMA control
   dependency.

## Random read-only snapshot

The Random files are untracked on 5090-LSQ and were not copied to 4090.

| File | SHA256 |
| --- | --- |
| `src/vlnce_src/random_eval.py` | `208b1f744e7d7603a85bf64f4690a776016cd84cdc7495fb62307d27159c9841` |
| `scripts/random_eval.sh` | `1b823b1641879b55f0860f97c16afb9985a1abac30210ef55209f775172e6c18` |

The snapshot script uses `SEED=${SEED:-0}`, GPU 0, batch 1, 1,000 control
steps, scene reuse 80, the seen split, communication delay, and Fast Eval x10.

## Required before publication-scale run

1. Run the 1,418 unique seen episodes and report SR, OSR, CR, NE,
   action/decision counts, logical E2E, Waiting Time, and the stated latency
   summary (mean, median, p95, maximum, first call, and episode-balanced mean).
