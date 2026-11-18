#!/usr/bin/env python3
"""Run the fixed Seen284 Stop-go smoke and formal evaluations on 5090/4090."""
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/home/noble/mnt/R1_new_experiment')
REMOTE_OUT = Path('/HDD1/code/R1_new_experiment')
DATA = Path('/HDD2/TravelUAV_dataset/TravelUAV_data')
INDEX = REPO/'eval_subsets/seen_valset_stratified284_seed20261117.json'
SELECTION = INDEX.with_suffix('.selection.json')
TRACE = REPO/'bandwidth/ucc4g_bandwidth_trace.csv'
PYTHON = '/home/noble/miniforge3/envs/llamauav/bin/python'
REMOTE = 'zt@192.168.105.18'
SERVER_SESSION = 'match_stopgo_seen284_server'

def call(argv, **kw):
    return subprocess.run([str(a) for a in argv], check=True, **kw)

def capture(argv):
    return subprocess.check_output([str(a) for a in argv], text=True).strip()

def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check_mounts():
    for target, source in [(OUT,REMOTE+':'+str(REMOTE_OUT)),(Path('/HDD2/TravelUAV_dataset'),REMOTE+':/HDD2/TravelUAV_dataset'),(Path('/mnt/remote_models'),REMOTE+':/HDD1/code/MATCH/Model')]:
        actual=capture(['findmnt','-rn','-M',target,'-o','SOURCE'])
        if actual != source: raise RuntimeError(f'Mount mismatch: {target}: {actual!r}')

def preflight():
    check_mounts()
    rows=json.loads(INDEX.read_text())
    selected=json.loads(SELECTION.read_text())
    assert len({r['json'] for r in rows}) == 284
    assert sha(INDEX)==selected['index_sha256']
    areas=json.loads((DATA/'data/meta/map_spawnarea_info.json').read_text())
    json.loads((DATA/'data/meta/object_description.json').read_text())
    for rel in sorted({r['json'] for r in rows}):
        p=DATA/rel
        assert p.is_file(), str(p)
        json.loads(p.read_text())
        mark=json.loads((p.parent/'mark.json').read_text())
        assert 'object_name' in mark and 'target' in mark
        assert (p.parent/'object_description.json').is_file(),str(p)
        assert Path(rel).parts[0] in areas
    for rel in ['Model/LLaMA-UAV/work_dirs/llama-uav-7b/config.json','Model/LLaMA-UAV/work_dirs/traveluav-traj-model','Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5/config.json','Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth','Model/LLaMA-UAV/llamavid/processor/clip-patch14-224/preprocessor_config.json','src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth']:
        assert (REPO/rel).exists(),rel
    values=[float(r['bandwidth_mbps']) for r in csv.DictReader(TRACE.open())]
    assert len(values)==2004 and all(math.isfinite(v) and v>0 for v in values)
    changed=capture(['git','-C',REPO,'diff','--diff-filter=d','--name-only']).splitlines()
    for rel in changed+['src/vlnce_src/collision_state.py','src/vlnce_src/eval_stopgo_seen284.py','scripts/eval_stopgo_seen284.sh']:
        p=REPO/rel
        if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))
        elif p.suffix=='.sh': call(['bash','-n',p])
    call(['git','-C',REPO,'diff','--check'])
    print('STATIC_PREFLIGHT_OK: 284 episodes, dataset/model files, mounts, UCC4G, Python/bash syntax',flush=True)
    return rows,selected

def prepare_run(label, rows):
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%m%d-%H%M')
    run=OUT/f'eval_stop_go_{label}_{stamp}_fastx10'
    run.mkdir(exist_ok=False)
    (run/'inputs').mkdir(); (run/'logs').mkdir()
    write_json(run/'inputs/eval_index.json',rows)
    shutil.copy2(SELECTION,run/'inputs/seen284.selection.json')
    shutil.copy2(TRACE,run/'inputs/ucc4g_bandwidth_trace.csv')
    (run/'inputs/worktree.patch').write_text(capture(['git','-C',REPO,'diff','--binary'])+'\n')
    files=capture(['git','-C',REPO,'diff','--name-only']).splitlines()+['src/vlnce_src/collision_state.py','src/vlnce_src/eval_stopgo_seen284.py','scripts/eval_stopgo_seen284.sh']
    for rel in sorted(set(files)):
        p=REPO/rel
        if p.is_file():
            dest=run/'inputs/code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    cfg=dict(label=label,host_inference='192.168.105.17',host_simulator='192.168.105.18',simulator_port=25000,physical_output=str(REMOTE_OUT/run.name),mounted_output=str(run),source_index=str(INDEX),source_index_sha256=sha(INDEX),run_index_sha256=sha(run/'inputs/eval_index.json'),trace_sha256=sha(TRACE),bandwidth='UCC4G; UUID SHA256 first 8 hex digits offset; floor(episode logical ms / 1000); cycle=2004 samples',fast_eval_speedup=10,git_commit=capture(['git','-C',REPO,'rev-parse','HEAD']),git_status=capture(['git','-C',REPO,'status','--short']),collision_policy='episode_collision_v1',expected_episodes=len({r['json'] for r in rows}))
    write_json(run/'run_config.json',cfg)
    # Check a new file written through the mount from the actual storage host.
    remote_path=REMOTE_OUT/run.name/'run_config.json'
    remote_sha=capture(['ssh',REMOTE,'sha256sum',remote_path]).split()[0]
    assert remote_sha==sha(run/'run_config.json')
    return run

def start_server(run):
    assert subprocess.run(['ssh',REMOTE,'tmux','has-session','-t',SERVER_SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'Our server session already exists'
    import shlex
    log=REMOTE_OUT/run.name/'logs/server4090.log'
    command=f'cd /HDD1/code/MATCH && bash scripts/airsim_server_4090.sh > {shlex.quote(str(log))} 2>&1'
    call(['ssh',REMOTE,'tmux','new-session','-d','-s',SERVER_SESSION,shlex.quote(command)])
    # Wait only for the scene-control socket; scenes are opened by the evaluator.
    import socket,time
    for _ in range(30):
        try:
            with socket.create_connection(('192.168.105.18',25000),timeout=2): return
        except OSError: time.sleep(1)
    raise RuntimeError('4090 scene-control server did not start')

def run_eval(run):
    check_mounts()
    args=['bash',REPO/'scripts/eval.sh','--simulator_tool_host','192.168.105.18','--simulator_tool_port','25000','--DDP_MASTER_PORT','20001','--eval_json_path',run/'inputs/eval_index.json','--comm_trace_csv_path',run/'inputs/ucc4g_bandwidth_trace.csv','--eval_save_path',run,'--max_episodes_per_scene','80']
    cfg=json.loads((run/'run_config.json').read_text());cfg['command']=[str(a) for a in args];write_json(run/'run_config.json',cfg)
    print('RUNNING '+str(run),flush=True)
    env=os.environ.copy();env['PATH']=str(Path(PYTHON).parent)+':'+env['PATH'];env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false';env['MATCH_RUNTIME_CONFIG_DIR']=str(run/'inputs/runtime_vision_config')
    # Prevent inherited distributed or teleport settings from changing this experiment.
    for name in ['RANK','WORLD_SIZE','LOCAL_RANK','CMA_TELEPORT']: env.pop(name,None)
    with (run/'logs/eval5090.log').open('w') as log:
        call(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)

def verify_and_metrics(run):
    expected={Path(r['json']).parts[1] for r in json.loads((run/'inputs/eval_index.json').read_text())}
    terminals=[];last_time={}
    bandwidth=[float(r['bandwidth_mbps']) for r in csv.DictReader((run/'inputs/ucc4g_bandwidth_trace.csv').open())]
    for p in sorted((run/'profile_logs').glob('*.jsonl')):
        for line in p.open():
            r=json.loads(line)
            ids=r.get('seq_names',[])
            logical=r.get('logical_elapsed_ms')
            if logical is not None:
                assert math.isfinite(logical) and logical>=0
                # On a successful run without retries logical time is monotonic per episode.
                for uid in ids:
                    assert logical>=last_time.get(uid,0),(uid,logical,last_time.get(uid))
                    last_time[uid]=logical
            if r.get('record_type')=='episode_end': terminals.append(r)
    actual=[r['seq_names'][0] for r in terminals]
    assert len(actual)==len(expected) and len(set(actual))==len(actual) and set(actual)==expected,(len(actual),len(expected))
    scenes={}
    for r in terminals:
        uid=r['seq_names'][0]
        assert r['collision_policy']=='episode_collision_v1' and type(r['cr_collision']) is bool
        assert r['initial_obs_latency_ms']>=0
        assert abs(r['initial_uplink_start_logical_ms']-r['initial_obs_latency_ms'])<1e-6
        assert math.isfinite(r['episode_latency_ms']) and r['episode_latency_ms']>=r['initial_obs_latency_ms']+r['initial_uplink_latency_ms']
        offset=int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)%len(bandwidth)
        expected_bw=bandwidth[(offset+int(r['initial_uplink_start_logical_ms']//1000))%len(bandwidth)]
        assert abs(r['initial_uplink_bandwidth_mbps']-expected_bw)<1e-7
        assert abs(r['initial_uplink_latency_ms']-r['initial_uplink_payload_bits']/(expected_bw*1e6)*1000)<1e-5
        dirs=[run/(prefix+uid) for prefix in ('success_','oracle_','') if (run/(prefix+uid)).is_dir()]
        assert len(dirs)==1 and any((dirs[0]/'log').glob('*.json')) and (dirs[0]/'ori_info.json').is_file()
        assert any((dirs[0]/'frontcamera').glob('*.png'))
        scenes[r['map_names'][0]]=scenes.get(r['map_names'][0],0)+1
    call([PYTHON,REPO/'scripts/compute_metrics.py','--input',run,'--eval-dir',run,'--method','stopgo','--expected-episodes',len(expected),'--output',run/'combined_metrics.json'])
    metric=json.loads((run/'combined_metrics.json').read_text())
    assert metric['episodes']==len(expected)
    write_json(run/'validation.json',dict(passed=True,episodes=len(expected),unique_uuids=len(set(actual)),scene_counts=scenes,collision_policy='episode_collision_v1',initial_upload_validated=True,logical_time_monotonic=True,trajectory_files_present=True,index_exact_match=True))
    print('VALIDATED '+str(run),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    rows,selection=preflight()
    if args.preflight_only: return
    ids=[]
    for scene,count in [('Carla_Town01',2),('Carla_Town04',1)]:
        ids.extend(sorted(r['json'] for r in selection['episodes'] if r['scene']==scene)[:count])
    smoke_rows=[r for r in rows if r['json'] in ids]
    smoke=prepare_run('seen3_smoke',smoke_rows)
    formal=None
    try:
        start_server(smoke)
        run_eval(smoke)
        verify_and_metrics(smoke)
        formal=prepare_run('seen284',rows)
        write_json(formal/'smoke_reference.json',dict(directory=str(REMOTE_OUT/smoke.name),validated=True))
        run_eval(formal)
        verify_and_metrics(formal)
        print('FORMAL_COMPLETE '+str(REMOTE_OUT/formal.name),flush=True)
    except Exception as e:
        write_json((formal or smoke)/'failure.json',dict(error=repr(e)))
        raise
    finally:
        # Close only scenes owned by our dedicated server, then stop that server.
        subprocess.run([PYTHON,'-c',"import msgpackrpc; c=msgpackrpc.Client(msgpackrpc.Address('192.168.105.18',25000),timeout=30); c.call('close_scenes','192.168.105.18')"],check=False)
        subprocess.run(['ssh',REMOTE,'tmux','send-keys','-t',SERVER_SESSION,'C-c'],check=False)
        if formal is not None and (smoke/'logs/server4090.log').exists():
            shutil.copy2(smoke/'logs/server4090.log',formal/'logs/server4090.log')

if __name__=='__main__': main()
