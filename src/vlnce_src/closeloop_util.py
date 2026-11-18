from src.vlnce_src.collision_state import EpisodeCollisionState

import json
import os
import random
import shutil

import cv2
import numpy as np
from utils.utils import *
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.vlnce_src.env_uav import AirVLNENV, RGB_FOLDER, DEPTH_FOLDER
from src.vlnce_src.eval_contract import (
    check_collision_without_tiny_diff,
    mark_ne_regression_failure,
)


def setup(dagger_it=0, manual_init_distributed_mode=False):
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank() + dagger_it
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True


def initialize_env_eval(dataset_path, save_path, eval_json_path):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path)
    return train_env

        
def finalize_eval_episode(episode, ori_data_dir, success, oracle_success, done, saved, force=False):
    """Shared, idempotent trajectory finalization for evaluation entrypoints."""
    if saved:
        return True
    if not (force or done):
        return False
    prefix = "success_" if success else ("oracle_" if oracle_success else "")
    trajectory_name = prefix + ori_data_dir.split("/")[-1]
    save_to_dataset_eval(episode, os.path.join(args.eval_save_path, trajectory_name), ori_data_dir)
    return True


def save_to_dataset_eval(episodes, path, ori_traj_dir):
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump({'ori_traj_dir': ori_traj_dir}, f)

def save_logs(episodes, trajectory_dir):
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        info = {'frame': idx, 'sensors': episode['sensors']}
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(info, f)

def save_images(episodes, trajectory_dir):
    for idx, episode in enumerate(episodes):
        if 'rgb' in episode:
            for cid, camera_name in enumerate(RGB_FOLDER):
                if cid >= len(episode['rgb']):
                    break
                image = episode['rgb'][cid]
                if image is None or not hasattr(image, 'size') or image.size == 0:
                    continue
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
        if 'depth' in episode:
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                if cid >= len(episode['depth']):
                    break
                image = episode['depth'][cid]
                if image is None or not hasattr(image, 'size') or image.size == 0:
                    continue
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)


def target_distance_increasing_for_10frames(lst):
    if len(lst) < 10:
        return False
    sublist = lst[-10:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True


def target_distance_regression(lst, frames: int = 10):
    if len(lst) < frames:
        return False
    sublist = lst[-frames:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True


class BatchIterator:
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

                    
                    
class EvalBatchState:
    def __init__(
        self,
        batch_size,
        env_batchs,
        env,
        assist,
        ignore_tiny_diff=False,
    ):
        self.batch_size = batch_size
        self.eval_env = env
        self.assist = assist
        self.ignore_tiny_diff = bool(ignore_tiny_diff)
        self.episodes = [[] for _ in range(batch_size)]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.collisions = [False] * batch_size
        self.cr_states = [EpisodeCollisionState() for _ in range(batch_size)]
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = [False] * batch_size
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.envs_to_pause = []
        
        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))

    def _calculate_distance(self, observation, target_position):
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def update_from_env_output(self, outputs):
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        if self.ignore_tiny_diff:
            self.collisions, self.dones = check_collision_without_tiny_diff(
                self.episodes,
                observations,
                self.collisions,
                self.dones,
            )
        else:
            self.collisions, self.dones = self.assist.check_collision_by_depth(
                self.episodes,
                observations,
                self.collisions,
                self.dones,
            )
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.collisions[i] = self.cr_states[i].observe(observations[i])
            if self.collisions[i]:
                self.dones[i] = True
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                self.collisions, self.dones = mark_ne_regression_failure(
                    self.collisions,
                    self.dones,
                    i,
                )

    def get_assist_notices(self):
        return self.assist.get_assist_notice(self.episodes, self.trajs, self.object_infos, self.target_positions)

    def update_metric(self):
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                if self.distance_to_ends[i][-1] <= 20:
                    self.success[i] = True
                    self.dones[i] = True
                elif self.distance_to_ends[i][-1] > 20:
                    self.early_end[i] = True
                    
    # ── TERMINATION (stop-and-go paradigm) ──────────────────────────
    # Three sources: collision / DINO (vision) / NE-trend (geometry).
    # DINO predicts done + NE > 20 m → early_end flag ONLY (not done).
    # oracle_success is a PASSIVE post-hoc flag — NEVER terminates.
    # This is the *reference implementation*; rule_based_eval, continue_eval,
    # and drl_scheduler_env MUST match this contract.
    # DO NOT modify without updating ALL paradigms.
    # ──────────────────────────────────────────────────────────────────
    def maybe_finalize(self, force=False):
        for i in range(self.batch_size):
            self.skips[i] = finalize_eval_episode(
                self.episodes[i], self.ori_data_dirs[i], self.success[i],
                self.oracle_success[i], self.dones[i], self.skips[i], force,
            )
        return all(self.skips)

    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            if t == args.maxWaypoints:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                self.envs_to_pause.append(i)
                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex +  self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                save_to_dataset_eval(self.episodes[i], new_traj_dir, self.ori_data_dirs[i])
                self.skips[i] = True
                print(i, " has finished!")
        return np.array(self.skips).all()
