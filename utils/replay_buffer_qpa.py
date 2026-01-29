# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from utils.replay_buffer_drqv2 import load_episode, episode_len, ReplayBuffer, _worker_init_fn

class ReplayBufferQPA(ReplayBuffer):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        frame_stack,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        her_ratio=0.5,
        max_on_size=10,
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._frame_stack = frame_stack
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.her_ratio=her_ratio
        self.max_on_size=max_on_size
    
    def _sample_onpolicy(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        
        recent_ratio = self.her_ratio
    
        if np.random.rand() < recent_ratio:
            max_on_size = min(len(self._episode_fns), self.max_on_size)
            eps_fn = random.choice(self._episode_fns[-max_on_size:])
        else:
            eps_fn = random.choice(self._episode_fns)
        
        episode = self._episodes[eps_fn]

        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = np.concatenate(episode["observation"][max(idx - self._frame_stack, 0) : idx], 0)
        next_obs = np.concatenate(episode["observation"][max(idx + self._nstep - self._frame_stack, 0) : idx + self._nstep], 0)
        if idx < self._frame_stack:
            obs = np.concatenate([*[episode["observation"][0]] * (self._frame_stack - idx), obs], 0)
        if idx + self._nstep < self._frame_stack:
            next_obs = np.concatenate([*[episode["observation"][0]] * (self._frame_stack - idx - self._nstep), next_obs], 0)

        action = episode["action"][idx]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)
    
    def __iter__(self):
        while True:
            yield self._sample_onpolicy()
            
def make_replay_loader_qpa(
    replay_dir, max_size, batch_size, num_workers, save_snapshot, frame_stack, nstep, discount, her_ratio, max_on_size):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBufferQPA(
        replay_dir,
        max_size_per_worker,
        num_workers,
        frame_stack,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        her_ratio=her_ratio,
        max_on_size=max_on_size,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader