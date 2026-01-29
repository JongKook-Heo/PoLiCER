import numpy as np
import torch
from utils.replay_buffer import ReplayBuffer

class ReplayBufferQPA(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, device, window=1, max_episode_len=1000):
        super().__init__(obs_shape, action_shape, capacity, device, window)
        self.max_episode_len = max_episode_len
    
    def sample_onpolicy(self, batch_size, size):
        idxs = np.random.randint(self.idx-size*self.max_episode_len, self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
