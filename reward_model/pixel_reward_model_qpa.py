import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
# import utils

from agent.drqv2 import Encoder, RandomShiftsAug
from utils.replay_buffer_drqv2 import episode_len
from scipy.stats import norm
from .pixel_reward_model import RewardModel
device = 'cuda'
# device = 'cpu'

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class RewardPredictor(nn.Module):
    def __init__(
        self, 
        obs_shape, 
        action_shape, 
        feature_dim=50,
        hidden_dim=256, 
        hidden_depth=2, 
        activation='tanh'):
        
        super().__init__()
        self.encoder = Encoder(obs_shape)

        self.reward_model = mlp(
            self.encoder.repr_dim + action_shape[0], hidden_dim, 1, hidden_depth
        )
        if activation == 'tanh':
            self.output_act = nn.Tanh()
        elif activation == 'sig':
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.ReLU()
        
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, feature=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs)
        if detach_encoder:
            obs = obs.detach()

        obs_action = torch.cat([obs, action], dim=-1)
        if feature:
            return obs_action
        pred_reward = self.output_act(self.reward_model(obs_action))

        return pred_reward
    
def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index
    
def KMeans(x, K=3, Niter=50, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()    # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:,None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        
    center_index = D_ij.argmin(dim=0).long().view(-1)
    center = x[center_index]
    
    return  center_index


class RewardModelQPA(RewardModel):
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=10, activation='tanh', capacity=2000, 
                 teacher_type=0, teacher_noise=0.0, 
                 teacher_margin=0.0, teacher_thres=0.0, 
                 large_batch=1, label_margin=0.0, stack=1, 
                 img_shift=0,
                 time_shift=0,
                 time_crop=0,
                 device=torch.device('cuda:0'),
                 aug_ratio=10):
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size # maximum # of episodes for training
        self.activation = activation
        # frame stack
        self.stack = stack
        # augmentation
        self.img_aug = RandomShiftsAug(pad=img_shift)
        self.time_shift = time_shift
        self.time_crop = time_crop

        self.original_size_segment = size_segment
        self.size_segment = size_segment + (stack - 1) + 2 * time_shift
        self.original_stack_index = [list(range(i, i+stack)) for i in range(size_segment)]
        self.stack_index = [list(range(i, i+stack)) for i in range(size_segment + 2 * time_shift)]
        self.stack_index_torch = torch.LongTensor([list(range(i, i+stack)) for i in range(size_segment + 2 * time_shift)]).to(device)

        self.capacity = int(capacity)
        self.buffer_seg1_index = np.empty((self.capacity, 2), dtype=np.uint32)
        self.buffer_seg2_index = np.empty((self.capacity, 2), dtype=np.uint32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        
        self.construct_ensemble()
        # self.inputs = []
        # self.actions = []
        # self.targets = []
        
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 16
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.teacher_type = teacher_type
        self.teacher_noise = teacher_noise
        self.teacher_margin = teacher_margin
        self.teacher_thres = teacher_thres
        self.large_batch = large_batch
        
        file_name = os.getcwd()+'/sampling_log.txt'
        self.f_io = open(file_name, 'a')
        self.round_counter = 0
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
        self.replay_loader = None
        self.device = device
        self.aug_ratio = aug_ratio
        
    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        for i in range(w):
            B, L, _ = r_hat1.shape
            length = np.random.randint(self.original_size_segment-self.time_crop, self.original_size_segment+self.time_crop+1, size=B)
            start_index_1 = np.random.randint(0, L+1-length)
            start_index_2 = np.random.randint(0, L+1-length)
            mask_1 = np.zeros((B,L,1), dtype=np.float32)
            mask_2 = np.zeros((B,L,1), dtype=np.float32)
            for b in range(B):
                mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
                mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1
            mask_1_.append(torch.from_numpy(mask_1).to(self.device))
            mask_2_.append(torch.from_numpy(mask_2).to(self.device))
        return torch.cat(mask_1_), torch.cat(mask_2_)

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                self.opt.zero_grad()

                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                idx_1 = self.buffer_seg1_index[idxs]
                idx_2 = self.buffer_seg2_index[idxs]
                s_t_1, a_t_1, _ = self.replay_loader.dataset.get_segment_batch(idx_1, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)
                s_t_2, a_t_2, _ = self.replay_loader.dataset.get_segment_batch(idx_2, self.size_segment - (self.stack - 1))
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device) ## (B)
                temp_batch_size = labels.size(0)
                if member == 0:
                    if self.time_shift > 0 or self.time_crop > 0:
                        total += labels.size(0) * self.aug_ratio
                    else:
                        total += labels.size(0)
                        
                s_t_1 = torch.as_tensor(s_t_1, device=self.device).float()
                s_t_2 = torch.as_tensor(s_t_2, device=self.device).float()
                # image augmentation
                if self.img_aug.pad > 0:
                    orig_shape = s_t_1.shape
                    s_t_1 = s_t_1.reshape(orig_shape[0], -1, *orig_shape[3:]) ## (B, C*(L+S-1), H, W)
                    s_t_2 = s_t_2.reshape(orig_shape[0], -1, *orig_shape[3:])
                    s_t_1 = self.img_aug(s_t_1)
                    s_t_2 = self.img_aug(s_t_2)
                    s_t_1 = s_t_1.reshape(orig_shape) ## (B, L+S-1, C, H, W)
                    s_t_2 = s_t_2.reshape(orig_shape)

                # frame stacking
                if self.stack > 1:
                    s_t_1 = torch.index_select(s_t_1, dim=1, index=self.stack_index_torch.flatten())
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_1 = s_t_1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)
                    s_t_2 = torch.index_select(s_t_2, dim=1, index=self.stack_index_torch.flatten())
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack, self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S, C, H, W)
                    s_t_2 = s_t_2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), self.stack*self.ds[0], self.ds[1], self.ds[2]) ## (B, L, S*C, H, W)

                    # a_t_1 = a_t_1[:, self.stack-1:]
                    # a_t_2 = a_t_2[:, self.stack-1:]

                # get logits
                s_t_1 = s_t_1.reshape(-1, *s_t_1.shape[2:]) ## (B*L, S*C, H, W)
                a_t_1 = a_t_1.reshape(-1, a_t_1.shape[-1]) ## (B*L, A)
                s_t_2 = s_t_2.reshape(-1, *s_t_2.shape[2:])
                a_t_2 = a_t_2.reshape(-1, a_t_2.shape[-1])
                r_hat1 = self.r_hat_member(s_t_1, a_t_1, member=member) ## (B*L, 1)
                r_hat2 = self.r_hat_member(s_t_2, a_t_2, member=member)
                r_hat1 = r_hat1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1) ## (B, L, 1)
                r_hat2 = r_hat2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1)
                
                # shifting & cropping time
                # if self.time_shift > 0 or self.time_crop > 0:
                #     mask_1, mask_2 = self.get_cropping_mask(r_hat1, r_hat2)
                #     r_hat1 = (mask_1*r_hat1).sum(axis=1) ## (B, 1)
                #     r_hat2 = (mask_2*r_hat2).sum(axis=1)
                # else:
                #     r_hat1 = r_hat1.sum(axis=1) ## (B, 1)
                #     r_hat2 = r_hat2.sum(axis=1)
                
                if self.time_shift > 0 or self.time_crop > 0:
                    labels = labels.repeat(self.aug_ratio)
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.aug_ratio)
                    r_hat1 = r_hat1.repeat(self.aug_ratio, 1, 1)
                    r_hat2 = r_hat2.repeat(self.aug_ratio, 1, 1)
                    r_hat1 = (mask_1*r_hat1).sum(axis=1) ## (B, 1)
                    r_hat2 = (mask_2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1) ## (B, 1)
                    r_hat2 = r_hat2.sum(axis=1)
                    
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) ## (B, 2)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                curr_loss.backward()
                self.opt.step()
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc