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
import gc
# import utils

from agent.drqv2 import Encoder, RandomShiftsAug
from utils.replay_buffer_drqv2 import episode_len
from .pixel_reward_model import RewardModel, RewardPredictor
from scipy.stats import norm
from einops import rearrange, reduce
import utils.utils_drqv2 as utils
EPSILON=1e-6
# device = 'cuda'
# device = 'cpu'


# class RewardPredictorwithReset(RewardPredictor):
#     def __init__(self, obs_shape, action_shape, feature_dim=50, hidden_dim=256, hidden_depth=2, activation='tanh'):
#         super().__init__()
    
#     def reset_reward_model(self):
#         """Reset only the parameters of reward_model."""
#         self.reward_model.apply(weight_init)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class RewardModelPLS(RewardModel):
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=20, activation='tanh', capacity=2000, 
                 teacher_type=0, teacher_noise=0.0, 
                 teacher_margin=0.0, teacher_thres=0.0, 
                 large_batch=1, label_margin=0.0, stack=1, 
                 img_shift=0,
                 time_shift=0,
                 time_crop=0,
                 tau_max=1.0,
                 tau_min=1.0,
                 tau_delta=0.0,
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
        self.device = device
        self.stack_index_torch = torch.LongTensor([list(range(i, i+stack)) for i in range(size_segment + 2 * time_shift)]).to(self.device)

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
        
        self.tau_min=tau_min
        self.tau_max=tau_max
        self.tau_delta=tau_delta
        
        self.aug_ratio = aug_ratio
        
    def get_threshold_tau(self):
        if self.tau_delta < 0:
            return max(self.tau_min, self.tau_max + self.learn_step * self.tau_delta)
        elif (self.tau_delta == 0) or (self.tau_min == self.tau_max):
            return (self.tau_min + self.tau_max) / 2
        else:
            return min(self.tau_max, self.tau_min + self.learn_step * self.tau_delta)
    
    def reset_parameters(self):
        for model in self.ensemble:
            model.reward_model.apply(weight_init)
            print('.')
            
    def get_queries_mono(self, mb_size=20):
        self.replay_loader.dataset.try_fetch_instant()
        max_len = min(len(self.replay_loader.dataset._episode_fns), self.max_size)
        assert max_len > 0
        len_traj = episode_len(self.replay_loader.dataset._episodes[self.replay_loader.dataset._episode_fns[0]])
        
        batch_index = np.random.choice(max_len, size=mb_size, replace=True).reshape(-1,1) + (len(self.replay_loader.dataset._episode_fns) - max_len)
        time_index = np.random.choice(len_traj-self.size_segment + 1, size=mb_size, replace=True).reshape(-1,1) + 1
        return np.concatenate([batch_index, time_index], axis=1).astype(np.uint32)
                
    def pls_sampling(self, agent, step):
        torch.cuda.empty_cache()
        gc.collect()
        # get queries
        idx =  self.get_queries_mono(mb_size=self.mb_size*self.large_batch * 2)
        s_t, a_t, r_t = self.replay_loader.dataset.get_segment_batch(idx, self.size_segment - (self.stack - 1)) ## (B, L+S-1, C, H, W), (B, L, A)

        if self.time_shift > 0:
            probs = self.get_policy_likelihoods(s_t[:, self.time_shift:-self.time_shift], a_t[:, self.time_shift:-self.time_shift], agent, step)
        else:
            probs = self.get_policy_likelihoods(s_t, a_t, agent, step)
            
        # probs = self.get_policy_likelihoods(s_t, a_t, agent, step)s_t, a_t, agent, step)
        print(self.get_threshold_tau())
        selected_index = np.random.choice(s_t.shape[0], self.mb_size * 2, p=probs, replace=False)
        np.random.shuffle(selected_index)
        s_t_1, a_t_1, r_t_1 = s_t[selected_index][:self.mb_size], a_t[selected_index][:self.mb_size], r_t[selected_index][:self.mb_size]
        s_t_2, a_t_2, r_t_2 = s_t[selected_index][self.mb_size:], a_t[selected_index][self.mb_size:], r_t[selected_index][self.mb_size:]
        
        
        # get labels
        s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2, labels = self.get_label(
            s_t_1, s_t_2, a_t_1, a_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(idx[selected_index][:self.mb_size], idx[selected_index][self.mb_size:], labels)
        
        return len(labels)
    
    # def get_policy_likelihoods(self, s_t, a_t, agent, step, batch_size=1024): # DMControl
    def get_policy_likelihoods(self, s_t, a_t, agent, step, batch_size=128): # Metaworld
        # s_t : B, L + S -1, C, H, W -> B, L, S, C, H, W
        s_t = np.take(s_t, self.original_stack_index, axis=1)
        
        # s_t = np.take(s_t, self.stack_index, axis=1)
        b_s, l = s_t.shape[0], s_t.shape[1]
        
        s_t = rearrange(s_t, 'b l s c h w -> (b l) (s c) h w')
        a_t = rearrange(a_t, 'b l w -> (b l) w')
        # batch_size = 4096
        total_iter = int(b_s * l /batch_size)
        log_probs = []
        
        if s_t.shape[0] > total_iter * batch_size:
            total_iter += 1
            
        with torch.no_grad(), utils.eval_mode(agent):
            for idx in range(total_iter):
                last_index = (idx + 1) * batch_size
                if (idx + 1) * batch_size > s_t.shape[0]:
                    last_index = s_t.shape[0]
                obses = torch.as_tensor(s_t[idx * batch_size:last_index],device=self.device)
                # obses = agent.aug(obses)
                obses = agent.encoder(obses)
                actions = torch.as_tensor(a_t[idx * batch_size:last_index],device=self.device)
                actions = actions.clamp(-1.0 + EPSILON, 1.0 - EPSILON)
                
                stddev = utils.schedule(agent.stddev_schedule, step)
                dist = agent.actor(obses, stddev)
            
                log_prob = dist.log_prob(actions).mean(-1, keepdim=True)
                log_probs.append(log_prob)
            log_probs = torch.cat(log_probs, dim=0).cpu().detach()
        
        avg_log_probs = reduce(rearrange(log_probs, '(b l) d-> b l d', b=b_s), 'b l d -> b', 'mean')
        print(f'max value {max(avg_log_probs)} | min value {min(avg_log_probs)}')
        inverse_probs = torch.argsort(avg_log_probs, descending=True).argsort() + torch.ones_like(avg_log_probs)
        probs = (1/inverse_probs).numpy()
        tau = self.get_threshold_tau()
        probs = probs ** tau
        probs /= probs.sum()
        
        del obses, actions, dist, log_probs, inverse_probs
        torch.cuda.empty_cache()
        gc.collect()
        
        return probs
    
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
                # get logits
                s_t_1 = s_t_1.reshape(-1, *s_t_1.shape[2:]) ## (B*L, S*C, H, W)
                a_t_1 = a_t_1.reshape(-1, a_t_1.shape[-1]) ## (B*L, A)
                s_t_2 = s_t_2.reshape(-1, *s_t_2.shape[2:])
                a_t_2 = a_t_2.reshape(-1, a_t_2.shape[-1])
                r_hat1 = self.r_hat_member(s_t_1, a_t_1, member=member) ## (B*L, 1)
                r_hat2 = self.r_hat_member(s_t_2, a_t_2, member=member)
                r_hat1 = r_hat1.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1) ## (B, L, 1)
                r_hat2 = r_hat2.reshape(temp_batch_size, self.size_segment - (self.stack - 1), -1)
                
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
        del r_hat1, r_hat2, s_t_1, s_t_2, a_t_1, a_t_2, mask_1, mask_2, curr_loss, r_hat, labels
        gc.collect()
        torch.cuda.empty_cache()
        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc