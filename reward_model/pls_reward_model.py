import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from reward_model.vanilla_reward_model import RewardModel
from utils.utils import eval_mode
from einops import rearrange, reduce
import os
import imageio
from tqdm import tqdm
device = 'cuda'
EPSILON = 1e-4

def gen_net(in_size=1, out_size=1, hidden_size=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, hidden_size))
        net.append(nn.LeakyReLU())
        in_size = hidden_size
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation =='sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    
    return net

class RewardModelPLS(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 device=torch.device('cuda:0'), data_aug_ratio=1, path=None, dataaug_window=5, crop_range=5, use_crop_aug=True, tau_min=0.3, tau_max=0.7, tau_delta=0.1, video_record_path=None, human_label=False, rreset=False, use_min=False):
        super().__init__(obs_dim, action_dim, ensemble_size, lr, mb_size, size_segment, max_size,
                         activation, capacity, large_batch, label_margin,
                         teacher_beta, teacher_gamma, teacher_eps_mistake, teacher_eps_skip, teacher_eps_equal, video_record_path, rreset)
        self.device=device
        self.model = None
        self.path = path
        self.data_aug_ratio = data_aug_ratio if use_crop_aug else 1
        self.use_crop_aug=use_crop_aug
        self.dataaug_window = dataaug_window if use_crop_aug else 0
        self.crop_range = crop_range if use_crop_aug else 0
        self.original_size_segment = size_segment
        self.size_segment = size_segment + 2*dataaug_window
        self.buffer_seg1 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_delta = tau_delta
        self.human_label=human_label
        self.rreset = rreset
        self.learn_step = 0
        self.use_min = use_min
        self.session = 0

                    
    def softXEnt_loss(self, input, target, weights=None):
        if weights is not None:
            log_probs = torch.nn.functional.log_softmax(input, dim=1)
            weighted_log_probs = weights.repeat(2, 1).transpose(1, 0) * log_probs
            return - (target * weighted_log_probs).sum() / input.shape[0]
        return super().softXEnt_loss(input, target)
    
    def reset_parameters(self):
        for model in self.ensemble:
            for layer in model:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
    def reset_subset_parameters(self):
        i = np.random.randint(0,len(self.ensemble))
        model = self.ensemble[i]
        for layer in model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def reset_subset_parameters2(self):
        i = np.random.randint(0,len(self.ensemble))
        for idx, model in enumerate(self.ensemble):
            if idx != i:
                for layer in model:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                
    def get_threshold_tau(self):
        if self.tau_delta < 0:
            return max(self.tau_min, self.tau_max + self.learn_step * self.tau_delta)
        elif (self.tau_delta == 0) or (self.tau_min == self.tau_max):
            return (self.tau_min + self.tau_max) / 2
        else:
            return min(self.tau_max, self.tau_min + self.learn_step * self.tau_delta)
            
    # @torch.no_grad
    def get_queries_with_agent(self, agent, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        save_video = self.video_record_path and len(self.frames) > 0
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        
        batch_index = np.random.choice(max_len, size=mb_size * 2, replace=True)
        log_probs = []
        
        sa_t = train_inputs[batch_index] # (batch_size, episode_len(T), obs_dim + action_dim)
        r_t = train_targets[batch_index] # (batch_size, episode_len(T), 1)
        sa_t = sa_t.reshape(-1, sa_t.shape[-1]) # (batch_size * T, obs_dim + action_dim)
        r_t = r_t.reshape(-1, r_t.shape[-1])
        time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size * 2)])
        time_index = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size * 2, replace=True).reshape(-1, 1)
        sa_t = np.take(sa_t, time_index, axis=0) # Batch x size_seg x dim of s&a
        r_t = np.take(r_t, time_index, axis=0) # Batch x size_seg x 1
        b, l, _ = sa_t.shape
        
        if save_video:
            train_frames = np.array(self.frames[:max_len])
            f_t = train_frames[batch_index]
            b, t, h, w, c = f_t.shape
            f_t = f_t.reshape(-1, h, w, c)
            f_t = np.take(f_t, time_index, axis=0)
        else:
            f_t = None
            
        sa_t = rearrange(sa_t, 'b l d -> (b l) d')
        batch_size = 1000
        total_iter = int(b * l/batch_size)
        if sa_t.shape[0] > total_iter * batch_size:
            total_iter += 1
        
        with torch.no_grad():
            with eval_mode(agent):
                for idx in range(total_iter):
                    last_index = (idx + 1) * batch_size
                    if (idx + 1) * batch_size > sa_t.shape[0]:
                        last_index = sa_t.shape[0]
                    obses = torch.FloatTensor(sa_t[idx * batch_size:last_index,:self.obs_dim]).to(self.device)
                    actions = torch.FloatTensor(sa_t[idx * batch_size:last_index, self.obs_dim:]).to(self.device)
                    actions = actions.clamp(agent.action_range[0] + EPSILON, agent.action_range[1] - EPSILON) ## To avoid Nan Issue
                    dist = agent.actor(obses)
                    log_prob = dist.log_prob(actions).mean(-1, keepdim=True) # (batch_size, 1)
                    # log_prob = log_prob.clamp_(min=-88, max=88)
                    log_probs.append(log_prob)
                log_probs = torch.cat(log_probs, dim=0).cpu().detach()
                
        # avg_log_probs = reduce(rearrange(log_probs, '(b l) d -> b l d', b=b), 'b l d -> b', 'mean')
        # probs = torch.exp(avg_log_probs).numpy()
        
        avg_log_probs = reduce(rearrange(log_probs, '(b l) d -> b l d', b=b), 'b l d -> b', 'mean')
        inverse_probs = torch.argsort(avg_log_probs, descending=True).argsort() + torch.ones_like(avg_log_probs)
        probs = (1 / inverse_probs).numpy()
        tau = self.get_threshold_tau()
        probs = probs ** tau
        probs /= probs.sum()
        sa_t = rearrange(sa_t, '(b l) d -> b l d', b=b)
        return sa_t, r_t, probs, f_t
    
    def pls_sampling(self, agent):
        sa_t, r_t, p_t, f_t = self.get_queries_with_agent(mb_size = 2 * self.mb_size * self.large_batch, agent=agent)
        print(self.get_threshold_tau())
        # p_t /= self.tau
        idxs = np.random.choice(sa_t.shape[0], self.mb_size * 2, p=p_t, replace=False)
        
        np.random.shuffle(idxs)
        sa_t, r_t = sa_t[idxs], r_t[idxs]
        
        # sa_t, r_t = sa_t[idxs], r_t[idxs]
        sa_t_1, sa_t_2 = sa_t[:self.mb_size], sa_t[self.mb_size:]
        r_t_1, r_t_2 = r_t[:self.mb_size], r_t[self.mb_size:]
        if self.video_record_path:
            f_t = f_t[idxs]
            f_t_1, f_t_2 = f_t[:self.mb_size], f_t[self.mb_size:]
            f_cat = np.concatenate((f_t_1, f_t_2), axis=3)
            if self.human_label:
                labels = self.get_human_label(f_cat[:, self.dataaug_window:-self.dataaug_window])
            else:
                sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
         
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        self.learn_step += 1
        self.session += 1
        return len(labels)
    
    def get_human_label(self, f_cat):
        for idx, f_cat_ins in enumerate(f_cat):
            os.makedirs(f'{self.video_record_path}/session{self.session:05d}', exist_ok=True)
            writer = imageio.get_writer(f'{self.video_record_path}/session{self.session:05d}/video{idx:05d}.mp4', fps=20)
            for frame in f_cat_ins:
                writer.append_data(frame)
            writer.close()
        
        labels = []
        label_dict = {'1':(0, 'left'), '2':(1, 'right'), '3':(-1, 'equal')}
        for idx, f_cat_ins in enumerate(f_cat):
            s = False
            while not s:
                reward = input(f'Put Preference session{self.session:05d}/video{idx:05d} (1 (left), 2 (right), 3 (equal)): ').strip()
                try:
                    label, s = label_dict[reward]
                    print(s)
                except:
                    s = False
            labels.append(label)
        labels = np.array(labels).reshape(-1, 1)
        return labels
            
    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        for i in range(w):
            B, S, _ = r_hat1.shape
            length = np.random.randint(self.original_size_segment-self.crop_range, self.original_size_segment+self.crop_range+1, size=B)
            start_index_1 = np.random.randint(0, S+1-length)
            start_index_2 = np.random.randint(0, S+1-length)
            mask_1 = torch.zeros((B,S,1)).to(device)
            mask_2 = torch.zeros((B,S,1)).to(device)
            for b in range(B):
                mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
                mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1
            mask_1_.append(mask_1)
            mask_2_.append(mask_2)

        return torch.cat(mask_1_), torch.cat(mask_2_)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        # for it in tqdm(range(num_iters)):
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                ##
                if self.use_crop_aug:
                    labels = labels.repeat(self.data_aug_ratio)
                ##
                
                if member == 0:
                    total += labels.size(0)
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)

                ##
                if self.use_crop_aug:
                    mask1, mask2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                    r_hat1 = r_hat1.repeat(self.data_aug_ratio, 1, 1)
                    r_hat2 = r_hat2.repeat(self.data_aug_ratio, 1, 1)
                    r_hat1 = (mask1*r_hat1).sum(axis=1)
                    r_hat2 = (mask2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                ##
                
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) # (batch_size, 2)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                    
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index
    
    def train_reward_iter(self, num_iters):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        total = 0
        
        start_index = 0
        for epoch in tqdm(range(num_iters)):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                ##
                if self.use_crop_aug:
                    labels = labels.repeat(self.data_aug_ratio)
                ##
                
                if member == 0:
                    total += labels.size(0)
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)

                ##
                if self.use_crop_aug:
                    mask1, mask2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                    r_hat1 = r_hat1.repeat(self.data_aug_ratio, 1, 1)
                    r_hat2 = r_hat2.repeat(self.data_aug_ratio, 1, 1)
                    r_hat1 = (mask1*r_hat1).sum(axis=1)
                    r_hat2 = (mask2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                ##
                     
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) # (batch_size, 2)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break
        ensemble_acc = ensemble_acc / total
        return ensemble_acc

    
    def cropcat_inter_batch(self, sa_t_1, sa_t_2, target_onehot, w):
        indices_total, mask_1_total, mask_2_total, target_onehot_m_total = [], [], [], []
        B, S, _ = sa_t_1.shape
        for i in range(w):
            indices = torch.randperm(sa_t_1.shape[0])
            indices_total.append(indices)
            l1 = np.random.randint(np.ceil(self.original_size_segment * 0.6), np.ceil(self.original_size_segment * 0.8), size=B)
            l2 = self.original_size_segment - l1

            start_index_1_1 = np.random.randint(0, S+1-l1)
            start_index_2_1 = np.random.randint(0, S+1-l1)

            start_index_1_2 = np.random.randint(S, 2 * S + 1 -l2)
            start_index_2_2 = np.random.randint(S, 2 * S + 1 -l2)

            mask_1 = torch.zeros((B, 2 * S, 1))
            mask_2 = torch.zeros((B, 2 * S, 1))

            for b in range(B):
                m_id1 = list(range(start_index_1_1[b],start_index_1_1[b] + l1[b])) + list(range(start_index_1_2[b],start_index_1_2[b] + l2[b]))
                m_id2 = list(range(start_index_2_1[b],start_index_2_1[b] + l1[b])) + list(range(start_index_2_2[b],start_index_2_2[b] + l2[b]))
                mask_1[b, m_id1] = 1
                mask_2[b, m_id2] = 1
                
            mask_1_total.append(mask_1)
            mask_2_total.append(mask_2)
            
            target_onehot_m = torch.tensor(l1.reshape(-1, 1)) * target_onehot + torch.tensor(l2.reshape(-1, 1)) * target_onehot[indices]
            target_onehot_m = target_onehot_m / target_onehot_m.sum(axis=-1, keepdims=True)
            target_onehot_m_total.append(target_onehot_m)
        
        return torch.cat(indices_total), torch.cat(mask_1_total), torch.cat(mask_2_total), torch.cat(target_onehot_m_total)


    def cropcat_intra_batch(self, sa_t_1, sa_t_2, target_onehot, w):
        mask_1_total, mask_2_total, target_onehot_m_total = [], [], []
        B, S, _ = sa_t_1.shape
        for i in range(w):
            l1 = np.random.randint(np.ceil(self.original_size_segment * 0.6), np.ceil(self.original_size_segment * 0.9), size=B)
            l2 = self.original_size_segment - l1

            start_index_1_1 = np.random.randint(0, S+1-l1)
            start_index_2_1 = np.random.randint(0, S+1-l1)

            start_index_1_2 = np.random.randint(S, 2 * S + 1 -l2)
            start_index_2_2 = np.random.randint(S, 2 * S + 1 -l2)

            mask_1 = torch.zeros((B, 2 * S, 1))
            mask_2 = torch.zeros((B, 2 * S, 1))

            for b in range(B):
                m_id1 = list(range(start_index_1_1[b],start_index_1_1[b] + l1[b])) + list(range(start_index_1_2[b],start_index_1_2[b] + l2[b]))
                m_id2 = list(range(start_index_2_1[b],start_index_2_1[b] + l1[b])) + list(range(start_index_2_2[b],start_index_2_2[b] + l2[b]))
                mask_1[b, m_id1] = 1
                mask_2[b, m_id2] = 1
                
            mask_1_total.append(mask_1)
            mask_2_total.append(mask_2)
            
            target_onehot_m = torch.tensor(l1.reshape(-1, 1)) * target_onehot + torch.tensor(l2.reshape(-1, 1)) * (1-target_onehot)
            target_onehot_m = target_onehot_m / target_onehot_m.sum(axis=-1, keepdims=True)
            target_onehot_m_total.append(target_onehot_m)
        
        return torch.cat(mask_1_total), torch.cat(mask_2_total), torch.cat(target_onehot_m_total)
    
    
    def train_reward_iter_cropcat_inter(self, num_iters):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        total = 0
        
        start_index = 0
        for epoch in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()
                
                if member == 0:
                    total += labels.size(0)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros((labels.size(0), 2)).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                    
                #
                indices, mask_1, mask_2, target_onehot_m = self.cropcat_inter_batch(sa_t_1, sa_t_2, target_onehot, self.data_aug_ratio)
                # target_onehot_m = target_onehot_m.repeat(self.data_aug_ratio)
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1_m = torch.cat([r_hat1.repeat(self.data_aug_ratio, 1, 1), r_hat1[indices]], axis=1) # (b * N, 2 * S, 1)
                r_hat2_m = torch.cat([r_hat2.repeat(self.data_aug_ratio, 1, 1), r_hat2[indices]], axis=1) # (b * N, 2 * S, 1)
                r_hat_1_m = (mask_1.to(device) * r_hat1_m).sum(axis=1) # (b * N, 1)
                r_hat_2_m = (mask_2.to(device) * r_hat2_m).sum(axis=1) # (b * N, 1)
                r_hat_1 = r_hat1.sum(axis=1) # (b, 1)
                r_hat_2 = r_hat2.sum(axis=1) # (b, 1)
                
                r_hat = torch.cat([r_hat_1, r_hat_2], axis=-1)
                r_hat_m = torch.cat([r_hat_1_m, r_hat_2_m], axis=-1)
                curr_loss = self.softXEnt_loss(r_hat, target_onehot.to(device)) + self.softXEnt_loss(r_hat_m, target_onehot_m.to(device))
                #
                
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1)
                _, true = torch.max(target_onehot.to(device).data, 1)
                correct = (predicted == true).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def train_reward_cropcat_inter(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()

                if member == 0:
                    total += labels.size(0)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros(labels.size(0), 2).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin                
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                    
                #
                indices, mask_1, mask_2, target_onehot_m = self.cropcat_inter_batch(sa_t_1, sa_t_2, target_onehot, self.data_aug_ratio)
                # target_onehot_m = target_onehot_m.repeat(self.data_aug_ratio)
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1_m = torch.cat([r_hat1.repeat(self.data_aug_ratio, 1, 1), r_hat1[indices]], axis=1) # (b * N, 2 * S, 1)
                r_hat2_m = torch.cat([r_hat2.repeat(self.data_aug_ratio, 1, 1), r_hat2[indices]], axis=1) # (b * N, 2 * S, 1)
                r_hat_1_m = (mask_1.to(device) * r_hat1_m).sum(axis=1) # (b * N, 1)
                r_hat_2_m = (mask_2.to(device) * r_hat2_m).sum(axis=1) # (b * N, 1)
                r_hat_1 = r_hat1.sum(axis=1) # (b, 1)
                r_hat_2 = r_hat2.sum(axis=1) # (b, 1)
                
                r_hat = torch.cat([r_hat_1, r_hat_2], axis=-1)
                r_hat_m = torch.cat([r_hat_1_m, r_hat_2_m], axis=-1)
                curr_loss = self.softXEnt_loss(r_hat, target_onehot.to(device)) + self.softXEnt_loss(r_hat_m, target_onehot_m.to(device))
                #
                
                loss += curr_loss
                    
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1)
                _, true = torch.max(target_onehot.to(device).data, 1)
                correct = (predicted == true).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def train_reward_iter_cropcat_intra(self, num_iters):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        total = 0
        
        start_index = 0
        for epoch in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()
                
                if member == 0:
                    total += labels.size(0)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros((labels.size(0), 2)).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                    
                #
                mask_1, mask_2, target_onehot_m = self.cropcat_intra_batch(sa_t_1, sa_t_2, target_onehot, self.data_aug_ratio)
                # target_onehot_m = target_onehot_m.repeat(self.data_aug_ratio)
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1_m = torch.cat([r_hat1.repeat(self.data_aug_ratio, 1, 1), r_hat2.repeat(self.data_aug_ratio, 1, 1)], axis=1) # (b * N, 2 * S, 1)
                r_hat2_m = torch.cat([r_hat2.repeat(self.data_aug_ratio, 1, 1), r_hat1.repeat(self.data_aug_ratio, 1, 1)], axis=1) # (b * N, 2 * S, 1)
                r_hat_1_m = (mask_1.to(device) * r_hat1_m).sum(axis=1) # (b * N, 1)
                r_hat_2_m = (mask_2.to(device) * r_hat2_m).sum(axis=1) # (b * N, 1)
                r_hat_1 = r_hat1.sum(axis=1) # (b, 1)
                r_hat_2 = r_hat2.sum(axis=1) # (b, 1)
                
                r_hat = torch.cat([r_hat_1, r_hat_2], axis=-1)
                r_hat_m = torch.cat([r_hat_1_m, r_hat_2_m], axis=-1)
                curr_loss = self.softXEnt_loss(r_hat, target_onehot.to(device)) + self.softXEnt_loss(r_hat_m, target_onehot_m.to(device))
                #
                
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1)
                _, true = torch.max(target_onehot.to(device).data, 1)
                correct = (predicted == true).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.optimizer.step()
            
            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def train_reward_cropcat_intra(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()

                if member == 0:
                    total += labels.size(0)

                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros(labels.size(0), 2).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin                
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                    
                #
                
                mask_1, mask_2, target_onehot_m = self.cropcat_intra_batch(sa_t_1, sa_t_2, target_onehot, self.data_aug_ratio)
                # target_onehot_m = target_onehot_m.repeat(self.data_aug_ratio)
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1_m = torch.cat([r_hat1.repeat(self.data_aug_ratio, 1, 1), r_hat2.repeat(self.data_aug_ratio, 1, 1)], axis=1) # (b * N, 2 * S, 1)
                r_hat2_m = torch.cat([r_hat2.repeat(self.data_aug_ratio, 1, 1), r_hat1.repeat(self.data_aug_ratio, 1, 1)], axis=1) # (b * N, 2 * S, 1)
                r_hat_1_m = (mask_1.to(device) * r_hat1_m).sum(axis=1) # (b * N, 1)
                r_hat_2_m = (mask_2.to(device) * r_hat2_m).sum(axis=1) # (b * N, 1)
                r_hat_1 = r_hat1.sum(axis=1) # (b, 1)
                r_hat_2 = r_hat2.sum(axis=1) # (b, 1)
                
                r_hat = torch.cat([r_hat_1, r_hat_2], axis=-1)
                r_hat_m = torch.cat([r_hat_1_m, r_hat_2_m], axis=-1)
                curr_loss = self.softXEnt_loss(r_hat, target_onehot) + self.softXEnt_loss(r_hat_m, target_onehot_m.to(device))
                #
                
                loss += curr_loss
                    
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1)
                _, true = torch.max(target_onehot.to(device).data, 1)
                correct = (predicted == true).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def manifold_mixup_batch(self, model, sa_t_1, sa_t_2, target_onehot, alpha):
        indices = torch.randperm(sa_t_1.shape[0])
        lmda = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(self.device)
        target_onehot = target_onehot.to(self.device)
        sa_t_1 = torch.from_numpy(sa_t_1).float().to(self.device) #(b, l, d)
        sa_t_2 = torch.from_numpy(sa_t_2).float().to(self.device) #(b, l, d)
        mixup_layer_idx = np.random.randint(4) * 2
        for i, layer in enumerate(model):
            if i == mixup_layer_idx:
                sa_t_1 = sa_t_1 * lmda + sa_t_1[indices] * (1-lmda)
                sa_t_2 = sa_t_2 * lmda + sa_t_2[indices] * (1-lmda)
            sa_t_1 = layer(sa_t_1) #(b, l, d')
            sa_t_2 = layer(sa_t_2) #(b, l, d')
            
        r_hat_m1 = sa_t_1.sum(axis=1)
        r_hat_m2 = sa_t_2.sum(axis=1)
        r_hat_m = torch.cat([r_hat_m1, r_hat_m2], axis=1)
        target_onehot_m = target_onehot * lmda + target_onehot[indices] * (1-lmda)
        return r_hat_m, target_onehot_m
    
    def train_reward_iter_manifold_mixup(self, num_iters, alpha):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        total = 0
        
        start_index = 0
        for epoch in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()
                
                if member == 0:
                    total += labels.size(0)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros((labels.size(0), 2)).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                target_onehot = target_onehot.to(self.device)
                ## Mix up Data : sa_t_1_m, sa_t_2_m, target_onehot_m
                r_hat_m, target_onehot_m = self.manifold_mixup_batch(self.ensemble[member], sa_t_1, sa_t_2, target_onehot, alpha)
                
                concat_target_onehot = torch.cat([target_onehot, target_onehot_m], axis=0)
                
                ## Predict
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1) # (batch_size, 1)
                r_hat2 = r_hat2.sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                concat_r_hat = torch.cat([r_hat, r_hat_m], axis=0)
                curr_loss = self.softXEnt_loss(concat_r_hat, concat_target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat[:labels.size(0)].data, 1) # (batch_size, )
                correct = (predicted == labels.to(device)).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def train_reward_manifold_mixup(self, alpha):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()
                
                if member == 0:
                    total += labels.size(0)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros((labels.size(0), 2)).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                target_onehot = target_onehot.to(self.device)
                ## Mix up Data : sa_t_1_m, sa_t_2_m, target_onehot_m
                r_hat_m, target_onehot_m = self.manifold_mixup_batch(self.ensemble[member], sa_t_1, sa_t_2, target_onehot, alpha)
                
                concat_target_onehot = torch.cat([target_onehot.to(self.device), target_onehot_m], axis=0)
                
                ## Predict
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1) # (batch_size, 1)
                r_hat2 = r_hat2.sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                concat_r_hat = torch.cat([r_hat, r_hat_m], axis=0)
                curr_loss = self.softXEnt_loss(concat_r_hat, concat_target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat[:labels.size(0)].data, 1) # (batch_size, )
                correct = (predicted == labels.to(device)).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
        ensemble_acc = ensemble_acc / total
        return ensemble_acc