import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from reward_model.vanilla_reward_model import RewardModel
device = 'cuda'

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

## TO-BE FIXED
class RewardModelQPA(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0,
                 device=torch.device('cuda:0'), data_aug_ratio=1, path=None, dataaug_window=5, crop_range=5, video_record_path=None):
        super().__init__(obs_dim, action_dim, ensemble_size, lr, mb_size, size_segment, max_size,
                         activation, capacity, large_batch, label_margin,
                         teacher_beta, teacher_gamma, teacher_eps_mistake, teacher_eps_skip, teacher_eps_equal, video_record_path)
        
        
        self.device=device
        self.model = None
        self.path = path
        self.data_aug_ratio = data_aug_ratio
        self.buffer_mask = np.ones((self.capacity, 1), dtype=np.float32)
        
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128

        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []

        self.dataaug_window = dataaug_window
        self.crop_range = crop_range
        self.original_size_segment = size_segment
        self.size_segment = size_segment + 2*dataaug_window
        self.buffer_seg1 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.count=0
        
    def eval(self):
        for model in self.ensemble:
            model.eval()

    def get_queries(self, mb_size=20, get_frame=True):
        self.count += 1
        return super().get_queries(mb_size, get_frame)
    
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat=None):
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = super().get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat)
        
        if self.path:
            sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
            r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
            sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
            r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
            np.save(sa_t_1_path, sa_t_1)
            np.save(r_t_1_path, r_t_1)
            np.save(sa_t_2_path, sa_t_2)
            np.save(r_t_2_path, r_t_2)
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
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
                if self.data_aug_ratio:
                    labels = labels.repeat(self.data_aug_ratio)
                ##
                
                if member == 0:
                    total += labels.size(0)
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)

                ##
                if self.data_aug_ratio:
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
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    curr_loss = self.CELoss(r_hat, labels)
                    loss += curr_loss
                else:
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
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                ##
                if self.data_aug_ratio:
                    labels = labels.repeat(self.data_aug_ratio)
                ##
                
                if member == 0:
                    total += labels.size(0)
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)

                ##
                if self.data_aug_ratio:
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
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    curr_loss = self.CELoss(r_hat, labels)
                    loss += curr_loss
                else:
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