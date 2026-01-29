import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import utils
from agent import Agent
EPSILON = 1e-4

def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2)
            # obs : (obs_batch_size, obs_dim)
            # full_obs : (batch_size, obs_dim)
            # dist : (obs_batch_size, batch_size, obs_dim) -> (obs_batch_size, batch_size)
            dists.append(dist)
        
        dists = torch.cat(dists, dim=1)
        # dists : (obs_batch_size, full_obs_size)
        knn_dists = torch.kthvalue(dists, k=k+1, dim=1).values # knn_dists : (obs_batch_size)
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1) # (obs_batch_size, 1)

class SACAgent(Agent):
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg, actor_cfg, discount,
                 init_temperature, alpha_lr, alpha_betas, actor_lr,
                 actor_betas, actor_update_frequency, critic_lr, critic_betas,
                 critic_tau, critic_target_update_frequency, batch_size,
                 learnable_temperature, normalize_state_entropy=True):
        super().__init__()
        
        self.action_range = action_range
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        
        #configs
        self.critic_cfg = critic_cfg
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.actor_cfg = actor_cfg
        self.actor_lr = actor_lr
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        
        #instance
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        
        #optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas)
        self.last_critic_val = -1e-6
        self.train()
        self.critic_target.train()
        
    def train(self, training=True):
        self.training=True
        self.actor.train(training)
        self.critic.train(training)
    
    def reset_critic(self):
        self.critic = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)
    
    def reset_actor(self):
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.actor = hydra.utils.instantiate(self.actor_cfg).to(self.device)
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))
        torch.save(self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step))
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))
        self.critic_target.load_state_dict(torch.load('%s/critic_target_%s.pt' % (model_dir, step)))
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs) #SquashedNormal(mu, std)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range) #(batch_size=1, action_dim)
        assert len(action.shape) == 2 and action.shape[0] == 1
        return utils.to_np(action[0])
    
    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step, print_flag=True):
        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample() #a_t+1 / (batch_size, action_dim)
        log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True) #-H(a_t+1|s_t+1) / (batch_size, 1)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action) #Q_1(s_t+1, a_t+1), Q_2(s_t+1, a_t+1) / (batch_size, 1)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        #V'(s_t+1) = min(Q1, Q2|s_t+1, a_t+1) - alpha * log_pi(a_t+1|s_t+1)
        
        target_Q = (reward + not_done * self.discount * target_V).detach()
        #target : r + gamma * V'_(s_t+1)
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
    def update_critic_with_weights(self, obs, action, reward, next_obs, not_done, weights, logger, step, print_flag=True):
        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample() #a_t+1 / (batch_size, action_dim)
        log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True) #-H(a_t+1|s_t+1) / (batch_size, 1)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action) #Q_1(s_t+1, a_t+1), Q_2(s_t+1, a_t+1) / (batch_size, 1)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        #V'(s_t+1) = min(Q1, Q2|s_t+1, a_t+1) - alpha * log_pi(a_t+1|s_t+1)
        
        target_Q = (reward + not_done * self.discount * target_V).detach()
        current_Q1, current_Q2 = self.critic(obs, action)
        weights = weights.unsqueeze(1)
        critic_loss = ((target_Q-current_Q1).pow(2)*weights).mean() + ((target_Q-current_Q2).pow(2) * weights).mean()
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_critic_state_ent(self, obs, full_obs, action, next_obs, not_done, logger, step, K=5, print_flag=True):
        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample() #a_t+1 / (batch_size, action_dim)
        log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True) #-H(a_t+1|s_t+1) / (batch_size, 1)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action) #Q_1(s_t+1, a_t+1), Q_2(s_t+1, a_t+1) / (batch_size, 1)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        #V'(s_t+1) = min(Q1, Q2|s_t+1, a_t+1) - alpha * log_pi(a_t+1|s_t+1)
        
        ## State Entropy ##
        state_entropy = compute_state_entropy(obs, full_obs, k=K)
        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)
            logger.log('train_critic/entropy_max', state_entropy.max(), step)
            logger.log('train_critic/entropy_min', state_entropy.min(), step)
        
        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std
        
        if print_flag:
            logger.log('train_critic/norm_entropy', norm_state_entropy.mean(), step)
            logger.log('train_critic/norm_entropy_max', norm_state_entropy.max(), step)
            logger.log('train_critic/norm_entropy_min', norm_state_entropy.min(), step)
        
        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy
        ##
        
        target_Q = (state_entropy + not_done * self.discount * target_V).detach()
        #target : s_ent + gamma * V'_(s_t+1)
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        dist = self.actor(obs)
        action = dist.rsample() #a_t~pi(a_t|s_t) / (batch_size, action_dim)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) #log pi(a_t|s_t) / (batch_size, 1)
        
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2) #(batch_size, 1)
        
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        #E[alpha * log_pi(a_t|s_t) - Q(s_t, a_t)], a_t~pi, s_t~Buffer
        if actor_Q.mean().detach().data > self.last_critic_val:
            self.last_critic_val = actor_Q.mean().detach().data
            
        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor.log(logger, step)
        
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()
            
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
    
    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
            print_flag = False
            
            if index == gradient_update - 1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
            
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)
            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)
        
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
    
    # def update_with_likelihood(self, replay_buffer, logger, step, gradient_update=1, large_batch=10, alpha=0.7, reweight=False):
    #     for index in range(gradient_update):
    #         obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(int(self.batch_size * large_batch))
    #         with torch.no_grad():
    #             with utils.eval_mode(self):
    #                 obs = obs
    #                 action = action.clamp(self.action_range[0] + EPSILON, self.action_range[1] - EPSILON)
    #                 dist = self.actor(obs)
    #                 log_prob = dist.log_prob(action).sum(-1)
    #                 prob = torch.exp(log_prob.clamp_(max=88)).cpu().detach()
            
    #         inverse_prob = torch.argsort(prob, descending=True).argsort() + torch.ones_like(prob)
    #         total_prob = (1/inverse_prob).numpy()
    #         total_prob = total_prob ** alpha
    #         total_prob /= total_prob.sum()
    #         idxs = np.random.choice(obs.shape[0], self.batch_size, p=total_prob, replace=False)
            
    #         f_obs, f_action, f_reward, f_next_obs, f_not_done, f_not_done_no_max = obs[idxs], action[idxs], reward[idxs], next_obs[idxs], not_done[idxs], not_done_no_max[idxs]
            
    #         print_flag = False
            
    #         if index == gradient_update - 1:
    #             logger.log('train/batch_reward', f_reward.mean(), step)
    #             print_flag = True
            
    #         # weights = (torch.as_tensor(total_prob[idxs], device=self.device)) ** (-1)
    #         weights = (torch.as_tensor(total_prob[idxs], device=self.device) * self.batch_size * large_batch) ** (-1)
    #         # weights /= weights.max()
    #         if reweight:
    #             self.update_critic_with_weights(f_obs, f_action, f_reward, f_next_obs, f_not_done_no_max, weights, logger, step, print_flag)
    #         else:
    #             self.update_critic(f_obs, f_action, f_reward, f_next_obs, f_not_done_no_max, logger, step, print_flag)
    #         if step % self.actor_update_frequency == 0:
    #             self.update_actor_and_alpha(f_obs, logger, step, print_flag)
        
    #     if step % self.critic_target_update_frequency == 0:
    #         utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
    
    ## CHANGE
    def update_onpolicy_sample(self, replay_buffer, logger, step, size, gradient_update=1, her_ratio=0.5):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(int(self.batch_size * (1-her_ratio)))
            obs_on, action_on, reward_on, next_obs_on, not_done_on, not_done_no_max_on = replay_buffer.sample_onpolicy(int(self.batch_size * her_ratio), size=size)
            print_flag = False
            
            obs = torch.cat([obs, obs_on], axis=0)
            action = torch.cat([action, action_on], axis=0)
            reward = torch.cat([reward, reward_on], axis=0)
            next_obs = torch.cat([next_obs, next_obs_on], axis=0)
            not_done_no_max = torch.cat([not_done_no_max, not_done_no_max_on], axis=0)
            
            if index == gradient_update - 1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
            
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)
            
            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)
        
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
     
    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
            print_flag = False
            
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        self.last_critic_val = -1e-6
        
    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(self.batch_size)
            print_flag = False
            
            if index == gradient_update - 1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
            
            self.update_critic_state_ent(obs, full_obs, action, next_obs, not_done_no_max, logger, step, K, print_flag)
            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)
        
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
            