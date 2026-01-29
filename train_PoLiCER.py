import os
os.environ["PATH"] += os.pathsep + '/root/.mujoco/mujoco210/bin'
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ['MUJOCO_GL']='egl'
import metaworld_env as mw
from pathlib import Path
from collections import deque
import hydra
import numpy as np
import torch
import torch.nn as nn
from dm_env import specs
import dmc
import utils.utils_drqv2 as utils
from utils.logger_drqv2 import Logger
from utils.replay_buffer_drqv2 import ReplayBufferStorage, make_replay_loader
from video import VideoRecorder, TrainVideoRecorder
from reward_model.pixel_reward_model_pls import RewardModelPLS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    C, H, W = obs_spec.shape
    cfg.agent.obs_shape = (C*cfg.frame_stack, H, W)
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)

class RandomEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, hidden_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        
        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg,
        )
        
        self.random_encoder = RandomEncoder(self.train_env.observation_spec().shape, 
                                        cfg.agent.feature_dim, cfg.agent.hidden_dim).to(self.device)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.log_success = ('metaworld' in self.cfg.task_name)
        
    def setup(self):
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, agent='drqv2')
        if 'metaworld' in self.cfg.task_name:
            # dummy
            self.dummy_env = dmc.make('walker_walk', 1, self.cfg.action_repeat, self.cfg.seed)
            self.dummy_env.reset()
            #
            
            self.train_env = mw.make(
                self.cfg.task_name,
                1,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            self.eval_env = mw.make(
                self.cfg.task_name,
                1,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            # del self.dummy_env
        else:
            self.train_env = dmc.make(
                self.cfg.task_name,
                1,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            self.eval_env = dmc.make(
                self.cfg.task_name,
                1,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "true_reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            True,
            self.cfg.frame_stack,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._replay_iter = None

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0

        # instantiating the reward model
        self.reward_model = RewardModelPLS(
            self.train_env.observation_spec().shape,
            self.train_env.action_spec().shape,
            ensemble_size=self.cfg.ensemble_size,
            size_segment=self.cfg.segment,
            activation=self.cfg.activation, 
            lr=self.cfg.reward_lr,
            mb_size=self.cfg.reward_batch,
            max_size=self.cfg.reward_max_episodes,
            capacity=self.cfg.reward_capacity,
            teacher_type=self.cfg.teacher_type, 
            teacher_noise=self.cfg.teacher_noise, 
            teacher_margin=self.cfg.teacher_margin,
            teacher_thres=self.cfg.teacher_thres, 
            large_batch=self.cfg.large_batch, 
            label_margin=self.cfg.label_margin,
            stack=self.cfg.frame_stack if self.cfg.reward_stack else 1,
            img_shift=self.cfg.img_shift,
            time_shift=self.cfg.time_shift,
            time_crop=self.cfg.time_crop,
            tau_min=self.cfg.tau_min,
            tau_max=self.cfg.tau_max,
            tau_delta=self.cfg.tau_delta,
            aug_ratio=self.cfg.aug_ratio)
        
        self.reward_model.replay_loader = self.replay_loader

        # if 'metaworld' not in self.cfg.task_name:
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )
        
        self.last_feedback_session_time = self.cfg.num_seed_frames + self.cfg.num_unsup_frames + self.cfg.num_interact * (int(self.cfg.max_feedback / self.cfg.reward_batch) - 1)
        self.feedback_period = list(range(self.cfg.num_seed_frames + self.cfg.num_unsup_frames,
                                          self.last_feedback_session_time + 1, self.cfg.num_interact))
        self.feedback_period = self.feedback_period[1:]
        print(self.feedback_period)
        self.replay_ratio = 1
        self.increase_q = self.cfg.increase_q
        self.reset_threshold_q = self.cfg.init_k
        print(f'Initial Reset Threshold Q : {self.reset_threshold_q}, init_step {self.increase_q}, stepping by {self.cfg.step_k}')
        print(f'Replay Ratio : {self.replay_ratio}')
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        if self.log_success:
            success_rate = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            if self.log_success:
                episode_success = 0
            time_step = self.eval_env.reset()
            # if 'metaworld' not in self.cfg.task_name:
            #     self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            self.video_recorder.init(self.eval_env, enabled=(episode == 0), metaworld='metaworld' in self.cfg.task_name)
            observations = deque(maxlen=self.cfg.frame_stack)
            for _ in range(self.cfg.frame_stack):
                observations.append(time_step.observation)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        np.concatenate(observations, 0),
                        self.global_step,
                        eval_mode=True,
                    )
                time_step = self.eval_env.step(action)
                # if 'metaworld' not in self.cfg.task_name:
                #     self.video_recorder.record(self.eval_env)
                self.video_recorder.record(self.eval_env, 'metaworld' in self.cfg.task_name)
                
                total_reward += time_step.reward
                step += 1
                observations.append(time_step.observation)
                if self.log_success:
                    episode_success = max(episode_success, time_step['success'])

            episode += 1
            if self.log_success:
                success_rate += episode_success
            # if 'metaworld' not in self.cfg.task_name:
            #     self.video_recorder.save(f"{self.global_frame}.mp4")
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            if self.log_success:
                log("success_rate", success_rate * 100.0 / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def learn_reward(self, first_flag=0):
        print(self.global_frame)
                
        if self.reward_model.get_threshold_tau() > 0.:
            labeled_queries = self.reward_model.pls_sampling(self.agent, self.global_step)
        else:
            labeled_queries = self.reward_model.uniform_sampling()
                
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        debug_loss1, debug_loss2, bce_loss, train_acc = 0, 0, 0, 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0:
                    raise NotImplementedError
                else:
                    ensemble_losses, debug_loss1, debug_loss2, train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if (epoch % 10 == 0) or (epoch==self.cfg.reward_update-1):
                    print('[reward learning] epoch: ' + str(epoch) + ', train_acc: ' + str(total_acc))
                if total_acc > 0.97:
                    print('[reward learning] epoch: ' + str(epoch) + ', train_acc: ' + str(total_acc))
                    break;

            bce_loss = []
            print_bce = ""
            for member in range(self.reward_model.de):
                log_loss = np.mean(np.array(ensemble_losses[member]))
                bce_loss.append(log_loss)
                print_bce += "BCE "+str(member) +": " + str(log_loss) +", "

            print(print_bce)
            print(train_acc)
        
    def reset_and_train(self):
        assert self.cfg.rreset == 3
        print(f'resetting {int(self.cfg.rreset)} networks...')
        self.reward_model.reset_parameters()
        self.reward_model.replay_loader.dataset.try_fetch_instant()
        debug_loss1, debug_loss2, bce_loss, train_acc = 0, 0, 0, 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0:
                    ensemble_losses, debug_loss1, debug_loss2, train_acc = self.reward_model.train_soft_reward()
                else:
                    ensemble_losses, debug_loss1, debug_loss2, train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if epoch % 10 == 0:
                    print('[reward learning] epoch: ' + str(epoch) + ', train_acc: ' + str(total_acc))
                if total_acc > 0.97:
                    print('[reward learning] epoch: ' + str(epoch) + ', train_acc: ' + str(total_acc))
                    break;

            bce_loss = []
            print_bce = ""
            for member in range(self.reward_model.de):
                log_loss = np.mean(np.array(ensemble_losses[member]))
                bce_loss.append(log_loss)
                print_bce += "BCE "+str(member) +": " + str(log_loss) +", "

            print(print_bce)
            print(train_acc)
            
    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        # seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        tgt_feat = []

        episode_step, episode_reward, true_episode_reward = 0, 0, 0
        if self.log_success:
            episode_success = 0
        time_step = self.train_env.reset() # this corresponds to (obs_0, None, None)
        self.replay_storage.add(time_step)
        if self.global_step <= (self.cfg.num_seed_frames + self.cfg.num_unsup_frames) // self.cfg.action_repeat:
            with torch.no_grad():
                tgt_feat.append(self.random_encoder(torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0).float()))
        if 'metaworld' not in self.cfg.task_name:
            self.train_video_recorder.init(time_step.observation)
        metrics = None

        observations = deque(maxlen=self.cfg.frame_stack)
        for _ in range(self.cfg.frame_stack):
            observations.append(time_step.observation)

        interact_count = 0
        feedback_end = False
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if 'metaworld' not in self.cfg.task_name:
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("true_episode_reward", true_episode_reward)
                        if self.log_success:
                            log("episode_success", episode_success)
                        log('total_feedback', self.total_feedback)
                        log('labeled_feedback', self.labeled_feedback)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset() # this corresponds to (obs_0, None, None)
                self.replay_storage.add(time_step)
                if self.global_step <= (self.cfg.num_seed_frames + self.cfg.num_unsup_frames) // self.cfg.action_repeat:
                    with torch.no_grad():
                        tgt_feat.append(self.random_encoder(torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0).float()))
                if 'metaworld' not in self.cfg.task_name:
                    self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                # if self.cfg.save_snapshot:
                #     self.save_snapshot()
                self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0

                observations = deque(maxlen=self.cfg.frame_stack)
                for _ in range(self.cfg.frame_stack):
                    observations.append(time_step.observation)

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    np.concatenate(observations, 0), self.global_step, eval_mode=False
                )

            # run training update                
            if self.global_step == (self.cfg.num_seed_frames + self.cfg.num_unsup_frames) // self.cfg.action_repeat:
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_frames-self.global_step) / self.cfg.num_train_frames
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_frames / (self.cfg.num_train_frames-self.global_step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_storage.relabel_with_predictor(self.reward_model, stack=self.cfg.frame_stack if self.cfg.reward_stack else 1, batch_size=256)
                self.replay_loader = make_replay_loader(
                    self.work_dir / "buffer",
                    self.cfg.replay_buffer_size,
                    self.cfg.batch_size,
                    self.cfg.replay_buffer_num_workers,
                    True,
                    self.cfg.frame_stack,
                    self.cfg.nstep,
                    self.cfg.discount,
                )
                self._replay_iter = None
                self.reward_model.replay_loader = self.replay_loader
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()

                # update agent
                metrics = self.agent.update_after_reset(
                    self.replay_iter, self.global_step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                
                # reset interact_count
                interact_count = 0
            elif self.global_step > (self.cfg.num_seed_frames + self.cfg.num_unsup_frames) // self.cfg.action_repeat:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact // self.cfg.action_repeat:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_frames-self.global_step) / self.cfg.num_train_frames
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_frames / (self.cfg.num_train_frames-self.global_step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_storage.relabel_with_predictor(self.reward_model, stack=self.cfg.frame_stack if self.cfg.reward_stack else 1, batch_size=256)
                        self.replay_loader = make_replay_loader(
                            self.work_dir / "buffer",
                            self.cfg.replay_buffer_size,
                            self.cfg.batch_size,
                            self.cfg.replay_buffer_num_workers,
                            True,
                            self.cfg.frame_stack,
                            self.cfg.nstep,
                            self.cfg.discount,
                        )
                        self._replay_iter = None
                        self.reward_model.replay_loader = self.replay_loader
                        interact_count = 0
                        
                if (self.global_frame in self.feedback_period) and (self.agent.last_critic_val > self.reset_threshold_q) and self.cfg.qreset:
                    print(f'best critic output: {self.agent.last_critic_val}, current_threshold: {self.reset_threshold_q}, increase with {self.increase_q}')
                    self.reset_and_train()
                    
                    self.replay_storage.relabel_with_predictor(self.reward_model, stack=self.cfg.frame_stack if self.cfg.reward_stack else 1, batch_size=256)
                    self.replay_loader = make_replay_loader(
                        self.work_dir / "buffer",
                        self.cfg.replay_buffer_size,
                        self.cfg.batch_size,
                        self.cfg.replay_buffer_num_workers,
                        # True if not feedback_end else self.cfg.save_snapshot,
                        True,
                        self.cfg.frame_stack,
                        self.cfg.nstep,
                        self.cfg.discount,
                    )
                    self._replay_iter = None
                    self.reward_model.replay_loader = self.replay_loader
                    
                    self.agent.reset_critic()
                    for _iter in range(self.replay_ratio):
                        metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    
                    self.agent.last_critic_val= -1e-6
                    self.reset_threshold_q += self.increase_q
                    self.reset_threshold_q = min(self.reset_threshold_q, 100.0)
                    self.increase_q = self.increase_q * self.cfg.step_k
                    self.replay_ratio += 1
                    self.replay_ratio = min(self.replay_ratio, self.cfg.max_rr)
                    
                else:
                    for _iter in range(self.replay_ratio if self.global_frame <= self.last_feedback_session_time else 1):
                        metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty="train")

                ##
                if (self.total_feedback >= self.cfg.max_feedback) and (not feedback_end):
                    self.replay_loader = make_replay_loader(
                        self.work_dir / "buffer",
                        self.cfg.replay_buffer_size,
                        self.cfg.batch_size,
                        self.cfg.replay_buffer_num_workers,
                        self.cfg.save_snapshot,
                        self.cfg.frame_stack,
                        self.cfg.nstep,
                        self.cfg.discount,
                    )
                    self._replay_iter = None
                    self.reward_model.replay_loader = self.replay_loader
                    feedback_end = True
                ##
                
            # unsupervised exploration
            elif self.global_step > self.cfg.num_seed_frames // self.cfg.action_repeat:
                metrics = self.agent.update_state_ent(self.replay_iter, self.random_encoder, torch.cat(tgt_feat, dim=0), self.global_step, gradient_update=1, K=self.cfg.topK)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action) # this corresponds to (obs_i, action_i, reward_i)
            if self.cfg.reward_stack:
                reward_hat = self.reward_model.r_hat(np.expand_dims(np.concatenate(observations, 0), axis=0), np.expand_dims(action, axis=0))
            else:
                reward_hat = self.reward_model.r_hat(np.expand_dims(observations[-1], axis=0), np.expand_dims(action, axis=0)) ## w/o stacking

            episode_reward += reward_hat
            true_episode_reward += time_step.reward
            if self.log_success:
                episode_success = max(episode_success, time_step['success'])

            time_step = time_step._replace(true_reward = time_step.reward)
            time_step = time_step._replace(reward = reward_hat)

            self.replay_storage.add(time_step)
            if self.global_step <= (self.cfg.num_seed_frames + self.cfg.num_unsup_frames) // self.cfg.action_repeat:
                with torch.no_grad():
                    tgt_feat.append(self.random_encoder(torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0).float()))
            
            if 'metaworld' not in self.cfg.task_name:
                self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            interact_count += 1
            observations.append(time_step.observation)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="config", config_name="config_policer")
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()