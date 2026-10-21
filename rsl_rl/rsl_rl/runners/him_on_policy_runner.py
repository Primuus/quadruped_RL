# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import statistics
import time
from collections import deque

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from rsl_rl.algorithms import HIMPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import HIMActorCritic


class HIMOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: HIMActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            if SummaryWriter is None:
                raise ImportError("tensorboard is required when log_dir is enabled")
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iteration = self.current_learning_iteration
        start_total_time = self.tot_time
        tot_iter = start_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            command_metrics = {}
            motion_metrics = {}
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    termination_ids = infos.get("termination_ids")
                    if termination_ids is None:
                        termination_ids = torch.empty(0, dtype=torch.long, device=self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = infos.get("termination_privileged_obs")
                    if termination_privileged_obs is None:
                        termination_privileged_obs = torch.empty(0, critic_obs.shape[1], device=self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)
                    if len(termination_ids) > 0:
                        next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        for name, values in infos.get('command_metrics', {}).items():
                            if name not in command_metrics:
                                command_metrics[name] = torch.zeros_like(values)
                            command_metrics[name] += values
                        for name, values in infos.get('motion_metrics', {}).items():
                            if name not in motion_metrics:
                                motion_metrics[name] = torch.zeros_like(values)
                            motion_metrics[name] += values
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            self.current_learning_iteration = it + 1
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        completed_iteration = locs['it'] + 1
        completed_in_call = completed_iteration - locs['start_iteration']
        remaining_iterations = locs['tot_iter'] - completed_iteration
        eta = (self.tot_time - locs['start_total_time']) / completed_in_call * remaining_iterations

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, completed_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        for name, values in locs['command_metrics'].items():
            count = values[0].item()
            if count == 0:
                continue
            prefix = f'Command/{name}'
            self.writer.add_scalar(prefix + '/fraction', count / (self.num_steps_per_env * self.env.num_envs), completed_iteration)
            self.writer.add_scalar(prefix + '/yaw_abs_error', (values[1] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/linear_error', (values[2] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/valid_steps_per_second', (values[3] / (count * self.env.dt)).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/foot_slip_speed', (values[4] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/feet_in_contact', (values[5] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/tilt', (values[6] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/base_height', (values[7] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/joint_pose_error', (values[8] / count).item(), completed_iteration)
            self.writer.add_scalar(prefix + '/supporting_feet', (values[9] / count).item(), completed_iteration)

        curriculum_metrics = getattr(self.env, "get_command_curriculum_metrics", None)
        if curriculum_metrics is not None:
            for name, value in curriculum_metrics().items():
                self.writer.add_scalar('Curriculum/' + name, value, completed_iteration)

        mean_std = self.alg.actor_critic.std.mean()
        for name, values in locs['motion_metrics'].items():
            if name in ('Transition/interrupted', 'Transition/censored'):
                self.writer.add_scalar(name, values[1].item(), completed_iteration)
            elif values.ndim == 2:
                for direction, pair in enumerate(values):
                    if pair[0] > 0:
                        self.writer.add_scalar(name + '/direction_' + str(direction), (pair[1] / pair[0]).item(), completed_iteration)
            elif values[0] > 0:
                self.writer.add_scalar(name, (values[1] / values[0]).item(), completed_iteration)
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], completed_iteration)
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], completed_iteration)
        self.writer.add_scalar('Loss/estimation', locs['mean_estimation_loss'], completed_iteration)
        self.writer.add_scalar('Loss/swap', locs['mean_swap_loss'], completed_iteration)
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, completed_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), completed_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, completed_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], completed_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], completed_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), completed_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), completed_iteration)

        header = f" \033[1m Learning iteration {completed_iteration}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, "
            f"learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
            f"{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"
            f"{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        )
        if len(locs['rewbuffer']) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {eta:.1f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        state = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        get_env_state = getattr(self.env, "get_training_state", None)
        if get_env_state is not None:
            state['env_state'] = get_env_state()
        torch.save(state, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
            if 'estimator_optimizer_state_dict' in loaded_dict:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
                self.alg.actor_critic.estimator.learning_rate = self.alg.actor_critic.estimator.optimizer.param_groups[0]['lr']
        self.current_learning_iteration = loaded_dict['iter']
        load_env_state = getattr(self.env, "load_training_state", None)
        if load_env_state is not None:
            load_env_state(loaded_dict.get('env_state'))
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
