# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.him_estimator import HIMEstimator


class HIMActorCritic(nn.Module):
    """HIMLoco 风格 actor-critic。

    外部 actor 输入仍是 45*6 历史观测；内部 estimator 输出 vel3 + latent16 后，
    actor 实际输入为 当前45维 + vel3 + latent16 = 64维。
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation='elu',
        init_noise_std=1.0,
        estimator_hidden_dims=(128, 64, 16),
        estimator_target_hidden_dims=(128, 64),
        estimator_learning_rate=1e-3,
        estimator_max_grad_norm=10.0,
        estimator_num_prototype=32,
        estimator_temperature=3.0,
        **kwargs,
    ):
        if kwargs:
            print("HIMActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()

        activation_fn = get_activation(activation)
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_actions = int(num_actions)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_estimator_outputs = 3 + int(estimator_hidden_dims[-1])

        if self.history_size * self.num_one_step_obs != self.num_actor_obs:
            raise ValueError(
                f'HIMActorCritic expects num_actor_obs to be a multiple of num_one_step_obs, '
                f'got {num_actor_obs} and {num_one_step_obs}'
            )

        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            enc_hidden_dims=tuple(estimator_hidden_dims),
            tar_hidden_dims=tuple(estimator_target_hidden_dims),
            activation=activation,
            learning_rate=estimator_learning_rate,
            max_grad_norm=estimator_max_grad_norm,
            num_prototype=estimator_num_prototype,
            temperature=estimator_temperature,
        )

        actor_input_dim = self.num_one_step_obs + self.num_estimator_outputs
        critic_input_dim = num_critic_obs

        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator: {self.estimator.encoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _build_actor_input(self, obs_history):
        vel, latent = self.estimator(obs_history)
        return torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1)

    def update_distribution(self, obs_history):
        actor_input = self._build_actor_input(obs_history)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        return self.actor(self._build_actor_input(obs_history))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
