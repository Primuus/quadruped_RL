# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HIMEstimator(nn.Module):
    """HIMLoco 风格的历史观测编码器。

    由 HIMPPO 调用 update() 训练速度估计和 latent 表征。
    """

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=(128, 64, 16),
        tar_hidden_dims=(128, 64),
        activation='elu',
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=32,
        temperature=3.0,
        **kwargs,
    ):
        if kwargs:
            print("HIMEstimator.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()
        activation_fn = get_activation(activation)

        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_latent = int(enc_hidden_dims[-1])
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers += [nn.Linear(enc_input_dim, hidden_dim), activation_fn]
            enc_input_dim = hidden_dim
        enc_layers += [nn.Linear(enc_input_dim, self.num_latent + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for hidden_dim in tar_hidden_dims:
            tar_layers += [nn.Linear(tar_input_dim, hidden_dim), activation_fn]
            tar_input_dim = hidden_dim
        tar_layers += [nn.Linear(tar_input_dim, self.num_latent)]
        self.target = nn.Sequential(*tar_layers)

        self.proto = nn.Embedding(num_prototype, self.num_latent)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, latent = parts[..., :3], parts[..., 3:]
        latent = F.normalize(latent, dim=-1, p=2)
        return vel.detach(), latent.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, latent = parts[..., :3], parts[..., 3:]
        latent = F.normalize(latent, dim=-1, p=2)
        return vel, latent

    def get_latent(self, obs_history):
        vel, latent = self.encode(obs_history)
        return vel.detach(), latent.detach()

    def update(self, obs_history, next_critic_obs, lr=None):
        """HIMPPO 使用的 estimator 更新接口。"""
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs + 3]

        encoded = self.encoder(obs_history)
        target_latent = self.target(next_obs)
        pred_vel, encoded_latent = encoded[..., :3], encoded[..., 3:]

        encoded_latent = F.normalize(encoded_latent, dim=-1, p=2)
        target_latent = F.normalize(target_latent, dim=-1, p=2)

        with torch.no_grad():
            proto_weight = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
            self.proto.weight.copy_(proto_weight)

        score_s = encoded_latent @ self.proto.weight.T
        score_t = target_latent @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    q = torch.exp(out / eps).T
    num_prototypes, batch_size = q.shape
    q /= q.sum()

    for _ in range(iters):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= num_prototypes
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch_size
    return (q * batch_size).T


def get_activation(act_name):
    if act_name == 'elu':
        return nn.ELU()
    if act_name == 'selu':
        return nn.SELU()
    if act_name == 'relu':
        return nn.ReLU()
    if act_name == 'crelu':
        return nn.ReLU()
    if act_name == 'silu':
        return nn.SiLU()
    if act_name == 'lrelu':
        return nn.LeakyReLU()
    if act_name == 'tanh':
        return nn.Tanh()
    if act_name == 'sigmoid':
        return nn.Sigmoid()
    raise ValueError(f'Invalid activation function: {act_name}')
