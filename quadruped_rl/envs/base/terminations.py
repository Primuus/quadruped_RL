"""四足机器人通用 episode 终止条件。"""

import torch


def update_quadruped_termination(env):
    """根据接触、姿态和超时更新 reset buffer。"""
    env.reset_buf = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
    env.reset_buf |= torch.logical_or(torch.abs(env.rpy[:, 1]) > 1.0, torch.abs(env.rpy[:, 0]) > 0.8)
    env.time_out_buf = env.episode_length_buf > env.max_episode_length
    env.reset_buf |= env.time_out_buf
