"""四足机器人通用命令采样与课程学习逻辑。"""

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, torch_rand_float

from quadruped_rl.utils.math import wrap_to_pi


def post_quadruped_command_step(env):
    """按配置重采样速度命令，并根据 heading 生成偏航速度命令。"""
    env_ids = (env.episode_length_buf % int(env.cfg.commands.resampling_time / env.dt) == 0).nonzero(as_tuple=False).flatten()
    resample_quadruped_commands(env, env_ids)
    if env.cfg.commands.heading_command:
        forward = quat_apply(env.base_quat, env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(env.commands[:, 3] - heading), -1., 1.)


def resample_quadruped_commands(env, env_ids):
    """随机采样四足机器人速度/朝向命令。"""
    env.commands[env_ids, 0] = torch_rand_float(
        env.command_ranges["lin_vel_x"][0],
        env.command_ranges["lin_vel_x"][1],
        (len(env_ids), 1),
        device=env.device,
    ).squeeze(1)
    env.commands[env_ids, 1] = torch_rand_float(
        env.command_ranges["lin_vel_y"][0],
        env.command_ranges["lin_vel_y"][1],
        (len(env_ids), 1),
        device=env.device,
    ).squeeze(1)
    if env.cfg.commands.heading_command:
        env.commands[env_ids, 3] = torch_rand_float(
            env.command_ranges["heading"][0],
            env.command_ranges["heading"][1],
            (len(env_ids), 1),
            device=env.device,
        ).squeeze(1)
    else:
        env.commands[env_ids, 2] = torch_rand_float(
            env.command_ranges["ang_vel_yaw"][0],
            env.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=env.device,
        ).squeeze(1)

    # 很小的平面速度命令置零，避免机器人在近零命令下学到抖动。
    env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


def update_quadruped_command_curriculum(env, env_ids):
    """根据线速度跟踪表现扩展前向速度命令范围。"""
    tracking_ratio = torch.mean(env.episode_sums["tracking_lin_vel"][env_ids]) / env.max_episode_length
    if tracking_ratio > 0.8 * env.reward_scales["tracking_lin_vel"]:
        env.command_ranges["lin_vel_x"][0] = np.clip(
            env.command_ranges["lin_vel_x"][0] - 0.5,
            -env.cfg.commands.max_curriculum,
            0.,
        )
        env.command_ranges["lin_vel_x"][1] = np.clip(
            env.command_ranges["lin_vel_x"][1] + 0.5,
            0.,
            env.cfg.commands.max_curriculum,
        )
