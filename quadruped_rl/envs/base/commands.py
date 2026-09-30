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
    """按原地转向、边走边转、平移和站立场景采样命令。"""
    if len(env_ids) == 0:
        return

    probabilities = torch.as_tensor(env.cfg.commands.mode_probabilities, device=env.device)
    if probabilities.shape != (4,) or torch.any(probabilities < 0) or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0)):
        raise ValueError("commands.mode_probabilities must contain four nonnegative values summing to 1")
    modes = torch.searchsorted(torch.cumsum(probabilities, dim=0), torch.rand(len(env_ids), device=env.device))
    env.command_modes[env_ids] = modes
    low_speed = torch.rand(len(env_ids), device=env.device) < env.cfg.commands.low_speed_probability
    sampling_scale = torch.where(low_speed, env.cfg.commands.low_speed_scale, 1.0)

    linear_x = torch_rand_float(
        env.command_ranges["lin_vel_x"][0],
        env.command_ranges["lin_vel_x"][1],
        (len(env_ids), 1),
        device=env.device,
    ).squeeze(1)
    linear_y = torch_rand_float(
        env.command_ranges["lin_vel_y"][0],
        env.command_ranges["lin_vel_y"][1],
        (len(env_ids), 1),
        device=env.device,
    ).squeeze(1)
    linear = torch.stack((linear_x, linear_y), dim=1) * sampling_scale.unsqueeze(1)
    linear_speed = torch.linalg.vector_norm(linear, dim=1, keepdim=True)
    target_speed = linear_speed.clamp(
        min=env.cfg.commands.min_translation_speed,
        max=env.cfg.commands.max_translation_speed,
    )
    linear *= target_speed / linear_speed.clamp(min=1e-6)
    env.commands[env_ids, :2] = linear * ((modes == 1) | (modes == 2)).unsqueeze(1)

    yaw_range = env.command_ranges["ang_vel_yaw"]
    max_yaw = min(-yaw_range[0], yaw_range[1], 1.0 if env.cfg.commands.heading_command else float("inf"))
    min_yaw = env.cfg.commands.min_turn_yaw
    if not 0.0 < min_yaw <= max_yaw:
        raise ValueError("commands.min_turn_yaw must fit the positive and negative yaw ranges")
    yaw_upper = (max_yaw * sampling_scale).clamp(min=min_yaw)
    yaw_magnitude = min_yaw + (yaw_upper - min_yaw) * torch.rand(len(env_ids), device=env.device)
    yaw_sign = torch.where(torch.rand(len(env_ids), device=env.device) < 0.5, -1.0, 1.0)
    pure_turn_yaw = yaw_magnitude * yaw_sign
    combined_yaw = torch_rand_float(
        max(yaw_range[0], -1.0) if env.cfg.commands.heading_command else yaw_range[0],
        min(yaw_range[1], 1.0) if env.cfg.commands.heading_command else yaw_range[1],
        (len(env_ids), 1), device=env.device,
    ).squeeze(1) * sampling_scale
    yaw = torch.where(modes == 0, pure_turn_yaw, torch.where(modes == 1, combined_yaw, 0.0))
    env.commands[env_ids, 2] = yaw

    if env.cfg.commands.heading_command:
        forward = quat_apply(env.base_quat[env_ids], env.forward_vec[env_ids])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[env_ids, 3] = wrap_to_pi(heading + yaw / 0.5)
    else:
        env.commands[env_ids, 3] = 0.0


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
