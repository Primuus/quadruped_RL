"""四足机器人通用命令采样与课程学习逻辑。"""

import math
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, torch_rand_float

from quadruped_rl.utils.math import wrap_to_pi


def validate_command_sampling(cfg):
    """校验可选的轴向平移和纯前进分段采样配置。"""
    axis_probability = getattr(cfg, "translation_axis_probability", 0.0)
    if not math.isfinite(axis_probability) or not 0.0 <= axis_probability <= 1.0:
        raise ValueError("commands.translation_axis_probability must be in [0, 1]")
    if axis_probability > 0:
        if "translate" not in getattr(cfg, "mode_names", ("turn", "moving_turn", "translate", "stand")):
            raise ValueError("Axis sampling requires translate mode")
        axis_limits = (-cfg.ranges.lin_vel_x[0], cfg.ranges.lin_vel_y[1], -cfg.ranges.lin_vel_y[0])
        if any(not math.isfinite(limit) or limit < cfg.min_translation_speed for limit in axis_limits):
            raise ValueError("Axis sampling requires backward and both lateral ranges to fit min_translation_speed")

    bins = getattr(cfg, "forward_speed_bins", None)
    probabilities = getattr(cfg, "forward_speed_probabilities", None)
    if bins is None and probabilities is None:
        return
    if "forward" not in getattr(cfg, "mode_names", ()) or bins is None or probabilities is None:
        raise ValueError("Forward bins require forward mode and matching probabilities")
    if not bins or len(bins) != len(probabilities):
        raise ValueError("Forward bins and probabilities must have matching nonzero lengths")
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Forward speed probabilities must be finite, nonnegative and sum to 1")
    lower, upper = cfg.ranges.lin_vel_forward
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 < lower < upper:
        raise ValueError("Forward speed range must be finite, positive and ordered")
    edge = lower
    for interval in bins:
        if (len(interval) != 2 or any(not math.isfinite(x) for x in interval)
                or not math.isclose(interval[0], edge) or interval[1] <= interval[0]):
            raise ValueError("Forward speed bins must be ordered, contiguous and cover lin_vel_forward")
        edge = interval[1]
    if not math.isclose(edge, upper):
        raise ValueError("Forward speed bins must cover lin_vel_forward")


def post_quadruped_command_step(env):
    """按配置重采样速度命令，并根据 heading 生成偏航速度命令。"""
    env_ids = (env.episode_length_buf % int(env.cfg.commands.resampling_time / env.dt) == 0).nonzero(as_tuple=False).flatten()
    resample_quadruped_commands(env, env_ids)
    if env.cfg.commands.heading_command:
        forward = quat_apply(env.base_quat, env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(env.commands[:, 3] - heading), -1., 1.)


def resample_quadruped_commands(env, env_ids):
    """采样基础四类运动命令，并支持 Primuus 的独立纯前进状态。"""
    if len(env_ids) == 0:
        return

    probabilities = torch.as_tensor(env.cfg.commands.mode_probabilities, device=env.device)
    mode_names = getattr(env.cfg.commands, "mode_names", ("turn", "moving_turn", "translate", "stand"))
    if probabilities.shape != (len(mode_names),) or torch.any(probabilities < 0) or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0)):
        raise ValueError("commands.mode_probabilities must match mode_names and sum to 1 with nonnegative values")
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
    axis_probability = getattr(env.cfg.commands, "translation_axis_probability", 0.0)
    if axis_probability > 0:
        # Only replace translate samples; moving-turn keeps its original XY distribution.
        axis_mask = (modes == mode_names.index("translate")) & (torch.rand(len(env_ids), device=env.device) < axis_probability)
        axes = torch.randint(3, (len(env_ids),), device=env.device)
        directions = linear.new_tensor([[-1., 0.], [0., 1.], [0., -1.]])[axes]
        limits = linear.new_tensor([
            -env.command_ranges["lin_vel_x"][0],
            env.command_ranges["lin_vel_y"][1],
            -env.command_ranges["lin_vel_y"][0],
        ])[axes].clamp(max=env.cfg.commands.max_translation_speed)
        upper = (limits * sampling_scale).clamp(min=env.cfg.commands.min_translation_speed)
        magnitude = env.cfg.commands.min_translation_speed + (upper - env.cfg.commands.min_translation_speed) * torch.rand(len(env_ids), device=env.device)
        linear = torch.where(axis_mask.unsqueeze(1), directions * magnitude.unsqueeze(1), linear)
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

    if "forward" in mode_names:
        # 纯前进独立采样，不受其他平移状态的 1 m/s 合速度限幅。
        min_forward, max_forward = env.command_ranges["lin_vel_forward"]
        bins = getattr(env.cfg.commands, "forward_speed_bins", None)
        if bins is not None:
            # Do not scale these samples again: each bin has an explicit quota.
            bin_probs = linear.new_tensor(env.cfg.commands.forward_speed_probabilities)
            bin_ids = torch.multinomial(bin_probs, len(env_ids), replacement=True)
            intervals = linear.new_tensor(bins)[bin_ids]
            forward_speed = intervals[:, 0] + (intervals[:, 1] - intervals[:, 0]) * torch.rand(len(env_ids), device=env.device)
        else:
            forward_upper = (max_forward * sampling_scale).clamp(min=min_forward)
            forward_speed = min_forward + (forward_upper - min_forward) * torch.rand(len(env_ids), device=env.device)
        env.commands[env_ids, 0] = torch.where(
            modes == mode_names.index("forward"), forward_speed, env.commands[env_ids, 0]
        )

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
