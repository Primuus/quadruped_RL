"""Go2 域随机化与外推扰动逻辑。"""

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float


def apply_go2_safety_cfg(env, cfg):
    """应用阶段 2 的安全开关，默认屏蔽尚未验证的高风险功能。"""
    safety = getattr(cfg, "safety", None)
    env.enable_foot_clearance = bool(getattr(safety, "enable_foot_clearance", False))
    env.use_rigid_body_state = bool(getattr(safety, "enable_rigid_body_state_rewards", False))
    env.enable_joint_power = bool(getattr(safety, "enable_joint_power", False))
    env.enable_smoothness = bool(getattr(safety, "enable_smoothness", False))
    env.enable_noise = bool(getattr(safety, "enable_noise", False))
    env.enable_domain_rand = bool(getattr(safety, "enable_domain_rand", False))
    env.enable_push = bool(getattr(safety, "enable_push", False))
    env.enable_height_scan = bool(getattr(safety, "enable_height_scan", False))
    env.enable_command_curriculum = bool(getattr(safety, "enable_command_curriculum", False))
    env.enable_history_obs = bool(getattr(safety, "enable_history_obs", False))
    env.enable_him_actor_critic = bool(getattr(safety, "enable_him_actor_critic", False))
    env.enable_himppo = bool(getattr(safety, "enable_himppo", False))

    if env.enable_himppo:
        raise NotImplementedError("HIMPPO belongs to stage 6 and is not enabled yet")

    cfg.noise.add_noise = cfg.noise.add_noise and env.enable_noise
    cfg.domain_rand.randomize_friction = cfg.domain_rand.randomize_friction and env.enable_domain_rand
    cfg.domain_rand.randomize_base_mass = cfg.domain_rand.randomize_base_mass and env.enable_domain_rand
    cfg.domain_rand.push_robots = cfg.domain_rand.push_robots and env.enable_push
    cfg.terrain.measure_heights = cfg.terrain.measure_heights and env.enable_height_scan
    cfg.commands.curriculum = cfg.commands.curriculum and env.enable_command_curriculum


def process_go2_rigid_shape_props(env, props, env_id):
    """按配置随机化地面摩擦。"""
    if env.cfg.domain_rand.randomize_friction:
        if env_id == 0:
            friction_range = env.cfg.domain_rand.friction_range
            num_buckets = 64
            bucket_ids = torch.randint(0, num_buckets, (env.num_envs, 1))
            friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
            env.friction_coeffs = friction_buckets[bucket_ids]

        for shape_id in range(len(props)):
            props[shape_id].friction = env.friction_coeffs[env_id]
    return props


def process_go2_rigid_body_props(env, props, env_id):
    """按配置随机化机身质量。"""
    if env.cfg.domain_rand.randomize_base_mass:
        mass_range = env.cfg.domain_rand.added_mass_range
        props[0].mass += np.random.uniform(mass_range[0], mass_range[1])
    return props


def push_go2_robots(env):
    """对部分环境施加随机平面速度扰动。"""
    env_ids = torch.arange(env.num_envs, device=env.device)
    push_env_ids = env_ids[env.episode_length_buf[env_ids] % int(env.cfg.domain_rand.push_interval) == 0]
    if len(push_env_ids) == 0:
        return

    max_vel = env.cfg.domain_rand.max_push_vel_xy
    env.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (env.num_envs, 2), device=env.device)

    env_ids_int32 = push_env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
