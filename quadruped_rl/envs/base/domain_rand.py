"""四足机器人通用域随机化与外推扰动逻辑。"""

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float


def process_quadruped_rigid_shape_props(env, props, env_id):
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


def process_quadruped_rigid_body_props(env, props, env_id):
    """按配置随机化机身质量。"""
    if env.cfg.domain_rand.randomize_base_mass:
        mass_range = env.cfg.domain_rand.added_mass_range
        props[0].mass += np.random.uniform(mass_range[0], mass_range[1])
    return props


def push_quadruped_robots(env):
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
