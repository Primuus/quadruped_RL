"""四足机器人 HIMLoco 风格观测构造。

这里放通用观测接口；具体机器人只提供关节顺序、默认角、资产和配置。
"""

import torch

QUADRUPED_ONE_STEP_OBS_SIZE = 45
QUADRUPED_HISTORY_LENGTH = 6
QUADRUPED_HISTORY_OBS_SIZE = QUADRUPED_ONE_STEP_OBS_SIZE * QUADRUPED_HISTORY_LENGTH
QUADRUPED_PRIVILEGED_BASE_LIN_VEL_SIZE = 3
QUADRUPED_PRIVILEGED_EXTERNAL_FORCE_SIZE = 3
QUADRUPED_PRIVILEGED_HEIGHT_SCAN_SIZE = 187
QUADRUPED_PRIVILEGED_OBS_SIZE = (
    QUADRUPED_ONE_STEP_OBS_SIZE
    + QUADRUPED_PRIVILEGED_BASE_LIN_VEL_SIZE
    + QUADRUPED_PRIVILEGED_EXTERNAL_FORCE_SIZE
    + QUADRUPED_PRIVILEGED_HEIGHT_SCAN_SIZE
)


def build_him_one_step_observations(env):
    """构造 45 维 HIMLoco 单帧 actor 观测，训练侧和部署侧必须保持同一顺序。"""
    obs = torch.cat((
        env.commands[:, :3] * env.commands_scale,
        env.base_ang_vel * env.obs_scales.ang_vel,
        env.projected_gravity,
        (env.dof_pos - env.default_dof_pos) * env.obs_scales.dof_pos,
        env.dof_vel * env.obs_scales.dof_vel,
        env.actions,
    ), dim=-1)
    if obs.shape[1] != env.num_one_step_obs:
        raise RuntimeError(f"HIM one-step obs expects {env.num_one_step_obs} dims, got {obs.shape[1]}")
    return obs


def build_him_history_observations(env):
    """构造 policy 输入观测：当前 45 维观测加 5 帧历史。"""
    current_obs = build_him_one_step_observations(env)
    env.current_obs_buf = current_obs

    env.obs_history_buf = torch.cat(
        (current_obs, env.obs_history_buf[:, :-env.num_one_step_obs]),
        dim=-1,
    )
    return env.obs_history_buf


def _zero_privileged_slice(env, width):
    return torch.zeros(env.num_envs, width, dtype=torch.float, device=env.device, requires_grad=False)


def get_privileged_external_force_observation(env):
    """返回 3 维外力/扰动 privileged obs；未启用扰动时保持 0 占位。"""
    width = getattr(env, "num_privileged_external_force_obs", QUADRUPED_PRIVILEGED_EXTERNAL_FORCE_SIZE)
    external_force_obs = getattr(env, "external_force_obs", None)
    if external_force_obs is None:
        return _zero_privileged_slice(env, width)
    if external_force_obs.shape != (env.num_envs, width):
        raise RuntimeError(f"external_force_obs expects shape ({env.num_envs}, {width}), got {external_force_obs.shape}")
    return external_force_obs


def get_privileged_height_scan_observation(env):
    """返回 187 维地形高度 privileged obs；平地或未启用高度采样时保持 0 占位。"""
    width = getattr(env, "num_privileged_height_scan", QUADRUPED_PRIVILEGED_HEIGHT_SCAN_SIZE)
    height_scan_obs = getattr(env, "privileged_height_scan_obs", None)
    if height_scan_obs is None:
        return _zero_privileged_slice(env, width)
    if height_scan_obs.shape != (env.num_envs, width):
        raise RuntimeError(f"privileged_height_scan_obs expects shape ({env.num_envs}, {width}), got {height_scan_obs.shape}")
    return height_scan_obs


def build_him_privileged_observations(env, current_obs=None):
    """构造 critic privileged obs：45 + base_lin_vel3 + external_force3 + height_scan187 = 238。"""
    if current_obs is None:
        current_obs = build_him_one_step_observations(env)

    privileged_obs = torch.cat((
        current_obs,
        env.base_lin_vel * env.obs_scales.lin_vel,
        get_privileged_external_force_observation(env),
        get_privileged_height_scan_observation(env),
    ), dim=-1)
    expected_dim = getattr(env, "num_one_step_privileged_obs", QUADRUPED_PRIVILEGED_OBS_SIZE)
    if privileged_obs.shape[1] != expected_dim:
        raise RuntimeError(f"HIM privileged obs expects {expected_dim} dims, got {privileged_obs.shape[1]}")
    return privileged_obs


def reset_observation_history(env, env_ids):
    """重置指定环境的历史观测，避免跨 episode 泄漏。"""
    if len(env_ids) > 0:
        env.obs_history_buf[env_ids] = 0.
