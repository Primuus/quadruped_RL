import torch

GO2_ONE_STEP_OBS_SIZE = 45
GO2_HISTORY_LENGTH = 6
GO2_HISTORY_OBS_SIZE = GO2_ONE_STEP_OBS_SIZE * GO2_HISTORY_LENGTH
GO2_PRIVILEGED_OBS_SIZE = GO2_ONE_STEP_OBS_SIZE + 3


def build_go2_45_observations(env):
    """构造 Go2 45 维单步观测，训练侧和部署侧必须保持同一顺序。"""
    return torch.cat((
        env.commands[:, :3] * env.commands_scale,
        env.base_ang_vel * env.obs_scales.ang_vel,
        env.projected_gravity,
        (env.dof_pos - env.default_dof_pos) * env.obs_scales.dof_pos,
        env.dof_vel * env.obs_scales.dof_vel,
        env.actions,
    ), dim=-1)


def build_go2_observations(env):
    """构造当前阶段的 Go2 policy 观测。"""
    current_obs = build_go2_45_observations(env)
    env.current_obs_buf = current_obs
    if not env.enable_history_obs:
        return current_obs

    env.obs_history_buf = torch.cat(
        (current_obs, env.obs_history_buf[:, :-env.num_one_step_obs]),
        dim=-1,
    )
    return env.obs_history_buf


def build_go2_privileged_observations(env, current_obs=None):
    """构造 Go2 critic privileged obs：当前 45 维观测 + 机身线速度。"""
    if current_obs is None:
        current_obs = build_go2_45_observations(env)
    return torch.cat((
        current_obs,
        env.base_lin_vel * env.obs_scales.lin_vel,
    ), dim=-1)


def reset_go2_history(env, env_ids):
    """重置指定环境的历史观测，避免跨 episode 泄漏。"""
    if env.enable_history_obs and len(env_ids) > 0:
        env.obs_history_buf[env_ids] = 0.
