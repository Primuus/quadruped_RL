import torch


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
