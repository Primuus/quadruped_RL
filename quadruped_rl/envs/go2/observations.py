from quadruped_rl.envs.base.observations import (
    QUADRUPED_HISTORY_LENGTH,
    QUADRUPED_HISTORY_OBS_SIZE,
    QUADRUPED_ONE_STEP_OBS_SIZE,
    QUADRUPED_PRIVILEGED_OBS_SIZE,
    build_him_history_observations,
    build_him_one_step_observations,
    build_him_privileged_observations,
    reset_observation_history,
)

GO2_ONE_STEP_OBS_SIZE = QUADRUPED_ONE_STEP_OBS_SIZE
GO2_HISTORY_LENGTH = QUADRUPED_HISTORY_LENGTH
GO2_HISTORY_OBS_SIZE = QUADRUPED_HISTORY_OBS_SIZE
GO2_PRIVILEGED_OBS_SIZE = QUADRUPED_PRIVILEGED_OBS_SIZE


def build_go2_45_observations(env):
    """Go2 兼容封装：实际逻辑在四足通用观测模块中。"""
    return build_him_one_step_observations(env)


def build_go2_observations(env):
    """Go2 兼容封装：构造 45*6 history policy 观测。"""
    return build_him_history_observations(env)


def build_go2_privileged_observations(env, current_obs=None):
    """Go2 兼容封装：构造 238 维 critic privileged obs。"""
    return build_him_privileged_observations(env, current_obs)


def reset_go2_history(env, env_ids):
    """Go2 兼容封装：重置 history buffer。"""
    reset_observation_history(env, env_ids)
