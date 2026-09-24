"""四足机器人通用 reward 依赖声明。"""


QUADRUPED_RIGID_BODY_STATE_REWARDS = frozenset({
    "foot_clearance",
    "foot_slip",
    "turn_step",
})


def reward_requires_rigid_body_state(reward_scales):
    """判断当前非零 reward 是否需要刷新刚体状态。"""
    return any(
        float(reward_scales.get(reward_name, 0.0)) != 0.0
        for reward_name in QUADRUPED_RIGID_BODY_STATE_REWARDS
    )
