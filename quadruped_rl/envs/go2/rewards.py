"""Go2 reward 开关整理。

这里不改变 reward 公式，只集中管理哪些 reward 需要经过安全开关验证。
"""


GO2_GUARDED_REWARDS = {
    "foot_clearance": "enable_foot_clearance",  # 足端抬脚 reward，依赖足端刚体状态，默认关闭
    "joint_power": "enable_joint_power",  # 关节功率惩罚，默认关闭
    "smoothness": "enable_smoothness",  # 二阶动作平滑惩罚，默认关闭
}


def apply_go2_reward_safety_overrides(reward_scales, env):
    """根据 Go2 安全开关关闭尚未验证的 reward 项。"""
    for reward_name, flag_name in GO2_GUARDED_REWARDS.items():
        if not bool(getattr(env, flag_name, False)):
            reward_scales[reward_name] = 0.0
