"""Go2 任务的 reward 安全开关兼容封装。

实际实现放在四足通用 base 模块中，Go2 这里只保留旧名称，避免已有导入失效。
"""

from quadruped_rl.envs.base.rewards import (
    QUADRUPED_GUARDED_REWARDS,
    apply_quadruped_reward_safety_overrides,
)

GO2_GUARDED_REWARDS = QUADRUPED_GUARDED_REWARDS
apply_go2_reward_safety_overrides = apply_quadruped_reward_safety_overrides
