"""Go2 任务的 reward 依赖适配封装。

实际实现放在四足通用 base 模块中，Go2 这里只提供机器人命名空间下的别名。
"""

from quadruped_rl.envs.base.rewards import (
    QUADRUPED_RIGID_BODY_STATE_REWARDS,
    reward_requires_rigid_body_state,
)

GO2_RIGID_BODY_STATE_REWARDS = QUADRUPED_RIGID_BODY_STATE_REWARDS
go2_reward_requires_rigid_body_state = reward_requires_rigid_body_state
