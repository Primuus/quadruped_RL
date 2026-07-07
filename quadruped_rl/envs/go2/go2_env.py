"""Go2 环境入口。

当前阶段沿用稳定的 LeggedRobot 实现，后续 Go2 专用逻辑逐步迁移到本目录。
"""

from quadruped_rl.envs.base.legged_robot import LeggedRobot


class Go2Env(LeggedRobot):
    """Go2-only 训练环境。"""

    pass
