"""Go2 环境适配入口。"""

from quadruped_rl.envs.base.legged_robot import LeggedRobot


class Go2Env(LeggedRobot):
    """当前 Go2 任务适配器；通用四足逻辑在 LeggedRobot 中。"""

    pass
