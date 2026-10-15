"""Primuus 环境入口；观测、奖励和控制循环使用通用 base。"""

from quadruped_rl.envs.base.legged_robot import LeggedRobot


class PrimuusEnv(LeggedRobot):
    """独立机器人适配器，不继承或导入 Go2 环境。"""

    pass
