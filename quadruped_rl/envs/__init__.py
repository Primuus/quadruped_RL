from quadruped_rl.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .base.legged_robot import LeggedRobot

from quadruped_rl.utils.task_registry import task_registry

task_registry.register("go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
