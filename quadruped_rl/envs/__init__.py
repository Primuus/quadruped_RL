from quadruped_rl.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from quadruped_rl.envs.go2.go2_env import Go2Env
from quadruped_rl.envs.primuus.primuus_config import PrimuusCfg, PrimuusCfgPPO
from quadruped_rl.envs.primuus.primuus_env import PrimuusEnv
from quadruped_rl.envs.primuus.primuus_obstacles_config import PrimuusObstaclesCfg, PrimuusObstaclesCfgPPO
from quadruped_rl.envs.primuus.primuus_obstacles_env import PrimuusObstaclesEnv

from quadruped_rl.utils.task_registry import task_registry

# 各机器人独立注册，只复用 base 环境和 HIM-PPO，不互相继承机器人配置。
task_registry.register("go2", Go2Env, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register("primuus", PrimuusEnv, PrimuusCfg(), PrimuusCfgPPO())
task_registry.register("primuus_obstacles", PrimuusObstaclesEnv, PrimuusObstaclesCfg(), PrimuusObstaclesCfgPPO())
